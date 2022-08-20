import attr
from threading import Thread

from time import time

from ....config import deferred_config
from ....backend_interface.task.development.stop_signal import TaskStopSignal
from ....backend_api.services import tasks
from ....utilities.lowlevel.threads import kill_thread
from ....utilities.process.mp import SafeEvent


class DevWorker(object):
    property_abort_callback_completed = "_abort_callback_completed"
    property_abort_callback_timeout = "_abort_callback_timeout"
    property_abort_poll_freq = "_abort_poll_freq"

    prefix = attr.ib(type=str, default="MANUAL:")

    report_stdout = deferred_config('development.worker.log_stdout', True)
    report_period = deferred_config(
        'development.worker.report_period_sec', 30.,
        transform=lambda x: float(max(x, 1.0)))
    ping_period = deferred_config(
        'development.worker.ping_period_sec', 30.,
        transform=lambda x: float(max(x, 1.0)))

    def __init__(self):
        self._dev_stop_signal = None
        self._thread = None
        self._exit_event = SafeEvent()
        self._task = None
        self._support_ping = False
        self._poll_freq = None
        self._abort_cb = None
        self._abort_cb_timeout = None
        self._cb_completed = None

    def ping(self, timestamp=None):
        try:
            if self._task:
                self._task.send(tasks.PingRequest(self._task.id))
        except Exception:  # noqa
            return False
        return True

    def register(self, task, stop_signal_support=None):
        if self._thread:
            return True
        if (stop_signal_support is None and TaskStopSignal.enabled) or stop_signal_support is True:
            self._dev_stop_signal = TaskStopSignal(task=task)
        self._support_ping = hasattr(tasks, 'PingRequest')
        # if there is nothing to monitor, leave
        if not self._support_ping and not self._dev_stop_signal:
            return
        self._task = task
        self._exit_event.clear()
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()
        return True

    def register_abort_callback(self, callback_function, execution_timeout, poll_freq):
        if not self._task:
            return

        self._poll_freq = float(poll_freq) if poll_freq else None
        self._abort_cb = callback_function
        self._abort_cb_timeout = float(execution_timeout)

        if not callback_function:
            # noinspection PyProtectedMember
            self._task._set_runtime_properties({DevWorker.property_abort_callback_timeout: float(-1)})
            return

        # noinspection PyProtectedMember
        self._task._set_runtime_properties({
            self.property_abort_callback_timeout: float(execution_timeout),
            self.property_abort_poll_freq: float(poll_freq),
            self.property_abort_callback_completed: "",
        })

    def _inner_abort_cb_wrapper(self):
        # store the task object because we might nullify it
        task = self._task
        # call the user abort callback
        try:
            if self._abort_cb:
                self._abort_cb()
            self._cb_completed = True
        except SystemError:
            # we will get here if we killed the thread externally,
            # we should not try to mark as completed, just leave the thread
            return
        except BaseException as ex:  # noqa
            if task and task.log:
                task.log.warning(
                    "### TASK STOPPING - USER ABORTED - CALLBACK EXCEPTION: {} ###".format(ex))

        # set runtime property, abort completed for the agent to know we are done
        if task:
            # noinspection PyProtectedMember
            task._set_runtime_properties({self.property_abort_callback_completed: 1})

    def _launch_abort_cb(self):
        timeout = self._abort_cb_timeout or 300.
        if self._task and self._task.log:
            self._task.log.warning(
                "### TASK STOPPING - USER ABORTED - "
                "LAUNCHING CALLBACK (timeout {} sec) ###".format(timeout))

        tic = time()
        timed_out = False
        try:
            callback_thread = Thread(target=self._inner_abort_cb_wrapper)
            callback_thread.daemon = True
            callback_thread.start()
            callback_thread.join(timeout=timeout)
            if callback_thread.is_alive():
                kill_thread(callback_thread, wait=False)
                timed_out = True
        except:  # noqa
            # something went wrong no just leave the process
            pass

        if self._task and self._task.log:
            self._task.log.warning(
                "### TASK STOPPING - USER ABORTED - CALLBACK {} ({:.2f} sec) ###".format(
                    "TIMED OUT" if timed_out else ("COMPLETED" if self._cb_completed else "FAILED"), time()-tic))

    def _daemon(self):
        last_ping = time()
        while self._task is not None:
            try:
                wait_timeout = min(float(self.ping_period), float(self.report_period))
                if self._poll_freq:
                    wait_timeout = min(self._poll_freq, wait_timeout)
                if self._exit_event.wait(wait_timeout):
                    return
                # send ping request
                if self._support_ping and (time() - last_ping) >= float(self.ping_period):
                    self.ping()
                    last_ping = time()

                if self._dev_stop_signal:
                    stop_reason = self._dev_stop_signal.test()
                    if stop_reason and self._task:
                        # call abort callback
                        if self._abort_cb:
                            self._launch_abort_cb()

                        # noinspection PyProtectedMember
                        self._task._dev_mode_stop_task(stop_reason)
            except Exception:  # noqa
                pass

    def unregister(self):
        self._dev_stop_signal = None
        self._task = None
        self._thread = None
        self._exit_event.set()
        return True
