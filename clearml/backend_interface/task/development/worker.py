import attr
from threading import Thread, Event

from time import time

from ....config import deferred_config
from ....backend_interface.task.development.stop_signal import TaskStopSignal
from ....backend_api.services import tasks


class DevWorker(object):
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
        self._exit_event = Event()
        self._task = None
        self._support_ping = False

    def ping(self, timestamp=None):
        try:
            if self._task:
                self._task.send(tasks.PingRequest(self._task.id))
        except Exception:
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

    def _daemon(self):
        last_ping = time()
        while self._task is not None:
            try:
                if self._exit_event.wait(min(float(self.ping_period), float(self.report_period))):
                    return
                # send ping request
                if self._support_ping and (time() - last_ping) >= float(self.ping_period):
                    self.ping()
                    last_ping = time()
                if self._dev_stop_signal:
                    stop_reason = self._dev_stop_signal.test()
                    if stop_reason and self._task:
                        self._task._dev_mode_stop_task(stop_reason)
            except Exception:
                pass

    def unregister(self):
        self._dev_stop_signal = None
        self._task = None
        self._thread = None
        self._exit_event.set()
        return True
