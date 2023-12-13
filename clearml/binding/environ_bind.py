import os
from functools import partial
from time import sleep
from multiprocessing import pool
import six

from ..config import TASK_LOG_ENVIRONMENT, running_remotely, config
from ..utilities.process.mp import BackgroundMonitor


class EnvironmentBind(object):
    _current_task = None
    _environment_section = 'Environment'
    __patched = False

    @classmethod
    def update_current_task(cls, task):
        cls._current_task = task
        # noinspection PyBroadException
        try:
            if not cls.__patched:
                cls.__patched = True
                cls._bind_environment()
        except Exception:
            pass

    @classmethod
    def _bind_environment(cls):
        if not cls._current_task:
            return

        # get ENVIRONMENT and put it into the OS environment
        if running_remotely():
            params = cls._current_task.get_parameters_as_dict()
            if params and cls._environment_section in params:
                # put back into os:
                os.environ.update(params[cls._environment_section])
            return

        environ_log = \
            str(TASK_LOG_ENVIRONMENT.get() or '').strip() or config.get('development.log_os_environments', [])
        if environ_log and isinstance(environ_log, str):
            environ_log = [e.strip() for e in environ_log.split(',')]

        if not environ_log:
            return

        env_param = dict()
        for match in environ_log:
            match = match.strip()
            if match == '*':
                env_param.update({k: os.environ.get(k) for k in os.environ
                                  if not k.startswith('TRAINS_') and not k.startswith('CLEARML_')})
            elif match.endswith('*'):
                match = match.rstrip('*')
                env_param.update({k: os.environ.get(k) for k in os.environ if k.startswith(match)})
            elif match.startswith('*'):
                match = match.lstrip('*')
                env_param.update({k: os.environ.get(k) for k in os.environ if k.endswith(match)})
            elif match in os.environ:
                env_param.update({match: os.environ.get(match)})
        # store os environments
        cls._current_task.connect(env_param, cls._environment_section)


class SimpleQueueWrapper(object):
    def __init__(self, task, simple_queue):
        self.__current_task = task
        self.__simple_queue = simple_queue

    def __getattr__(self, attr):
        if attr in ["__simple_queue", "__current_task"]:
            return self.__dict__.get(attr)

        if attr == "put":
            def _patched_put(*a_args, **a_kwargs):
                # make sure we flush everything, because after we push the result we will get terminated
                try:
                    if self.__current_task and self.__current_task.is_main_task():
                        self.__current_task.flush(wait_for_uploads=True)
                except:  # noqa
                    pass
                return getattr(self.__simple_queue, "put")(*a_args, **a_kwargs)

            return _patched_put

        return getattr(self.__simple_queue, attr)


class PatchOsFork(object):
    _original_fork = None
    _registered_fork_callbacks = False
    _current_task = None
    _original_process_run = None

    @classmethod
    def patch_fork(cls, task):
        cls._current_task = task
        if not task:
            return

        # first we need to patch regular fork
        # because forked processes do not support atexit, they call os._exit directly)

        # noinspection PyBroadException
        try:
            # only once
            if cls._registered_fork_callbacks or cls._original_fork:
                return
            try:
                os.register_at_fork(before=PatchOsFork._fork_callback_before,
                                    after_in_child=PatchOsFork._fork_callback_after_child)
                cls._registered_fork_callbacks = True
            except Exception:
                # python <3.6
                if six.PY2:
                    cls._original_fork = staticmethod(os.fork)
                else:
                    cls._original_fork = os.fork
                os.fork = cls._patched_fork

        except Exception:
            pass

        # now we need to patch Process.run because the bootstrap code
        # shuts everything down before calling os._exit that we patched above
        try:
            from multiprocessing.process import BaseProcess
            PatchOsFork._original_process_run = BaseProcess.run
            BaseProcess.run = PatchOsFork._patched_process_run
        except:  # noqa
            pass

    @staticmethod
    def _patched_pool_worker(original_worker, *args, **kwargs):
        if not PatchOsFork._current_task:
            return original_worker(*args, **kwargs)

        try:
            if len(args) >= 2 and hasattr(args[1], "put"):
                args = list(args)
                args[1] = SimpleQueueWrapper(PatchOsFork._current_task, args[1])
                args = tuple(args)
            elif "outqueue" in kwargs and hasattr(kwargs["outqueue"], "put"):
                kwargs["outqueue"] = SimpleQueueWrapper(PatchOsFork._current_task, kwargs["outqueue"])
        except:  # noqa
            pass

        return original_worker(*args, **kwargs)

    @staticmethod
    def _patched_process_run(self, *args, **kwargs):
        if not PatchOsFork._current_task:
            return PatchOsFork._original_process_run(self, *args, **kwargs)

        try:
            from ..task import Task
            task = Task.current_task()
        except:  # noqa
            task = None

        # check if this is Process Pool function
        patched_worker = False
        if hasattr(self, "_target"):
            # Now we have to patch Pool, because pool terminates subprocess directly after
            # the return value of the pool worker function is pushed into the queue,
            # which means it will terminate the process before we finish running our "atexit" call
            try:
                if self._target == pool.worker:  # noqa
                    self._target = partial(PatchOsFork._patched_pool_worker, pool.worker)  # noqa
                    patched_worker = True
            except:  # noqa
                pass

        try:
            return PatchOsFork._original_process_run(self, *args, **kwargs)
        finally:
            if task and patched_worker:
                try:
                    # noinspection PyProtectedMember
                    if task._report_subprocess_enabled:
                        # just in case, remove at exit hooks, we will deadlock when the
                        # main Pool manager will terminate this process, and it will...
                        # noinspection PyProtectedMember
                        task._at_exit_called = True
                    else:
                        # terminate the current Task
                        # noinspection PyProtectedMember
                        task._at_exit()
                except:  # noqa
                    pass

    @staticmethod
    def _fork_callback_before():
        if not PatchOsFork._current_task:
            return
        from ..task import Task

        # ensure deferred is done, but never try to generate a Task object
        # noinspection PyProtectedMember
        task = Task._Task__main_task
        # this will force the deferred init call to finish
        # noinspection PyProtectedMember
        Task._wait_for_deferred(task)

    @staticmethod
    def _fork_callback_after_child():
        if not PatchOsFork._current_task:
            return

        from ..task import Task

        # force creating a Task
        task = Task.current_task()
        if not task:
            return

        if not Task._report_subprocess_enabled:
            # https://stackoverflow.com/a/34507557
            # NOTICE: subprocesses do not exit through exit we have to register signals
            if task._Task__exit_hook:
                task._Task__exit_hook.register_signal_and_exception_hooks()
        else:
            # noinspection PyProtectedMember
            task._remove_signal_hooks()

        # noinspection PyProtectedMember
        if Task._report_subprocess_enabled:
            # noinspection PyProtectedMember
            task._remove_exception_hooks()

        PatchOsFork._current_task = task
        # # Hack: now make sure we setup the reporter threads (Log+Reporter)
        # noinspection PyProtectedMember
        if not bool(task._report_subprocess_enabled):
            BackgroundMonitor.start_all(task=task)

        # if we are reporting into a subprocess, no need to further patch the exit functions
        if Task._report_subprocess_enabled:
            return

        # The signal handler method is Not enough, for the time being, we have both
        # even though it makes little sense
        # # if we got here patch the os._exit of our instance to call us
        def _at_exit_callback(*a_args, **a_kwargs):
            # just make sure we flush the internal state (the at exist caught by the external signal does the rest
            # in theory we should not have to do any of that, but for some reason if we do not
            # the signal is never caught by the signal call backs, not sure why....
            sleep(0.1)
            # Since at_exist handlers do not work on forked processes, we have to manually call them here
            if task:
                try:
                    # not to worry there is a double _at_exit protection implemented inside task._at_exit()
                    # noinspection PyProtectedMember
                    task._at_exit()
                except:  # noqa
                    pass

            # noinspection PyProtectedMember, PyUnresolvedReferences
            return os._org_exit(*a_args, **a_kwargs)

        if not hasattr(os, '_org_exit'):
            # noinspection PyProtectedMember, PyUnresolvedReferences
            os._org_exit = os._exit

        # noinspection PyProtectedMember
        # https://stackoverflow.com/a/34507557
        # NOTICE: subprocesses do not exit through exit, and in most cases not with _exit,
        # this means at_exit calls are Not registered respected
        os._exit = _at_exit_callback

    @staticmethod
    def _patched_fork(*args, **kwargs):
        if not PatchOsFork._current_task:
            return PatchOsFork._original_fork(*args, **kwargs)

        PatchOsFork._fork_callback_before()

        ret = PatchOsFork._original_fork(*args, **kwargs)
        if not PatchOsFork._current_task:
            return ret
        # Make sure the new process stdout is logged
        if not ret:
            PatchOsFork._fork_callback_after_child()

        return ret

    @staticmethod
    def unpatch_fork():
        try:
            if PatchOsFork._original_fork and os._exit != PatchOsFork._original_fork:
                os._exit = PatchOsFork._original_fork
                PatchOsFork._original_fork = None
        except Exception:
            pass

    @staticmethod
    def unpatch_process_run():
        try:
            from multiprocessing.process import BaseProcess

            if PatchOsFork._original_process_run and BaseProcess.run != PatchOsFork._original_process_run:
                BaseProcess.run = PatchOsFork._original_process_run
                PatchOsFork._original_process_run = None
        except Exception:
            pass
