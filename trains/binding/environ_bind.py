import os

import six

from ..config import TASK_LOG_ENVIRONMENT, running_remotely


class EnvironmentBind(object):
    _task = None

    @classmethod
    def update_current_task(cls, current_task):
        cls._task = current_task
        # noinspection PyBroadException
        try:
            cls._bind_environment()
        except Exception:
            pass

    @classmethod
    def _bind_environment(cls):
        if not cls._task:
            return
        environ_log = str(TASK_LOG_ENVIRONMENT.get() or '').strip()
        if not environ_log:
            return

        if environ_log == '*':
            env_param = {k: os.environ.get(k) for k in os.environ
                         if not k.startswith('TRAINS_') and not k.startswith('ALG_')}
        else:
            environ_log = [e.strip() for e in environ_log.split(',')]
            env_param = {k: os.environ.get(k) for k in os.environ if k in environ_log}

        env_param = cls._task.connect(env_param)
        if running_remotely():
            # put back into os:
            os.environ.update(env_param)


class PatchOsFork(object):
    _original_fork = None

    @classmethod
    def patch_fork(cls):
        try:
            # only once
            if cls._original_fork:
                return
            if six.PY2:
                cls._original_fork = staticmethod(os.fork)
            else:
                cls._original_fork = os.fork
            os.fork = cls._patched_fork
        except Exception:
            pass

    @staticmethod
    def _patched_fork(*args, **kwargs):
        ret = PatchOsFork._original_fork(*args, **kwargs)
        # Make sure the new process stdout is logged
        if not ret:
            from ..task import Task
            if Task.current_task() is not None:
                # bind sub-process logger
                task = Task.init(project_name=None, task_name=None, task_type=None)
                task.get_logger().flush()

                # Hack: now make sure we setup the reporter thread
                task._setup_reporter()

                # TODO: Check if the signal handler method is enough, for the time being, we have both
                # # if we got here patch the os._exit of our instance to call us
                def _at_exit_callback(*args, **kwargs):
                    # call at exit manually
                    # noinspection PyProtectedMember
                    task._at_exit()
                    # noinspection PyProtectedMember
                    return os._org_exit(*args, **kwargs)

                if not hasattr(os, '_org_exit'):
                    os._org_exit = os._exit
                os._exit = _at_exit_callback

        return ret
