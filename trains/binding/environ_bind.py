import os

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
