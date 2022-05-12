import os
from time import sleep

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


class PatchOsFork(object):
    _original_fork = None
    _current_task = None

    @classmethod
    def patch_fork(cls, task):
        cls._current_task = task
        if not task:
            return
        # noinspection PyBroadException
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
        from ..task import Task

        # ensure deferred is done, but never try to generate a Task object
        # noinspection PyProtectedMember
        task = Task._Task__main_task
        # this will force the deferred init call to finish
        # noinspection PyProtectedMember
        Task._wait_for_deferred(task)

        ret = PatchOsFork._original_fork(*args, **kwargs)
        if not PatchOsFork._current_task:
            return ret
        # Make sure the new process stdout is logged
        if not ret:
            # force creating a Task
            task = Task.current_task()
            if not task:
                return ret

            # # Hack: now make sure we setup the reporter threads (Log+Reporter)
            if not task._report_subprocess_enabled:
                BackgroundMonitor.start_all(task=task)

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

            os._exit = _at_exit_callback

        return ret
