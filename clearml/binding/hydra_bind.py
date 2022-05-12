import io
import sys
from functools import partial

from ..config import running_remotely, DEV_TASK_NO_REUSE
from ..debugging.log import LoggerRoot


class PatchHydra(object):
    _original_run_job = None
    _original_hydra_run = None
    _allow_omegaconf_edit = None
    _current_task = None
    _last_untracked_state = {}
    _config_section = 'OmegaConf'
    _parameter_section = 'Hydra'
    _parameter_allow_full_edit = '_allow_omegaconf_edit_'

    @classmethod
    def patch_hydra(cls):
        # noinspection PyBroadException
        try:
            # only once
            if cls._original_run_job:
                return True
            # if hydra is not loaded, do not patch anything
            if not sys.modules.get('hydra'):
                return False

            from hydra._internal import hydra as internal_hydra  # noqa
            from hydra.core import utils as utils_hydra  # noqa
            from hydra._internal.core_plugins import basic_launcher  # noqa

            cls._original_hydra_run = internal_hydra.Hydra.run
            internal_hydra.Hydra.run = cls._patched_hydra_run

            cls._original_run_job = utils_hydra.run_job
            utils_hydra.run_job = cls._patched_run_job
            internal_hydra.run_job = cls._patched_run_job
            basic_launcher.run_job = cls._patched_run_job
            return True
        except Exception:
            return False

    @staticmethod
    def update_current_task(task):
        # set current Task before patching
        PatchHydra._current_task = task
        if not task:
            return
        if PatchHydra.patch_hydra():
            # check if we have an untracked state, store it.
            if PatchHydra._last_untracked_state.get('connect'):
                PatchHydra._current_task.connect(**PatchHydra._last_untracked_state['connect'])
            if PatchHydra._last_untracked_state.get('_set_configuration'):
                # noinspection PyProtectedMember
                PatchHydra._current_task._set_configuration(**PatchHydra._last_untracked_state['_set_configuration'])
            PatchHydra._last_untracked_state = {}
        else:
            # if patching failed set it to None
            PatchHydra._current_task = None

    @staticmethod
    def _patched_hydra_run(self, config_name, task_function, overrides, *args, **kwargs):
        PatchHydra._allow_omegaconf_edit = False

        # store the config
        # noinspection PyBroadException
        try:
            if running_remotely():
                # get the _parameter_allow_full_edit casted back to boolean
                connected_config = dict()
                connected_config[PatchHydra._parameter_allow_full_edit] = False
                PatchHydra._current_task.connect(connected_config, name=PatchHydra._parameter_section)
                PatchHydra._allow_omegaconf_edit = connected_config.pop(PatchHydra._parameter_allow_full_edit, None)
                # get all the overrides
                full_parameters = PatchHydra._current_task.get_parameters(backwards_compatibility=False)
                stored_config = {k[len(PatchHydra._parameter_section)+1:]: v for k, v in full_parameters.items()
                                 if k.startswith(PatchHydra._parameter_section+'/')}
                stored_config.pop(PatchHydra._parameter_allow_full_edit, None)
                overrides = ['{}={}'.format(k, v) for k, v in stored_config.items()]
            else:
                # We take care of it inside the _patched_run_job
                pass
        except Exception:
            pass

        return PatchHydra._original_hydra_run(self, config_name, task_function, overrides, *args, **kwargs)

    @staticmethod
    def _patched_run_job(config, task_function, *args, **kwargs):
        # noinspection PyBroadException
        try:
            from hydra.core.utils import JobStatus

            failed_status = JobStatus.FAILED
        except Exception:
            LoggerRoot.get_base_logger(PatchHydra).warning(
                "Could not import JobStatus from Hydra. Failed tasks will be marked as completed"
            )
            failed_status = None

        # store the config
        # noinspection PyBroadException
        try:
            if running_remotely():
                # we take care of it in the hydra run (where we have access to the overrides)
                pass
            else:
                overrides = config.hydra.overrides.task
                stored_config = dict(arg.split('=', 1) for arg in overrides)
                stored_config[PatchHydra._parameter_allow_full_edit] = False
                if PatchHydra._current_task:
                    PatchHydra._current_task.connect(stored_config, name=PatchHydra._parameter_section)
                    PatchHydra._last_untracked_state.pop('connect', None)
                else:
                    PatchHydra._last_untracked_state['connect'] = dict(
                        mutable=stored_config, name=PatchHydra._parameter_section)
                # todo: remove the overrides section from the Args (we have it here)
                # PatchHydra._current_task.delete_parameter('Args/overrides')
        except Exception:
            pass

        pre_app_task_init_call = bool(PatchHydra._current_task)

        if pre_app_task_init_call and not running_remotely():
            LoggerRoot.get_base_logger(PatchHydra).info(
                "Task.init called outside of Hydra-App. For full Hydra multi-run support, "
                "move the Task.init call into the Hydra-App main function"
            )

        kwargs["config"] = config
        kwargs["task_function"] = partial(PatchHydra._patched_task_function, task_function,)
        result = PatchHydra._original_run_job(*args, **kwargs)
        # noinspection PyBroadException
        try:
            result_status = result.status
        except Exception:
            LoggerRoot.get_base_logger(PatchHydra).warning(
                "Could not get Hydra job status. Failed tasks will be marked as completed"
            )
            result_status = None

        # if we have Task.init called inside the App, we close it after the app is done.
        # This will make sure that hydra run will create multiple Tasks
        if (
            not running_remotely()
            and not pre_app_task_init_call
            and PatchHydra._current_task
            and (failed_status is None or result_status is None or result_status != failed_status)
        ):
            PatchHydra._current_task.close()
            # make sure we do not reuse the Task if we have a multi-run session
            DEV_TASK_NO_REUSE.set(True)
            PatchHydra._current_task = None

        return result

    @staticmethod
    def _patched_task_function(task_function, a_config, *a_args, **a_kwargs):
        from omegaconf import OmegaConf  # noqa
        if not running_remotely() or not PatchHydra._allow_omegaconf_edit:
            PatchHydra._register_omegaconf(a_config)
        else:
            # noinspection PyProtectedMember
            omega_yaml = PatchHydra._current_task._get_configuration_text(PatchHydra._config_section)
            loaded_config = OmegaConf.load(io.StringIO(omega_yaml))
            a_config = OmegaConf.merge(a_config, loaded_config)
            PatchHydra._register_omegaconf(a_config, is_read_only=False)
        return task_function(a_config, *a_args, **a_kwargs)

    @staticmethod
    def _register_omegaconf(config, is_read_only=True):
        from omegaconf import OmegaConf  # noqa

        if is_read_only:
            description = \
                'Full OmegaConf YAML configuration. ' \
                'This is a read-only section, unless \'{}/{}\' is set to True'.format(
                    PatchHydra._parameter_section, PatchHydra._parameter_allow_full_edit)
        else:
            description = 'Full OmegaConf YAML configuration overridden! ({}/{}=True)'.format(
                PatchHydra._parameter_section, PatchHydra._parameter_allow_full_edit)

        # we should not have the hydra section in the config, but this seems never to be the case anymore.
        # config = config.copy()
        # config.pop('hydra', None)

        configuration = dict(
            name=PatchHydra._config_section,
            description=description,
            config_type='OmegaConf YAML',
            config_text=OmegaConf.to_yaml(config, resolve=False)
        )
        if PatchHydra._current_task:
            # noinspection PyProtectedMember
            PatchHydra._current_task._set_configuration(**configuration)
            PatchHydra._last_untracked_state.pop('_set_configuration', None)
        else:
            PatchHydra._last_untracked_state['_set_configuration'] = configuration


def __global_hydra_bind():
    # noinspection PyBroadException
    try:
        import hydra  # noqa
        PatchHydra.patch_hydra()
    except Exception:
        pass


# patch hydra
__global_hydra_bind()
