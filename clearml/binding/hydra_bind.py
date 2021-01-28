import io
import sys

from ..config import running_remotely, get_remote_task_id


class PatchHydra(object):
    _original_run_job = None
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

            cls._original_run_job = internal_hydra.Hydra.run
            internal_hydra.Hydra.run = cls._patched_run_job
            return True
        except Exception:
            return False

    @staticmethod
    def update_current_task(task):
        # set current Task before patching
        PatchHydra._current_task = task
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
    def _patched_run_job(self, config_name, task_function, overrides, *args, **kwargs):
        allow_omegaconf_edit = False

        def patched_task_function(a_config, *a_args, **a_kwargs):
            from omegaconf import OmegaConf  # noqa
            if not running_remotely() or not allow_omegaconf_edit:
                PatchHydra._register_omegaconf(a_config)
            else:
                # noinspection PyProtectedMember
                omega_yaml = PatchHydra._current_task._get_configuration_text(PatchHydra._config_section)
                loaded_config = OmegaConf.load(io.StringIO(omega_yaml))
                a_config = OmegaConf.merge(a_config, loaded_config)
                PatchHydra._register_omegaconf(a_config, is_read_only=False)
            return task_function(a_config, *a_args, **a_kwargs)

        # store the config
        # noinspection PyBroadException
        try:
            if running_remotely():
                if not PatchHydra._current_task:
                    from ..task import Task
                    PatchHydra._current_task = Task.get_task(task_id=get_remote_task_id())
                # get the _parameter_allow_full_edit casted back to boolean
                connected_config = dict()
                connected_config[PatchHydra._parameter_allow_full_edit] = False
                PatchHydra._current_task.connect(connected_config, name=PatchHydra._parameter_section)
                allow_omegaconf_edit = connected_config.pop(PatchHydra._parameter_allow_full_edit, None)
                # get all the overrides
                full_parameters = PatchHydra._current_task.get_parameters(backwards_compatibility=False)
                stored_config = {k[len(PatchHydra._parameter_section)+1:]: v for k, v in full_parameters.items()
                                 if k.startswith(PatchHydra._parameter_section+'/')}
                stored_config.pop(PatchHydra._parameter_allow_full_edit, None)
                overrides = ['{}={}'.format(k, v) for k, v in stored_config.items()]
            else:
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
        return PatchHydra._original_run_job(self, config_name, patched_task_function, overrides, *args, **kwargs)

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

        configuration = dict(
            name=PatchHydra._config_section,
            description=description,
            config_type='OmegaConf YAML',
            config_text=OmegaConf.to_yaml({k: v for k, v in config.items() if k not in ('hydra', )})
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
