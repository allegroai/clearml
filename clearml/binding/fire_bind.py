try:
    import fire
    import fire.core  # noqa
except ImportError:
    fire = None

from .frameworks import _patched_call  # noqa
from ..config import running_remotely, get_remote_task_id


class PatchFire:
    _args = {}
    _kwargs = {}
    _main_task = None
    _section_name = "Args"
    __patched = False

    @classmethod
    def patch(cls, task=None):
        if fire is None:
            return

        if task:
            cls._main_task = task
            PatchFire._update_task_args()

        if not cls.__patched:
            cls.__patched = True
            fire.core._CallAndUpdateTrace = _patched_call(fire.core._CallAndUpdateTrace, PatchFire._CallAndUpdateTrace)

    @classmethod
    def _update_task_args(cls):
        if running_remotely() or not cls._main_task or not cls._args:
            return
        args = {cls._section_name + "/" + k: v for k, v in cls._args.items()}
        # noinspection PyProtectedMember
        cls._main_task._set_parameters(
            args,
            __update=True,
        )

    @staticmethod
    def _CallAndUpdateTrace(original_fn, component, cmd_args, *args, **kwargs):  # noqa
        if not running_remotely() and "treatment" in kwargs:
            treatment = kwargs["treatment"]
            metadata = fire.decorators.GetMetadata(component)
            fn = component.__call__ if treatment == "callable" else component
            fn_spec = fire.inspectutils.GetFullArgSpec(fn)
            parse = fire.core._MakeParseFn(fn, metadata)  # noqa
            (parsed_args, parsed_kwargs), _, _, _ = parse(cmd_args)
            PatchFire._args = {**parsed_kwargs, **{k: v for k, v in zip(fn_spec.args, parsed_args)}}
        return original_fn(component, cmd_args, *args, **kwargs)


# patch fire before anything
PatchFire.patch()
