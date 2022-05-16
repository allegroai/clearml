try:
    import fire
    import fire.core
    import fire.helptext
except ImportError:
    fire = None

import inspect
from types import SimpleNamespace
from .frameworks import _patched_call  # noqa
from ..config import get_remote_task_id, running_remotely
from ..utilities.dicts import cast_str_to_bool


class PatchFire:
    _args = {}
    _command_type = "fire.Command"
    _command_arg_type_template = "fire.Arg@%s"
    _shared_arg_type = "fire.Arg.shared"
    _section_name = "Args"
    _args_sep = "/"
    _commands_sep = "."
    _current_task = None
    __remote_task_params = None
    __remote_task_params_dict = {}
    __patched = False
    __groups = []
    __commands = {}
    __default_args = SimpleNamespace(
        completion=None, help=False, interactive=False, separator="-", trace=False, verbose=False
    )
    __current_command = None
    __fetched_current_command = False
    __command_args = {}

    @classmethod
    def patch(cls, task=None):
        if fire is None:
            return

        cls._current_task = task
        if task:
            cls._update_task_args()

        if not cls.__patched:
            cls.__patched = True
            if running_remotely():
                fire.core._Fire = _patched_call(fire.core._Fire, PatchFire.__Fire)
            else:
                fire.core._CallAndUpdateTrace = _patched_call(
                    fire.core._CallAndUpdateTrace, PatchFire.__CallAndUpdateTrace
                )

    @classmethod
    def _update_task_args(cls):
        if running_remotely() or not cls._current_task:
            return
        args = {}
        parameters_types = {}
        if cls.__current_command is None:
            args = {cls._section_name + cls._args_sep + k: v for k, v in cls._args.items()}
            parameters_types = {cls._section_name + cls._args_sep + k: cls._shared_arg_type for k in cls._args.keys()}
            for k in PatchFire.__command_args.get(None) or []:
                k = cls._section_name + cls._args_sep + k
                if k not in args:
                    args[k] = None
        else:
            args[cls._section_name + cls._args_sep + cls.__current_command] = True
            parameters_types[cls._section_name + cls._args_sep + cls.__current_command] = cls._command_type
            args = {
                **args,
                **{
                    cls._section_name + cls._args_sep + cls.__current_command + cls._args_sep + k: v
                    for k, v in cls._args.items()
                    if k in (PatchFire.__command_args.get(cls.__current_command) or [])
                },
                **{
                    cls._section_name + cls._args_sep + k: v
                    for k, v in cls._args.items()
                    if k not in (PatchFire.__command_args.get(cls.__current_command) or [])
                },
            }
            parameters_types = {
                **parameters_types,
                **{
                    cls._section_name
                    + cls._args_sep
                    + cls.__current_command
                    + cls._args_sep
                    + k: cls._command_arg_type_template % cls.__current_command
                    for k in cls._args.keys()
                    if k in (PatchFire.__command_args.get(cls.__current_command) or [])
                },
                **{
                    cls._section_name + cls._args_sep + k: cls._shared_arg_type
                    for k in cls._args.keys()
                    if k not in (PatchFire.__command_args.get(cls.__current_command) or [])
                },
            }
        for command in cls.__commands:
            if command == cls.__current_command:
                continue
            args[cls._section_name + cls._args_sep + command] = False
            parameters_types[cls._section_name + cls._args_sep + command] = cls._command_type
            unused_command_args = {
                cls._section_name + cls._args_sep + command + cls._args_sep + k: None
                for k in (cls.__command_args.get(command) or [])
            }
            unused_paramenters_types = {
                cls._section_name
                + cls._args_sep
                + command
                + cls._args_sep
                + k: cls._command_arg_type_template % command
                for k in (cls.__command_args.get(command) or [])
            }
            args = {**args, **unused_command_args}
            parameters_types = {**parameters_types, **unused_paramenters_types}

        # noinspection PyProtectedMember
        cls._current_task._set_parameters(
            args,
            __update=True,
            __parameters_types=parameters_types,
        )

    @staticmethod
    def __Fire(original_fn, component, args_, parsed_flag_args, context, name, *args, **kwargs):  # noqa
        if not running_remotely():
            return original_fn(component, args_, parsed_flag_args, context, name, *args, **kwargs)
        command = PatchFire._load_task_params()
        if command is not None:
            replaced_args = command.split(PatchFire._commands_sep)
        else:
            replaced_args = []
        for param in PatchFire.__remote_task_params[PatchFire._section_name].values():
            if command is not None and param.type == PatchFire._command_arg_type_template % command:
                replaced_args.append("--" + param.name[len(command + PatchFire._args_sep):])
                value = PatchFire.__remote_task_params_dict[param.name]
                if len(value) > 0:
                    replaced_args.append(value)
            if param.type == PatchFire._shared_arg_type:
                replaced_args.append("--" + param.name)
                value = PatchFire.__remote_task_params_dict[param.name]
                if len(value) > 0:
                    replaced_args.append(value)
        return original_fn(component, replaced_args, parsed_flag_args, context, name, *args, **kwargs)

    @staticmethod
    def __CallAndUpdateTrace(  # noqa
        original_fn, component, args_, component_trace, treatment, target, *args, **kwargs
    ):
        if running_remotely():
            return original_fn(component, args_, component_trace, treatment, target, *args, **kwargs)
        if not PatchFire.__fetched_current_command:
            PatchFire.__fetched_current_command = True
            context, component_context = PatchFire.__get_context_and_component(component)
            PatchFire.__groups, PatchFire.__commands = PatchFire.__get_all_groups_and_commands(
                component_context, context
            )
            PatchFire.__current_command = PatchFire.__get_current_command(
                args_, PatchFire.__groups, PatchFire.__commands
            )
            for command in PatchFire.__commands:
                PatchFire.__command_args[command] = PatchFire.__get_command_args(
                    component_context, command.split(PatchFire._commands_sep), PatchFire.__default_args, context
                )
            PatchFire.__command_args[None] = PatchFire.__get_command_args(
                component_context,
                "",
                PatchFire.__default_args,
                context,
            )
        for k, v in PatchFire.__commands.items():
            if v == component:
                PatchFire.__current_command = k
                break
            # Comparing methods in Python is equivalent to comparing the __func__ of the methods
            # and the objects they are bound to. We do not care about the object in this case,
            # so we just compare the __func__
            if inspect.ismethod(component) and inspect.ismethod(v) and v.__func__ == component.__func__:
                PatchFire.__current_command = k
                break
        fn = component.__call__ if treatment == "callable" else component
        metadata = fire.decorators.GetMetadata(component)
        fn_spec = fire.inspectutils.GetFullArgSpec(component)
        parse = fire.core._MakeParseFn(fn, metadata)  # noqa
        (parsed_args, parsed_kwargs), _, _, _ = parse(args_)
        PatchFire._args = {**PatchFire._args, **{k: v for k, v in zip(fn_spec.args, parsed_args)}, **parsed_kwargs}
        PatchFire._update_task_args()
        return original_fn(component, args_, component_trace, treatment, target, *args, **kwargs)

    @staticmethod
    def __get_context_and_component(component):
        context = {}
        component_context = component
        # Walk through the stack to find the arguments with fire.Fire() has been called.
        # Can't do it by patching the function because we want to patch _CallAndUpdateTrace,
        # which is called by fire.Fire()
        frame_infos = inspect.stack()
        for frame_info_ind, frame_info in enumerate(frame_infos):
            if frame_info.function == "Fire":
                component_context = inspect.getargvalues(frame_info.frame).locals["component"]
                if inspect.getargvalues(frame_info.frame).locals["component"] is None:
                    # This is similar to how fire finds this context
                    fire_context_frame = frame_infos[frame_info_ind + 1].frame
                    context.update(fire_context_frame.f_globals)
                    context.update(fire_context_frame.f_locals)
                    # Ignore modules, as they yield too many commands.
                    # Also ignore clearml.task.
                    context = {
                        k: v
                        for k, v in context.items()
                        if not inspect.ismodule(v) and (not inspect.isclass(v) or v.__module__ != "clearml.task")
                    }
                break
        return context, component_context

    @staticmethod
    def __get_all_groups_and_commands(component, context):
        groups = []
        commands = {}
        component_trace_result = PatchFire.__safe_Fire(component, [], PatchFire.__default_args, context)
        group_args = [[]]
        while len(group_args) > 0:
            query_group = group_args[-1]
            groups.append(PatchFire._commands_sep.join(query_group))
            group_args = group_args[:-1]
            current_groups, current_commands = PatchFire.__get_groups_and_commands_for_args(
                component_trace_result, query_group, PatchFire.__default_args, context
            )
            for command in current_commands:
                prefix = (
                    PatchFire._commands_sep.join(query_group) + PatchFire._commands_sep if len(query_group) > 0 else ""
                )
                commands[prefix + command[0]] = command[1]
            for group in current_groups:
                group_args.append(query_group + [group[0]])
        return groups, commands

    @staticmethod
    def __get_groups_and_commands_for_args(component, args_, parsed_flag_args, context, name=None):
        component_trace = PatchFire.__safe_Fire(component, args_, parsed_flag_args, context, name=name)
        groups, commands, _, _ = fire.helptext._GetActionsGroupedByKind(component_trace, verbose=False)  # noqa
        groups = [(name, member) for name, member in groups.GetItems()]
        commands = [(name, member) for name, member in commands.GetItems()]
        return groups, commands

    @staticmethod
    def __get_current_command(args_, groups, commands):
        current_command = ""
        for arg in args_:
            prefix = (current_command + PatchFire._commands_sep) if len(current_command) > 0 else ""
            potential_current_command = prefix + arg
            if potential_current_command not in groups:
                if potential_current_command in commands:
                    return potential_current_command
                else:
                    return None
            current_command = potential_current_command
        return None

    @staticmethod
    def __get_command_args(component, args_, parsed_flag_args, context, name=None):
        component_trace = PatchFire.__safe_Fire(component, args_, parsed_flag_args, context, name=None)
        fn_spec = fire.inspectutils.GetFullArgSpec(component_trace)
        return fn_spec.args

    @staticmethod
    def __safe_Fire(component, args_, parsed_flag_args, context, name=None):
        orig = None
        # noinspection PyBroadException
        try:

            def __CallAndUpdateTrace_rogue_call_guard(*args, **kwargs):
                raise fire.core.FireError()

            orig = fire.core._CallAndUpdateTrace  # noqa
            fire.core._CallAndUpdateTrace = __CallAndUpdateTrace_rogue_call_guard  # noqa
            result = fire.core._Fire(component, args_, parsed_flag_args, context, name=name).GetResult()  # noqa
        except Exception:
            result = None
        finally:
            if orig:
                fire.core._CallAndUpdateTrace = orig  # noqa
        return result

    @staticmethod
    def _load_task_params():
        if not PatchFire.__remote_task_params:
            from clearml import Task

            t = Task.get_task(task_id=get_remote_task_id())
            # noinspection PyProtectedMember
            PatchFire.__remote_task_params = t._get_task_property("hyperparams") or {}
            params_dict = t.get_parameters(backwards_compatibility=False)
            skip = len(PatchFire._section_name) + 1
            PatchFire.__remote_task_params_dict = {
                k[skip:]: v
                for k, v in params_dict.items()
                if k.startswith(PatchFire._section_name + PatchFire._args_sep)
            }

        command = [
            p.name
            for p in PatchFire.__remote_task_params[PatchFire._section_name].values()
            if p.type == PatchFire._command_type and cast_str_to_bool(p.value, strip=True)
        ]
        return command[0] if command else None


# patch fire before anything
PatchFire.patch()
