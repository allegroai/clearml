try:
    import fire
    import inspect
    import fire.core  # noqa
except ImportError:
    fire = None

from os import stat
from types import SimpleNamespace
from .frameworks import _patched_call  # noqa
from ..config import running_remotely, get_remote_task_id
from ..utilities.dicts import cast_str_to_bool


class PatchFire:
    _args = {}
    _args_types = {}
    _kwargs = {}
    _command_type = "fire.Command"
    _multi_command = False
    _main_task = None
    _section_name = "Args"
    _command_type = "fire.Command"
    __remote_task_params = None
    __remote_task_params_dict = {}
    __patched = False
    __processed_args = False
    __groups = []
    __commands = {}
    __parsed_flag_args = SimpleNamespace(
        completion=None, help=False, interactive=False, separator="-", trace=False, verbose=False
    )
    __current_command = None
    __command_args = {}

    @classmethod
    def patch(cls, task=None):
        if fire is None:
            return

        if task:
            cls._main_task = task
            PatchFire._update_task_args()

        if not cls.__patched:
            cls.__patched = True
            fire.core._Fire = _patched_call(fire.core._Fire, PatchFire.__Fire)

    @classmethod
    def _update_task_args(cls):
        if running_remotely() or not cls._main_task:
            return
        args = {}
        parameters_types = {}
        if cls.__current_command is None:
            args = {cls._section_name + "/" + k: v for k, v in cls._args.items()}
        else:
            args[cls._section_name + "/" + cls.__current_command] = True
            parameters_types[cls._section_name + "/" + cls.__current_command] = cls._command_type
            args = {
                **args,
                **{cls._section_name + "/" + cls.__current_command + "/" + k: v for k, v in cls._args.items()},
            }
        for command in cls.__commands:
            if command == cls.__current_command:
                continue
            args[cls._section_name + "/" + command] = False
            parameters_types[cls._section_name + "/" + command] = cls._command_type
            unused_command_args = {
                cls._section_name + "/" + command + "/" + k: None for k in cls.__command_args[command]
            }
            args = {**args, **unused_command_args}

        print(f"Args are {args}")
        # noinspection PyProtectedMember
        cls._main_task._set_parameters(
            args,
            __update=True,
            __parameters_types=parameters_types,
        )

    @staticmethod
    def __Fire(original_fn, component, args_, parsed_flag_args, context, name, *args, **kwargs):
        print("GOT IN FIRE")
        print(args_)
        if running_remotely():
            command = PatchFire._load_task_params()
            print(command)
            if command is not None:
                start_with = command + "/"
                replaced_args = ".".split(command)
            else:
                start_with = ""
                replaced_args = []
            for k, v in PatchFire.__remote_task_params_dict.items():
                if k.startswith(start_with):
                    replaced_args.append("--" + k[len(start_with) :])
                    if v is not None:
                        replaced_args.append(v)
            return original_fn(component, replaced_args, parsed_flag_args, context, name, *args, **kwargs)
        if PatchFire.__processed_args:
            return original_fn(component, args_, parsed_flag_args, context, name, *args, **kwargs)
        PatchFire.__processed_args = True
        PatchFire.__groups, PatchFire.__commands = PatchFire.__get_all_groups_and_commands(component, context)
        PatchFire.__current_command = PatchFire.__get_current_command(args_, PatchFire.__groups, PatchFire.__commands)
        for command in PatchFire.__commands:
            PatchFire.__command_args[command] = PatchFire.__get_command_args(
                component, command.split("."), parsed_flag_args, context, name=name
            )
        command_as_args = []
        if PatchFire.__current_command is not None:
            command_as_args = PatchFire.__current_command.split(".")
        PatchFire._args = PatchFire.__get_used_args(
            component, command_as_args, args_, parsed_flag_args, context, name, *args, **kwargs
        )
        PatchFire._update_task_args()
        return original_fn(component, args_, parsed_flag_args, context, name, *args, **kwargs)

    @staticmethod
    def __get_all_groups_and_commands(component, context):
        groups = []
        commands = {}
        component_trace_result = fire.core._Fire(component, [], PatchFire.__parsed_flag_args, context).GetResult()
        group_args = [[]]
        while len(group_args) > 0:
            query_group = group_args[-1]
            groups.append(".".join(query_group))
            group_args = group_args[:-1]
            current_groups, current_commands = PatchFire.__get_groups_and_commands_for_args(
                component_trace_result, query_group, PatchFire.__parsed_flag_args, context
            )
            for command in current_commands:
                prefix = ".".join(query_group) + "." if len(query_group) > 0 else ""
                commands[prefix + command[0]] = command[1]
            for group in current_groups:
                group_args.append(query_group + [group[0]])
        return groups, commands

    @staticmethod
    def __get_groups_and_commands_for_args(component, args_, parsed_flag_args, context, name=None):
        component_trace = fire.core._Fire(component, args_, parsed_flag_args, context, name=name).GetResult()
        groups, commands, _, _ = fire.helptext._GetActionsGroupedByKind(component_trace, verbose=False)
        groups = [(name, member) for name, member in groups.GetItems()]
        commands = [(name, member) for name, member in commands.GetItems()]
        return groups, commands

    @staticmethod
    def __get_current_command(args_, groups, commands):
        current_command = ""
        for arg in args_:
            prefix = (current_command + ".") if len(current_command) > 0 else ""
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
        component_trace = fire.core._Fire(component, args_, parsed_flag_args, context, name=name).GetResult()
        fn_spec = fire.inspectutils.GetFullArgSpec(component_trace)
        return fn_spec.args

    @staticmethod
    def __get_used_args(component, command_as_args, full_args, parsed_flag_args, context, name=None):
        component_trace = fire.core._Fire(component, command_as_args, parsed_flag_args, context, name=name).GetResult()
        metadata = fire.decorators.GetMetadata(component)
        component = component.__call__
        fn_spec = fire.inspectutils.GetFullArgSpec(component_trace)
        parse = fire.core._MakeParseFn(component, metadata)  # noqa
        (parsed_args, parsed_kwargs), _, _, _ = parse(full_args)
        return {**parsed_kwargs, **{k: v for k, v in zip(fn_spec.args, parsed_args[len(command_as_args) :])}}

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
                k[skip:]: v for k, v in params_dict.items() if k.startswith(PatchFire._section_name + "/")
            }

        params = PatchFire.__remote_task_params
        command = [
            p.name
            for p in params["Args"].values()
            if p.type == PatchFire._command_type and cast_str_to_bool(p.value, strip=True)
        ]
        return command[0] if command else None


# patch fire before anything
PatchFire.patch()
