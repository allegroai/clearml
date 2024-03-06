import json
import copy
import logging

try:
    # public import capabilities of namespace, util, actions will be deprecated
    # import from "protected" instead

    from jsonargparse._namespace import Namespace
    # noinspection PyProtectedMember
    from jsonargparse._util import Path
    # noinspection PyProtectedMember
    from jsonargparse import ArgumentParser
except ImportError:
    try:
        from jsonargparse.namespace import Namespace
        from jsonargparse.util import Path
        from jsonargparse import ArgumentParser
    except ImportError:
        ArgumentParser = None

try:
    # public import capabilities of jsonargparse_typehints will be deprecated
    # import from "protected" instead

    # noinspection PyProtectedMember
    import jsonargparse._typehints as jsonargparse_typehints
except ImportError:
    try:
        import jsonargparse.typehints as jsonargparse_typehints
    except ImportError:
        jsonargparse_typehints = None

from ..config import running_remotely, get_remote_task_id
from .frameworks import _patched_call  # noqa
from ..utilities.proxy_object import verify_basic_type, flatten_dictionary


class PatchJsonArgParse(object):
    namespace_type = "jsonargparse_namespace"
    path_type = "jsonargparse_path"
    _args = {}
    _current_task = None
    _args_sep = "/"
    _args_type = {}
    _commands_sep = "."
    _command_type = "jsonargparse.Command"
    _command_name = "subcommand"
    _special_fields = ["config", "subcommand"]
    _section_name = "Args"
    _allow_jsonargparse_overrides = "_allow_config_file_override_from_ui_"
    _ignore_ui_overrides = "_ignore_ui_overrides_"
    __remote_task_params = {}
    __remote_task_params_dict = {}
    __patched = False

    @classmethod
    def update_current_task(cls, task):
        cls._current_task = task
        if not task:
            return
        cls.patch(task)

    @classmethod
    def patch(cls, task=None):
        if ArgumentParser is None:
            return
        PatchJsonArgParse._update_task_args()

        if not cls.__patched:
            cls.__patched = True
            ArgumentParser.parse_args = _patched_call(ArgumentParser.parse_args, PatchJsonArgParse._parse_args)
            if jsonargparse_typehints:
                jsonargparse_typehints.adapt_typehints = _patched_call(
                    jsonargparse_typehints.adapt_typehints, PatchJsonArgParse._adapt_typehints
                )

    @classmethod
    def _update_task_args(cls, parser=None, subcommand=None):
        if running_remotely() or not cls._current_task or not cls._args:
            return
        args = {}
        args_type = {}
        have_config_file = False
        for k, v in cls._args.items():
            key_with_section = cls._section_name + cls._args_sep + k
            args[key_with_section] = v
            if k in cls._args_type:
                args_type[key_with_section] = cls._args_type[k]
                continue
            if not verify_basic_type(v, basic_types=(float, int, bool, str, type(None))) and v:
                # noinspection PyBroadException
                try:
                    if isinstance(v, Namespace) or (
                        isinstance(v, list) and all(isinstance(sub_v, Namespace) for sub_v in v)
                    ):
                        args[key_with_section] = json.dumps(PatchJsonArgParse._handle_namespace(v))
                        args_type[key_with_section] = PatchJsonArgParse.namespace_type
                    elif isinstance(v, Path) or (isinstance(v, list) and all(isinstance(sub_v, Path) for sub_v in v)):
                        args[key_with_section] = json.dumps(PatchJsonArgParse._handle_path(v))
                        args_type[key_with_section] = PatchJsonArgParse.path_type
                        have_config_file = True
                    else:
                        args[key_with_section] = str(v)
                except Exception:
                    pass
        cls._current_task._set_parameters(args, __update=True, __parameters_types=args_type)
        if have_config_file:
            cls._current_task.set_parameter(
                cls._section_name + cls._args_sep + cls._ignore_ui_overrides,
                False,
                description="If True, values in the config file will be overriden by values found in the UI. Otherwise, the values in the config file have priority",  # noqa
            )

    @staticmethod
    def _adapt_typehints(original_fn, val, *args, **kwargs):
        if not PatchJsonArgParse._current_task or not running_remotely():
            return original_fn(val, *args, **kwargs)
        return original_fn(val, *args, **kwargs)

    @staticmethod
    def __restore_args(parser, args, subcommand=None):
        paths = PatchJsonArgParse.__get_paths_from_dict(args)
        for path in paths:
            args_to_restore = PatchJsonArgParse.__get_args_from_path(parser, path, subcommand=subcommand)
            for arg_to_restore_key, arg_to_restore_value in args_to_restore.items():
                if arg_to_restore_key in PatchJsonArgParse._special_fields:
                    continue
                args[arg_to_restore_key] = arg_to_restore_value
        return args

    @staticmethod
    def _parse_args(original_fn, obj, *args, **kwargs):
        if len(args) == 1:
            kwargs["args"] = args[0]
            args = []
        if len(args) > 1:
            return original_fn(obj, *args, **kwargs)
        if running_remotely():
            try:
                PatchJsonArgParse._load_task_params(parser=obj)
                params = PatchJsonArgParse.__remote_task_params_dict
                allow_jsonargparse_overrides_value = True
                if PatchJsonArgParse._allow_jsonargparse_overrides in params:
                    allow_jsonargparse_overrides_value = params.pop(PatchJsonArgParse._allow_jsonargparse_overrides)
                if PatchJsonArgParse._ignore_ui_overrides in params:
                    allow_jsonargparse_overrides_value = not params.pop(PatchJsonArgParse._ignore_ui_overrides)
                params_namespace = Namespace()
                for k, v in params.items():
                    params_namespace[k] = v
                if not allow_jsonargparse_overrides_value:
                    params_namespace = PatchJsonArgParse.__restore_args(
                        obj,
                        params_namespace,
                        subcommand=params_namespace.get(PatchJsonArgParse._command_name)
                    )
                if PatchJsonArgParse._allow_jsonargparse_overrides in params_namespace:
                    del params_namespace[PatchJsonArgParse._allow_jsonargparse_overrides]
                if PatchJsonArgParse._ignore_ui_overrides in params_namespace:
                    del params_namespace[PatchJsonArgParse._ignore_ui_overrides]
                return params_namespace
            except Exception as e:
                logging.getLogger(__file__).warning("Failed parsing jsonargparse arguments: {}".format(e))
                return original_fn(obj, **kwargs)
        parsed_args = original_fn(obj, **kwargs)
        # noinspection PyBroadException
        try:
            subcommand = None
            for ns_name, ns_val in Namespace(parsed_args).items():
                PatchJsonArgParse._args[ns_name] = ns_val
                if ns_name == PatchJsonArgParse._command_name:
                    PatchJsonArgParse._args_type[ns_name] = PatchJsonArgParse._command_type
                    subcommand = ns_val
            try:
                import lightning
            except ImportError:
                try:
                    import pytorch_lightning

                    lightning = pytorch_lightning
                except ImportError:
                    lightning = None
            if subcommand and subcommand in PatchJsonArgParse._args and lightning:
                subcommand_args = flatten_dictionary(
                    PatchJsonArgParse._args[subcommand],
                    prefix=subcommand + PatchJsonArgParse._commands_sep,
                    sep=PatchJsonArgParse._commands_sep,
                )
                del PatchJsonArgParse._args[subcommand]
                PatchJsonArgParse._args.update(subcommand_args)
            PatchJsonArgParse._args = {k: v for k, v in PatchJsonArgParse._args.items()}
            PatchJsonArgParse._update_task_args(parser=obj, subcommand=subcommand)
        except Exception as e:
            logging.getLogger(__file__).warning("Failed parsing jsonargparse arguments: {}".format(e))
        return parsed_args

    @classmethod
    def _load_task_params(cls, parser=None):
        if cls.__remote_task_params:
            return
        from clearml import Task

        t = Task.get_task(task_id=get_remote_task_id())
        # noinspection PyProtectedMember
        cls.__remote_task_params = t._get_task_property("hyperparams") or {}
        params_dict = t.get_parameters(backwards_compatibility=False, cast=True)
        for key, section_param in cls.__remote_task_params[cls._section_name].items():
            if section_param.type == cls.namespace_type:
                params_dict["{}/{}".format(cls._section_name, key)] = cls._get_namespace_from_json(section_param.value)
            elif section_param.type == cls.path_type:
                params_dict["{}/{}".format(cls._section_name, key)] = cls._get_path_from_json(section_param.value)
            elif (not section_param.type or section_param.type == "NoneType") and not section_param.value:
                params_dict["{}/{}".format(cls._section_name, key)] = None
        skip = len(cls._section_name) + 1
        cls.__remote_task_params_dict = {
            k[skip:]: v for k, v in params_dict.items() if k.startswith(cls._section_name + cls._args_sep)
        }
        cls.__update_remote_task_params_dict_based_on_paths(parser)

    @classmethod
    def __update_remote_task_params_dict_based_on_paths(cls, parser):
        paths = PatchJsonArgParse.__get_paths_from_dict(cls.__remote_task_params_dict)
        for path in paths:
            args = PatchJsonArgParse.__get_args_from_path(
                parser, path, subcommand=cls.__remote_task_params_dict.get("subcommand")
            )
            for subarg_key, subarg_value in args.items():
                if subarg_key not in cls.__remote_task_params_dict:
                    cls.__remote_task_params_dict[subarg_key] = subarg_value

    @staticmethod
    def __get_paths_from_dict(dict_):
        paths = [(path_key, path) for path_key, path in dict_.items() if isinstance(path, Path)]
        for subargs_key, subargs in dict_.items():
            if isinstance(subargs, list) and all(isinstance(path, Path) for path in subargs):
                paths.extend((subargs_key, path) for path in subargs)
        return paths

    @staticmethod
    def __get_args_from_path(parser, path, subcommand=None):
        try:
            # make sure no side effects happen in parser
            parser = copy.deepcopy(parser)
            argument = path[0]
            if subcommand and argument.startswith(subcommand + PatchJsonArgParse._commands_sep):
                argument = argument[len(subcommand + PatchJsonArgParse._commands_sep):]
                result = parser.parse_args(
                    [subcommand, parser.prefix_chars[0] * 2 + argument, path[1].rel_path],
                    _skip_check=True,
                    defaults=False,
                )
                if PatchJsonArgParse._command_name in result:
                    del result[PatchJsonArgParse._command_name]
            else:
                result = parser.parse_args(
                    [parser.prefix_chars[0] * 2 + argument, path[1].rel_path], _skip_check=True, defaults=False
                )
            if argument in result:
                del result[argument]
            return result
        except Exception as e:
            logging.getLogger(__file__).warning("Failed parsing jsonargparse config: {}".format(e))
            return Namespace()

    @staticmethod
    def _handle_namespace(value):
        if isinstance(value, list):
            return [PatchJsonArgParse._handle_namespace(sub_value) for sub_value in value]
        return value.as_dict()

    @staticmethod
    def _handle_path(value):
        if isinstance(value, list):
            return [PatchJsonArgParse._handle_path(sub_value) for sub_value in value]
        return {"path": str(value.rel_path), "mode": value.mode, "cwd": None, "skip_check": value.skip_check}

    @staticmethod
    def _get_namespace_from_json(json_):
        json_ = json.loads(json_)
        if isinstance(json_, list):
            return [Namespace(dict_) for dict_ in json_]
        return Namespace(json_)

    @staticmethod
    def _get_path_from_json(json_):
        json_ = json.loads(json_)
        if isinstance(json_, list):
            return [Path(**dict_) for dict_ in json_]
        return Path(**json_)


# patch jsonargparse before anything else
PatchJsonArgParse.patch()
