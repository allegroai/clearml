import json

try:
    from jsonargparse import ArgumentParser
    from jsonargparse.namespace import Namespace
    from jsonargparse.util import Path
except ImportError:
    ArgumentParser = None

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
    _section_name = "Args"
    __remote_task_params = {}
    __patched = False

    @classmethod
    def update_current_task(cls, task):
        cls._current_task = task
        if not task:
            return
        cls.patch(task)

    @classmethod
    def patch(cls, task):
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
    def _update_task_args(cls):
        if running_remotely() or not cls._current_task or not cls._args:
            return
        args = {}
        args_type = {}
        for k, v in cls._args.items():
            key_with_section = cls._section_name + cls._args_sep + k
            args[key_with_section] = v
            if k in cls._args_type:
                args_type[key_with_section] = cls._args_type[k]
                continue
            if not verify_basic_type(v) and v:
                # noinspection PyBroadException
                try:
                    if isinstance(v, Namespace) or (isinstance(v, list) and all(isinstance(sub_v, Namespace) for sub_v in v)):
                        args[key_with_section] = json.dumps(PatchJsonArgParse._handle_namespace(v))
                        args_type[key_with_section] = PatchJsonArgParse.namespace_type
                    elif isinstance(v, Path) or (isinstance(v, list) and all(isinstance(sub_v, Path) for sub_v in v)):
                        args[key_with_section] = json.dumps(PatchJsonArgParse._handle_path(v))
                        args_type[key_with_section] = PatchJsonArgParse.path_type
                    else:
                        args[key_with_section] = str(v)
                except Exception:
                    pass
        cls._current_task._set_parameters(args, __update=True, __parameters_types=args_type)

    @staticmethod
    def _adapt_typehints(original_fn, val, *args, **kwargs):
        if not PatchJsonArgParse._current_task or not running_remotely():
            return original_fn(val, *args, **kwargs)
        return original_fn(val, *args, **kwargs)

    @staticmethod
    def _parse_args(original_fn, obj, *args, **kwargs):
        if not PatchJsonArgParse._current_task:
            return original_fn(obj, *args, **kwargs)
        if len(args) == 1:
            kwargs["args"] = args[0]
            args = []
        if len(args) > 1:
            return original_fn(obj, *args, **kwargs)
        if running_remotely():
            try:
                PatchJsonArgParse._load_task_params()
                params = PatchJsonArgParse.__remote_task_params_dict
                params_namespace = Namespace()
                for k, v in params.items():
                    params_namespace[k] = v
                return params_namespace
            except Exception:
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
                import pytorch_lightning
            except ImportError:
                pytorch_lightning = None
            if subcommand and subcommand in PatchJsonArgParse._args and pytorch_lightning:
                subcommand_args = flatten_dictionary(
                    PatchJsonArgParse._args[subcommand],
                    prefix=subcommand + PatchJsonArgParse._commands_sep,
                    sep=PatchJsonArgParse._commands_sep,
                )
                del PatchJsonArgParse._args[subcommand]
                PatchJsonArgParse._args.update(subcommand_args)
            PatchJsonArgParse._args = {k: v for k, v in PatchJsonArgParse._args.items()}
            PatchJsonArgParse._update_task_args()
        except Exception:
            pass
        return parsed_args

    @staticmethod
    def _load_task_params():
        if not PatchJsonArgParse.__remote_task_params:
            from clearml import Task

            t = Task.get_task(task_id=get_remote_task_id())
            # noinspection PyProtectedMember
            PatchJsonArgParse.__remote_task_params = t._get_task_property("hyperparams") or {}
            params_dict = t.get_parameters(backwards_compatibility=False, cast=True)
            for key, section_param in PatchJsonArgParse.__remote_task_params[PatchJsonArgParse._section_name].items():
                if section_param.type == PatchJsonArgParse.namespace_type:
                    params_dict[
                        "{}/{}".format(PatchJsonArgParse._section_name, key)
                    ] = PatchJsonArgParse._get_namespace_from_json(section_param.value)
                elif section_param.type == PatchJsonArgParse.path_type:
                    params_dict[
                        "{}/{}".format(PatchJsonArgParse._section_name, key)
                    ] = PatchJsonArgParse._get_path_from_json(section_param.value)
                elif (not section_param.type or section_param.type == "NoneType") and not section_param.value:
                    params_dict["{}/{}".format(PatchJsonArgParse._section_name, key)] = None
            skip = len(PatchJsonArgParse._section_name) + 1
            PatchJsonArgParse.__remote_task_params_dict = {
                k[skip:]: v
                for k, v in params_dict.items()
                if k.startswith(PatchJsonArgParse._section_name + PatchJsonArgParse._args_sep)
            }

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
