import ast
import copy

try:
    from jsonargparse import ArgumentParser
    from jsonargparse.namespace import Namespace
except ImportError:
    ArgumentParser = None

from ..config import running_remotely, get_remote_task_id
from .frameworks import _patched_call  # noqa


class PatchJsonArgParse(object):
    _args = {}
    _main_task = None
    _args_sep = "/"
    _args_type = {}
    _commands_sep = "."
    _command_type = "jsonargparse.Command"
    _command_name = "subcommand"
    _section_name = "Args"
    __remote_task_params = {}
    __patched = False

    @classmethod
    def patch(cls, task):
        if ArgumentParser is None:
            return

        if task:
            cls._main_task = task
            PatchJsonArgParse._update_task_args()

        if not cls.__patched:
            cls.__patched = True
            ArgumentParser.parse_args = _patched_call(ArgumentParser.parse_args, PatchJsonArgParse._parse_args)

    @classmethod
    def _update_task_args(cls):
        if running_remotely() or not cls._main_task or not cls._args:
            return
        args = {cls._section_name + cls._args_sep + k: v for k, v in cls._args.items()}
        args_type = {cls._section_name + cls._args_sep + k: v for k, v in cls._args_type.items()}
        cls._main_task._set_parameters(args, __update=True, __parameters_types=args_type)

    @staticmethod
    def _parse_args(original_fn, obj, *args, **kwargs):
        if len(args) == 1:
            kwargs["args"] = args[0]
            args = []
        if len(args) > 1:
            return original_fn(obj, *args, **kwargs)
        if running_remotely():
            try:
                PatchJsonArgParse._load_task_params()
                params = PatchJsonArgParse.__remote_task_params_dict
                for k, v in params.items():
                    if v == '':
                        v = None
                    # noinspection PyBroadException
                    try:
                        v = ast.literal_eval(v)
                    except Exception:
                        pass
                    params[k] = v
                params = PatchJsonArgParse.__unflatten_dict(params)
                params = PatchJsonArgParse.__nested_dict_to_namespace(params)
                return params
            except Exception:
                return original_fn(obj, **kwargs)
        orig_parsed_args = original_fn(obj, **kwargs)
        # noinspection PyBroadException
        try:
            parsed_args = vars(copy.deepcopy(orig_parsed_args))
            for ns_name, ns_val in parsed_args.items():
                if not isinstance(ns_val, (Namespace, list)):
                    PatchJsonArgParse._args[ns_name] = str(ns_val)
                    if ns_name == PatchJsonArgParse._command_name:
                        PatchJsonArgParse._args_type[ns_name] = PatchJsonArgParse._command_type
                else:
                    ns_val = PatchJsonArgParse.__nested_namespace_to_dict(ns_val)
                    ns_val = PatchJsonArgParse.__flatten_dict(ns_val, parent_name=ns_name)
                    for k, v in ns_val.items():
                        PatchJsonArgParse._args[k] = str(v)
            PatchJsonArgParse._update_task_args()
        except Exception:
            pass
        return orig_parsed_args

    @staticmethod
    def _load_task_params():
        if not PatchJsonArgParse.__remote_task_params:
            from clearml import Task

            t = Task.get_task(task_id=get_remote_task_id())
            # noinspection PyProtectedMember
            PatchJsonArgParse.__remote_task_params = t._get_task_property("hyperparams") or {}
            params_dict = t.get_parameters(backwards_compatibility=False)
            skip = len(PatchJsonArgParse._section_name) + 1
            PatchJsonArgParse.__remote_task_params_dict = {
                k[skip:]: v
                for k, v in params_dict.items()
                if k.startswith(PatchJsonArgParse._section_name + PatchJsonArgParse._args_sep)
            }

    @staticmethod
    def __nested_namespace_to_dict(namespace):
        if isinstance(namespace, list):
            return [PatchJsonArgParse.__nested_namespace_to_dict(n) for n in namespace]
        if not isinstance(namespace, Namespace):
            return namespace
        namespace = vars(namespace)
        for k, v in namespace.items():
            namespace[k] = PatchJsonArgParse.__nested_namespace_to_dict(v)
        return namespace

    @staticmethod
    def __nested_dict_to_namespace(dict_):
        if isinstance(dict_, list):
            return [PatchJsonArgParse.__nested_dict_to_namespace(d) for d in dict_]
        if not isinstance(dict_, dict):
            return dict_
        for k, v in dict_.items():
            dict_[k] = PatchJsonArgParse.__nested_dict_to_namespace(v)
        return Namespace(**dict_)

    @staticmethod
    def __flatten_dict(dict_, parent_name=None):
        if isinstance(dict_, list):
            if parent_name:
                return {parent_name: [PatchJsonArgParse.__flatten_dict(d) for d in dict_]}
            return [PatchJsonArgParse.__flatten_dict(d) for d in dict_]
        if not isinstance(dict_, dict):
            if parent_name:
                return {parent_name: dict_}
            return dict_
        result = {}
        for k, v in dict_.items():
            v = PatchJsonArgParse.__flatten_dict(v, parent_name=k)
            if isinstance(v, dict):
                for flattened_k, flattened_v in v.items():
                    if parent_name:
                        result[parent_name + PatchJsonArgParse._commands_sep + flattened_k] = flattened_v
                    else:
                        result[flattened_k] = flattened_v
            else:
                result[k] = v
        return result

    @staticmethod
    def __unflatten_dict(dict_):
        if isinstance(dict_, list):
            return [PatchJsonArgParse.__unflatten_dict(d) for d in dict_]
        if not isinstance(dict_, dict):
            return dict_
        result = {}
        for k, v in dict_.items():
            keys = k.split(PatchJsonArgParse._commands_sep)
            current_dict = result
            for k_part in keys[:-1]:
                if k_part not in current_dict:
                    current_dict[k_part] = {}
                current_dict = current_dict[k_part]
            current_dict[keys[-1]] = PatchJsonArgParse.__unflatten_dict(v)
        return result
