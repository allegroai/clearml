import ast
import six

try:
    from jsonargparse import ArgumentParser
    from jsonargparse.namespace import Namespace
except ImportError:
    ArgumentParser = None

from ..config import running_remotely, get_remote_task_id
from .frameworks import _patched_call  # noqa
from ..utilities.proxy_object import flatten_dictionary


class PatchJsonArgParse(object):
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

    @classmethod
    def _update_task_args(cls):
        if running_remotely() or not cls._current_task or not cls._args:
            return
        args = {cls._section_name + cls._args_sep + k: v for k, v in cls._args.items()}
        args_type = {cls._section_name + cls._args_sep + k: v for k, v in cls._args_type.items()}
        cls._current_task._set_parameters(args, __update=True, __parameters_types=args_type)

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
                    if v == "":
                        v = None
                    # noinspection PyBroadException
                    try:
                        v = ast.literal_eval(v)
                    except Exception:
                        pass
                    params_namespace[k] = PatchJsonArgParse.__namespace_eval(v)
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
            PatchJsonArgParse._args = {k: str(v) for k, v in PatchJsonArgParse._args.items()}
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
            params_dict = t.get_parameters(backwards_compatibility=False)
            skip = len(PatchJsonArgParse._section_name) + 1
            PatchJsonArgParse.__remote_task_params_dict = {
                k[skip:]: v
                for k, v in params_dict.items()
                if k.startswith(PatchJsonArgParse._section_name + PatchJsonArgParse._args_sep)
            }

    @staticmethod
    def __namespace_eval(val):
        if isinstance(val, six.string_types) and val.startswith("Namespace(") and val[-1] == ")":
            val = val[len("Namespace("):]
            val = val[:-1]
            return Namespace(PatchJsonArgParse.__namespace_eval(ast.literal_eval("{" + val + "}")))
        if isinstance(val, list):
            return [PatchJsonArgParse.__namespace_eval(v) for v in val]
        if isinstance(val, dict):
            for k, v in val.items():
                val[k] = PatchJsonArgParse.__namespace_eval(v)
            return val
        return val
