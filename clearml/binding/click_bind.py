try:
    from click.core import Command, Option, Argument, Group, Context  # noqa
    from click.types import BoolParamType  # noqa
except ImportError:
    Command = None

from .frameworks import _patched_call  # noqa
from ..config import running_remotely, get_remote_task_id
from ..utilities.dicts import cast_str_to_bool


class PatchClick:
    _args = {}
    _args_desc = {}
    _args_type = {}
    _num_commands = 0
    _command_type = 'click.Command'
    _section_name = 'Args'
    _current_task = None
    __remote_task_params = None
    __remote_task_params_dict = {}
    __patched = False

    @classmethod
    def patch(cls, task=None):
        if Command is None:
            return

        cls._current_task = task
        if task:
            PatchClick._update_task_args()

        if not cls.__patched:
            cls.__patched = True
            Command.__init__ = _patched_call(Command.__init__, PatchClick._command_init)
            Command.parse_args = _patched_call(Command.parse_args, PatchClick._parse_args)
            Context.__init__ = _patched_call(Context.__init__, PatchClick._context_init)

    @classmethod
    def args(cls):
        # remove prefix and main command
        if cls._num_commands == 1:
            cmd = sorted(cls._args.keys())[0]
            skip = len(cmd)+1
        else:
            skip = 0

        _args = {cls._section_name+'/'+k[skip:]: v for k, v in cls._args.items() if k[skip:]}
        _args_type = {cls._section_name+'/'+k[skip:]: v for k, v in cls._args_type.items() if k[skip:]}
        _args_desc = {cls._section_name+'/'+k[skip:]: v for k, v in cls._args_desc.items() if k[skip:]}

        return _args, _args_type, _args_desc

    @classmethod
    def _update_task_args(cls):
        if running_remotely() or not cls._current_task or not cls._args:
            return
        param_val, param_types, param_desc = cls.args()
        # noinspection PyProtectedMember
        cls._current_task._set_parameters(
            param_val,
            __update=True,
            __parameters_descriptions=param_desc,
            __parameters_types=param_types
        )

    @staticmethod
    def _command_init(original_fn, self, *args, **kwargs):
        if isinstance(self, (Command, Group)) and 'name' in kwargs:
            if isinstance(self, Command):
                PatchClick._num_commands += 1
            if not running_remotely():
                name = kwargs['name']
                if name:
                    PatchClick._args[name] = False
                    if isinstance(self, Command):
                        PatchClick._args_type[name] = PatchClick._command_type
                    # maybe we should take it post initialization
                    if kwargs.get('help'):
                        PatchClick._args_desc[name] = str(kwargs.get('help'))

                for option in kwargs.get('params') or []:
                    if not option or not isinstance(option, (Option, Argument)) or \
                            not getattr(option, 'expose_value', True):
                        continue
                    # store default value
                    PatchClick._args[name+'/'+option.name] = str(option.default or '')
                    # store value type
                    if option.type is not None:
                        PatchClick._args_type[name+'/'+option.name] = str(option.type)
                    # store value help
                    if getattr(option, 'help', None):
                        PatchClick._args_desc[name+'/'+option.name] = str(option.help)

        return original_fn(self, *args, **kwargs)

    @staticmethod
    def _parse_args(original_fn, self, *args, **kwargs):
        if running_remotely() and isinstance(self, Command) and isinstance(self, Group):
            command = PatchClick._load_task_params()
            if command:
                init_args = kwargs['args'] if 'args' in kwargs else args[1]
                init_args = [command] + (init_args[1:] if init_args else [])
                if 'args' in kwargs:
                    kwargs['args'] = init_args
                else:
                    args = (args[0], init_args) + args[2:]

        ret = original_fn(self, *args, **kwargs)

        if isinstance(self, Command):
            ctx = kwargs.get('ctx') or args[0]
            if running_remotely():
                PatchClick._load_task_params()
                for p in self.params:
                    name = '{}/{}'.format(self.name, p.name) if PatchClick._num_commands > 1 else p.name
                    value = PatchClick.__remote_task_params_dict.get(name)
                    ctx.params[p.name] = p.process_value(
                        ctx, cast_str_to_bool(value, strip=True) if isinstance(p.type, BoolParamType) else value)
            else:
                if not isinstance(self, Group):
                    PatchClick._args[self.name] = True
                for k, v in ctx.params.items():
                    # store passed value
                    PatchClick._args[self.name + '/' + str(k)] = str(v or '')

                PatchClick._update_task_args()
        return ret

    @staticmethod
    def _context_init(original_fn, self, *args, **kwargs):
        if running_remotely():
            kwargs['resilient_parsing'] = True
        return original_fn(self, *args, **kwargs)

    @staticmethod
    def _load_task_params():
        if not PatchClick.__remote_task_params:
            from clearml import Task
            t = Task.get_task(task_id=get_remote_task_id())
            # noinspection PyProtectedMember
            PatchClick.__remote_task_params = t._get_task_property('hyperparams') or {}
            params_dict = t.get_parameters(backwards_compatibility=False)
            skip = len(PatchClick._section_name)+1
            PatchClick.__remote_task_params_dict = {
                k[skip:]: v for k, v in params_dict.items()
                if k.startswith(PatchClick._section_name+'/')
            }

        params = PatchClick.__remote_task_params
        command = [
            p.name for p in params['Args'].values()
            if p.type == PatchClick._command_type and cast_str_to_bool(p.value, strip=True)]
        return command[0] if command else None


# patch click before anything
PatchClick.patch()
