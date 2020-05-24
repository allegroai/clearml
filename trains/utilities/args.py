""" Argparse utilities"""
import sys
from six import PY2
from argparse import ArgumentParser, _SubParsersAction


class PatchArgumentParser:
    _original_parse_args = None
    _original_parse_known_args = None
    _original_add_subparsers = None
    _add_subparsers_counter = 0
    _current_task = None
    _calling_current_task = False
    _last_parsed_args = None
    _last_arg_parser = None

    @staticmethod
    def add_subparsers(self, **kwargs):
        if 'dest' not in kwargs:
            if kwargs.get('title'):
                kwargs['dest'] = '/' + kwargs['title']
            else:
                PatchArgumentParser._add_subparsers_counter += 1
                kwargs['dest'] = '/subparser%d' % PatchArgumentParser._add_subparsers_counter
        return PatchArgumentParser._original_add_subparsers(self, **kwargs)

    @staticmethod
    def parse_args(self, args=None, namespace=None):
        return PatchArgumentParser._patched_parse_args(PatchArgumentParser._original_parse_args,
                                                       self, args=args, namespace=namespace)

    @staticmethod
    def parse_known_args(self, args=None, namespace=None):
        return PatchArgumentParser._patched_parse_args(PatchArgumentParser._original_parse_known_args,
                                                       self, args=args, namespace=namespace)

    @staticmethod
    def _patched_parse_args(original_parse_fn, self, args=None, namespace=None):
        # if we are running remotely, we always have a task id, so we better patch the argparser as soon as possible.
        if not PatchArgumentParser._current_task:
            from ..config import running_remotely
            if running_remotely():
                # this will cause the current_task() to set PatchArgumentParser._current_task
                from trains import Task
                # noinspection PyBroadException
                try:
                    Task.init()
                except Exception:
                    pass
        # automatically connect to current task:
        if PatchArgumentParser._current_task:
            from ..config import running_remotely

            if PatchArgumentParser._calling_current_task:
                # if we are here and running remotely by now we should try to parse the arguments
                if original_parse_fn:
                    PatchArgumentParser._last_parsed_args = \
                        original_parse_fn(self, args=args, namespace=namespace)
                return PatchArgumentParser._last_parsed_args

            PatchArgumentParser._calling_current_task = True
            # Store last instance and result
            PatchArgumentParser._last_arg_parser = self
            parsed_args = None
            # parse if we are running in dev mode
            if not running_remotely() and original_parse_fn:
                parsed_args = original_parse_fn(self, args=args, namespace=namespace)
                PatchArgumentParser._last_parsed_args = parsed_args
            # noinspection PyBroadException
            try:
                # sync to/from task
                PatchArgumentParser._current_task._connect_argparse(self, args=args, namespace=namespace,
                                                                    parsed_args=parsed_args[0]
                                                                    if isinstance(parsed_args, tuple) else parsed_args)
            except Exception:
                pass
            # sync back and parse
            if running_remotely() and original_parse_fn:
                # if we are running python2 check if we have subparsers,
                # if we do we need to patch the args, because there is no default subparser
                if PY2:
                    import itertools

                    def _get_sub_parsers_defaults(subparser, prev=[]):
                        actions_grp = [a._actions for a in subparser.choices.values()] if isinstance(
                            subparser, _SubParsersAction) else [subparser._actions]
                        sub_parsers_defaults = [[subparser]] if hasattr(
                            subparser, 'default') and subparser.default else []
                        for actions in actions_grp:
                            sub_parsers_defaults += [_get_sub_parsers_defaults(a, prev)
                                                     for a in actions if isinstance(a, _SubParsersAction) and
                                                     hasattr(a, 'default') and a.default]

                        return list(itertools.chain.from_iterable(sub_parsers_defaults))
                    sub_parsers_defaults = _get_sub_parsers_defaults(self)
                    if sub_parsers_defaults:
                        if args is None:
                            # args default to the system args
                            import sys as _sys
                            args = _sys.argv[1:]
                        else:
                            args = list(args)
                        # make sure we append the subparsers
                        for a in sub_parsers_defaults:
                            if a.default not in args:
                                args.append(a.default)

                PatchArgumentParser._last_parsed_args = original_parse_fn(self, args=args, namespace=namespace)
            else:
                PatchArgumentParser._last_parsed_args = parsed_args or {}

            PatchArgumentParser._calling_current_task = False
            return PatchArgumentParser._last_parsed_args

        # Store last instance and result
        PatchArgumentParser._last_arg_parser = self
        PatchArgumentParser._last_parsed_args = {} if not original_parse_fn else \
            original_parse_fn(self, args=args, namespace=namespace)
        return PatchArgumentParser._last_parsed_args


def patch_argparse():
    # make sure we only patch once
    if not sys.modules.get('argparse') or hasattr(sys.modules['argparse'].ArgumentParser, '_parse_args_patched'):
        return
    # mark patched argparse
    sys.modules['argparse'].ArgumentParser._parse_args_patched = True
    # patch argparser
    PatchArgumentParser._original_parse_args = sys.modules['argparse'].ArgumentParser.parse_args
    PatchArgumentParser._original_parse_known_args = sys.modules['argparse'].ArgumentParser.parse_known_args
    PatchArgumentParser._original_add_subparsers = sys.modules['argparse'].ArgumentParser.add_subparsers
    sys.modules['argparse'].ArgumentParser.parse_args = PatchArgumentParser.parse_args
    sys.modules['argparse'].ArgumentParser.parse_known_args = PatchArgumentParser.parse_known_args
    sys.modules['argparse'].ArgumentParser.add_subparsers = PatchArgumentParser.add_subparsers


# Notice! we are patching argparser, sop we know if someone parsed arguments before connecting to task
patch_argparse()


def call_original_argparser(self, args=None, namespace=None):
    if PatchArgumentParser._original_parse_args:
        return PatchArgumentParser._original_parse_args(self, args=args, namespace=namespace)


def argparser_parseargs_called():
    return PatchArgumentParser._last_arg_parser is not None


def argparser_update_currenttask(task):
    PatchArgumentParser._current_task = task


def get_argparser_last_args():
    return (PatchArgumentParser._last_arg_parser,
            PatchArgumentParser._last_parsed_args[0] if isinstance(PatchArgumentParser._last_parsed_args, tuple) else
            PatchArgumentParser._last_parsed_args)


def add_params_to_parser(parser, params):
    assert isinstance(parser, ArgumentParser)
    assert isinstance(params, dict)

    def get_type_details(v):
        for t in (int, float, str):
            try:
                value = t(v)
                return t, value
            except ValueError:
                continue

    # AJB temporary protection from ui problems sending empty dicts
    params.pop('', None)

    for param, value in params.items():
        type, type_value = get_type_details(value)
        parser.add_argument('--%s' % param, type=type, default=type_value)
    return parser
