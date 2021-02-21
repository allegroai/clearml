""" Argparse utilities"""
import sys
from copy import copy

from six import PY2
from argparse import ArgumentParser, Namespace

try:
    from argparse import _SubParsersAction
except ImportError:
    _SubParsersAction = type(None)


class PatchArgumentParser:
    _original_parse_args = None
    _original_parse_known_args = None
    _original_add_subparsers = None
    _original_get_value = None
    _add_subparsers_counter = 0
    _current_task = None
    _calling_current_task = False
    _last_parsed_args = None
    _last_arg_parser = None
    _recursion_guard = False

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
        if PatchArgumentParser._recursion_guard:
            return {} if not PatchArgumentParser._original_parse_args else \
                PatchArgumentParser._original_parse_args(self, args=args, namespace=namespace)

        PatchArgumentParser._recursion_guard = True
        try:
            result = PatchArgumentParser._patched_parse_args(
                PatchArgumentParser._original_parse_args, self, args=args, namespace=namespace)
        finally:
            PatchArgumentParser._recursion_guard = False
        return result

    @staticmethod
    def parse_known_args(self, args=None, namespace=None):
        if PatchArgumentParser._recursion_guard:
            return {} if not PatchArgumentParser._original_parse_args else \
                PatchArgumentParser._original_parse_known_args(self, args=args, namespace=namespace)

        PatchArgumentParser._recursion_guard = True
        try:
            result = PatchArgumentParser._patched_parse_args(
                PatchArgumentParser._original_parse_known_args, self, args=args, namespace=namespace)
        finally:
            PatchArgumentParser._recursion_guard = False
        return result

    @staticmethod
    def _patched_parse_args(original_parse_fn, self, args=None, namespace=None):
        current_task = PatchArgumentParser._current_task
        # if we are running remotely, we always have a task id, so we better patch the argparser as soon as possible.
        if not current_task:
            from ..config import running_remotely, get_remote_task_id
            if running_remotely():
                # this will cause the current_task() to set PatchArgumentParser._current_task
                from clearml import Task
                # noinspection PyBroadException
                try:
                    current_task = Task.get_task(task_id=get_remote_task_id())
                    # make sure we do not store back the values
                    # (we will do that when we actually call parse args)
                    # this will make sure that if we have args we should not track we know them
                    # noinspection PyProtectedMember
                    current_task._arguments.exclude_parser_args({'*': True})
                except Exception:
                    pass
        # automatically connect to current task:
        if current_task:
            from ..config import running_remotely

            if PatchArgumentParser._calling_current_task:
                # if we are here and running remotely by now we should try to parse the arguments
                parsed_args = None
                if original_parse_fn:
                    parsed_args = original_parse_fn(self, args=args, namespace=namespace)
                    PatchArgumentParser._add_last_parsed_args(self, parsed_args)
                return parsed_args or PatchArgumentParser._last_parsed_args[-1]

            PatchArgumentParser._calling_current_task = True
            # Store last instance and result
            PatchArgumentParser._add_last_arg_parser(self)
            parsed_args = parsed_args_str = None
            # parse if we are running in dev mode
            if not running_remotely() and original_parse_fn:
                parsed_args = original_parse_fn(self, args=args, namespace=namespace)
                parsed_args_str = PatchArgumentParser._add_last_parsed_args(self, parsed_args)

            # noinspection PyBroadException
            try:
                # sync to/from task
                # noinspection PyProtectedMember
                current_task._connect_argparse(
                    self, args=args, namespace=namespace,
                    parsed_args=parsed_args_str[0] if isinstance(parsed_args_str, tuple) else parsed_args_str
                )
            except Exception:
                pass

            # sync back and parse
            if running_remotely():
                if original_parse_fn:
                    # if we are running python2 check if we have subparsers,
                    # if we do we need to patch the args, because there is no default subparser
                    if PY2:
                        import itertools

                        def _get_sub_parsers_defaults(subparser, prev=[]):
                            actions_grp = [v._actions for v in subparser.choices.values()] if isinstance(
                                subparser, _SubParsersAction) else [subparser._actions]
                            _sub_parsers_defaults = [[subparser]] if hasattr(
                                subparser, 'default') and subparser.default else []
                            for actions in actions_grp:
                                _sub_parsers_defaults += [_get_sub_parsers_defaults(v, prev)
                                                          for v in actions if isinstance(v, _SubParsersAction) and
                                                          hasattr(v, 'default') and v.default]

                            return list(itertools.chain.from_iterable(_sub_parsers_defaults))

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

                    parsed_args = original_parse_fn(self, args=args, namespace=namespace)
                    PatchArgumentParser._add_last_parsed_args(self, parsed_args)
                else:
                    # we should never get here
                    parsed_args = parsed_args_str or {}
                    PatchArgumentParser._add_last_parsed_args(self, parsed_args)

            PatchArgumentParser._calling_current_task = False
            return parsed_args

        # Store last instance and result
        PatchArgumentParser._add_last_arg_parser(self)
        parsed_args = {} if not original_parse_fn else original_parse_fn(self, args=args, namespace=namespace)
        PatchArgumentParser._add_last_parsed_args(self, parsed_args)
        return parsed_args

    @staticmethod
    def _add_last_parsed_args(parser, parsed_args):
        if hasattr(parser, '_parsed_arg_string_lookup'):
            if isinstance(parsed_args, tuple):
                parsed_args_namespace = copy(parsed_args[0])
                parsed_args = (parsed_args_namespace, parsed_args[1])
            else:
                parsed_args_namespace = copy(parsed_args)
                parsed_args = (parsed_args_namespace, [])

            # cast arguments in parsed_args_namespace entries to str
            if parsed_args_namespace and isinstance(parsed_args_namespace, Namespace):
                for k, v in parser._parsed_arg_string_lookup.items():  # noqa
                    if hasattr(parsed_args_namespace, k):
                        if isinstance(getattr(parsed_args_namespace, k, None), list) and not isinstance(v, list):
                            v = [v]
                        setattr(
                            parsed_args_namespace, k,
                            v if isinstance(v, list) else str(v)
                        )

        if not PatchArgumentParser._last_parsed_args or parsed_args not in PatchArgumentParser._last_parsed_args:
            PatchArgumentParser._last_parsed_args = (PatchArgumentParser._last_parsed_args or []) + [parsed_args]
        return parsed_args

    @staticmethod
    def _add_last_arg_parser(a_argparser):
        PatchArgumentParser._last_arg_parser = (PatchArgumentParser._last_arg_parser or []) + [a_argparser]

    @staticmethod
    def _get_value(self, action, arg_string):
        if not hasattr(self, '_parsed_arg_string_lookup'):
            setattr(self, '_parsed_arg_string_lookup', dict())
        k = str(action.dest)
        if k not in self._parsed_arg_string_lookup:
            self._parsed_arg_string_lookup[k] = arg_string
        else:
            self._parsed_arg_string_lookup[k] = \
                (self._parsed_arg_string_lookup[k]
                 if isinstance(self._parsed_arg_string_lookup[k], list)
                 else [self._parsed_arg_string_lookup[k]]) + [arg_string]
        return PatchArgumentParser._original_get_value(self, action, arg_string)


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
    if hasattr(sys.modules['argparse'].ArgumentParser, '_get_value'):
        PatchArgumentParser._original_get_value = sys.modules['argparse'].ArgumentParser._get_value
        sys.modules['argparse'].ArgumentParser._get_value = PatchArgumentParser._get_value


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
    if not PatchArgumentParser._last_arg_parser or not PatchArgumentParser._last_parsed_args:
        return []

    return [(parser, args[0] if isinstance(args, tuple) else args)
            for parser, args in zip(PatchArgumentParser._last_arg_parser, PatchArgumentParser._last_parsed_args)]


def add_params_to_parser(parser, params):
    assert isinstance(parser, ArgumentParser)
    assert isinstance(params, dict)

    def get_type_details(v):
        for t in (int, float, str):
            try:
                _value = t(v)
                return t, _value
            except ValueError:
                continue

    # AJB temporary protection from ui problems sending empty dicts
    params.pop('', None)

    for param, value in params.items():
        _type, type_value = get_type_details(value)
        parser.add_argument('--%s' % param, type=_type, default=type_value)
    return parser
