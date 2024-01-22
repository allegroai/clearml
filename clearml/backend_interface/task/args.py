import yaml

from enum import Enum
from inspect import isfunction
from six import PY2
from argparse import _StoreAction, ArgumentError, _StoreConstAction, _SubParsersAction, _AppendAction, SUPPRESS  # noqa
from copy import copy

from typing import Tuple, Type, Union

from ...backend_api import Session
from ...binding.args import call_original_argparser
from ...utilities.proxy_object import get_type_from_basic_type_str


class _Arguments(object):
    _prefix_sep = '/'
    # TODO: separate dict and argparse after we add UI support
    _prefix_args = 'Args' + _prefix_sep
    _prefix_tf_defines = 'TF_DEFINE' + _prefix_sep

    class _ProxyDictWrite(dict):
        """ Dictionary wrapper that updates an arguments instance on any item set in the dictionary """

        def __init__(self, __arguments, __section_name, *args, **kwargs):
            super(_Arguments._ProxyDictWrite, self).__init__(*args, **kwargs)
            self._arguments = __arguments
            self._section_name = (__section_name.strip(_Arguments._prefix_sep) + _Arguments._prefix_sep) \
                if __section_name else None

        def __setitem__(self, key, value):
            super(_Arguments._ProxyDictWrite, self).__setitem__(key, value)
            if self._arguments:
                self._arguments.copy_from_dict(self, prefix=self._section_name)

    class _ProxyDictReadOnly(dict):
        """ Dictionary wrapper that prevents modifications to the dictionary """

        def __init__(self, __arguments, __section_name, *args, **kwargs):
            super(_Arguments._ProxyDictReadOnly, self).__init__(*args, **kwargs)
            self._arguments = __arguments
            self._section_name = (__section_name.strip(_Arguments._prefix_sep) + _Arguments._prefix_sep) \
                if __section_name else None

        def __setitem__(self, key, value):
            if self._arguments:
                param_dict = self._arguments.copy_to_dict({key: value}, prefix=self._section_name)
                value = param_dict.get(key, value)
            super(_Arguments._ProxyDictReadOnly, self).__setitem__(key, value)

    def __init__(self, task):
        super(_Arguments, self).__init__()
        self._task = task
        self._exclude_parser_args = {}

    def exclude_parser_args(self, excluded_args):
        """
        You can use a dictionary for fined grained control of connected
        arguments. The dictionary keys are argparse variable names and the values are booleans.
        The ``False`` value excludes the specified argument from the Task's parameter section.
        Keys missing from the dictionary default to ``True``, and an empty dictionary defaults to ``False``.

        :param excluded_args: dict
        """
        self._exclude_parser_args = excluded_args or {}

    def set_defaults(self, *dicts, **kwargs):
        prefix = self._prefix_args if Session.check_min_api_version('2.9') else None
        # noinspection PyProtectedMember
        self._task._set_parameters(*dicts, __parameters_prefix=prefix, **kwargs)

    def add_argument(self, option_strings, type=None, default=None, help=None):
        if not option_strings:
            raise Exception('Expected at least one argument name (option string)')
        name = option_strings[0].strip('- \t') if isinstance(option_strings, list) else option_strings.strip('- \t')
        if Session.check_min_api_version('2.9'):
            name = self._prefix_args + name
        self._task.set_parameter(name=name, value=default, description=help, value_type=type)

    def connect(self, parser):
        self._task.connect_argparse(parser)

    @classmethod
    def _add_to_defaults(cls, a_parser, defaults, descriptions, arg_types,
                         a_args=None, a_namespace=None, a_parsed_args=None):
        # noinspection PyProtectedMember
        actions = [
            a for a in a_parser._actions
            if isinstance(a, _StoreAction) or isinstance(a, _StoreConstAction)
        ]
        args_dict = {}
        # noinspection PyBroadException
        try:
            if isinstance(a_parsed_args, dict):
                args_dict = a_parsed_args
            else:
                if a_parsed_args:
                    args_dict = a_parsed_args.__dict__
                else:
                    args_dict = call_original_argparser(a_parser, args=a_args, namespace=a_namespace).__dict__
            defaults_ = {
                a.dest: cls.__cast_arg(args_dict.get(a.dest), a.type) for a in actions
            }
        except Exception:
            # don't crash us if we failed parsing the inputs
            defaults_ = {
                a.dest: a.default if a.default is not None else ''
                for a in actions
            }

        desc_ = {
            a.dest: str(a.help) or (
                '{}default: {}'.format('choices: {}, '.format(a.choices) if a.choices else '',
                                       defaults_.get(a.dest, '')))
            for a in actions}
        descriptions.update(desc_)
        types_ = {a.dest: (a.type or None) for a in actions}
        arg_types.update(types_)

        full_args_dict = copy(defaults)
        full_args_dict.update(args_dict)
        defaults.update(defaults_)

        # deal with sub parsers
        # noinspection PyProtectedMember
        sub_parsers = [
            a for a in a_parser._actions
            if isinstance(a, _SubParsersAction)
        ]
        for sub_parser in sub_parsers:
            if sub_parser.dest and sub_parser.dest != SUPPRESS:
                defaults[sub_parser.dest] = full_args_dict.get(sub_parser.dest) or ''
            for choice in sub_parser.choices.values():
                # recursively parse
                defaults, descriptions, arg_types = cls._add_to_defaults(
                    a_parser=choice,
                    defaults=defaults,
                    descriptions=descriptions,
                    arg_types=arg_types,
                    a_parsed_args=a_parsed_args or full_args_dict
                )

        return defaults, descriptions, arg_types

    def copy_defaults_from_argparse(self, parser, args=None, namespace=None, parsed_args=None):
        task_defaults, task_defaults_descriptions, task_defaults_types = \
            self._get_defaults_from_argparse(parser, args=args, namespace=namespace, parsed_args=parsed_args)

        # Store to task
        self._task.update_parameters(
            task_defaults,
            __parameters_descriptions=task_defaults_descriptions,
            __parameters_types=task_defaults_types
        )

    def _get_defaults_from_argparse(self, parser, args=None, namespace=None, parsed_args=None):
        task_defaults = {}
        task_defaults_descriptions = {}
        task_defaults_types = {}
        self._add_to_defaults(parser, task_defaults, task_defaults_descriptions, task_defaults_types,
                              args, namespace, parsed_args)

        # Make sure we didn't miss anything
        if parsed_args:
            for k, v in parsed_args.__dict__.items():
                if k not in task_defaults:
                    # do not change this comparison because isinstance(type(v), type(None)) === False
                    if v is None:
                        task_defaults[k] = ''
                    elif type(v) in (str, int, float, bool, list):
                        task_defaults[k] = v

        # Verify arguments
        for k, v in task_defaults.items():
            # noinspection PyBroadException
            try:
                if type(v) is list:
                    task_defaults[k] = str(v)
                elif type(v) not in (str, int, float, bool):
                    task_defaults[k] = str(v)
            except Exception:
                del task_defaults[k]

        # Skip excluded arguments, Add prefix.
        if Session.check_min_api_version('2.9'):
            include_all = self._exclude_parser_args.get("*", True)
            task_defaults = dict(
                [(self._prefix_args + k, v) for k, v in task_defaults.items()
                 if self._exclude_parser_args.get(k, include_all)])
            task_defaults_descriptions = dict(
                [(self._prefix_args + k, v) for k, v in task_defaults_descriptions.items()
                 if self._exclude_parser_args.get(k, include_all)])
            task_defaults_types = dict(
                [(self._prefix_args + k, v) for k, v in task_defaults_types.items()
                 if self._exclude_parser_args.get(k, include_all)])
        else:
            task_defaults = dict(
                [(k, v) for k, v in task_defaults.items() if self._exclude_parser_args.get(k, True)])

        return task_defaults, task_defaults_descriptions, task_defaults_types

    @classmethod
    def _find_parser_action(cls, a_parser, name):
        # find by name
        # noinspection PyProtectedMember
        _actions = [(a_parser, a) for a in a_parser._actions if a.dest == name]
        if _actions:
            return _actions
        # iterate over subparsers
        _actions = []
        # noinspection PyProtectedMember
        sub_parsers = [a for a in a_parser._actions if isinstance(a, _SubParsersAction)]
        for sub_parser in sub_parsers:
            for choice in sub_parser.choices.values():
                # recursively parse
                _action = cls._find_parser_action(choice, name)
                if _action:
                    _actions.extend(_action)
        return _actions

    @classmethod
    def _remove_req_flag_from_mutex_groups(cls, parser):
        # noinspection PyBroadException
        try:
            # noinspection PyProtectedMember
            for group in parser._mutually_exclusive_groups:
                group.required = False
        except Exception:
            pass

    def copy_to_parser(self, parser, parsed_args):
        def cast_to_bool_int(value, strip=False):
            a_strip_v = value if not strip else str(value).lower().strip()
            if a_strip_v == 'false' or not a_strip_v:
                return False
            elif a_strip_v == 'true':
                return True
            else:
                # first try to cast to integer
                try:
                    return int(a_strip_v)
                except (ValueError, TypeError):
                    return None

        # Change to argparse prefix only
        prefix = self._prefix_args if Session.check_min_api_version('2.9') else ''
        task_arguments = dict([(k[len(prefix):], v) for k, v in self._task.get_parameters().items()
                               if k.startswith(prefix) and
                               self._exclude_parser_args.get(k[len(prefix):], True)])
        arg_parser_arguments = {}
        for k, v in task_arguments.items():
            # python2 unicode support
            # noinspection PyBroadException
            try:
                v = str(v)
            except Exception:
                pass

            # if we have a StoreTrueAction and the value is either False or Empty or 0 change the default to False
            # with the rest we have to make sure the type is correct
            matched_actions = self._find_parser_action(parser, k)
            for parent_parser, current_action in matched_actions:
                if current_action and current_action.default == SUPPRESS and not v:
                    # this value should be kept suppressed, do nothing
                    v = SUPPRESS
                elif current_action and isinstance(current_action, _StoreConstAction):
                    # make the default value boolean
                    # first check if False value
                    const_value = current_action.const if current_action.const is not None else (
                        current_action.default if current_action.default is not None else True)
                    const_type = type(const_value)
                    strip_v = str(v).lower().strip()
                    if const_type == bool:
                        bool_value = cast_to_bool_int(strip_v)
                        if bool_value is not None:
                            const_value = bool_value
                    else:
                        const_value = strip_v
                    # then cast to const type (might be boolean)
                    try:
                        const_value = const_type(const_value)
                        current_action.const = const_value
                    except ValueError:
                        pass
                    if current_action.default is not None or const_value not in (None, ''):
                        arg_parser_arguments[k] = const_value
                elif current_action and (
                        current_action.nargs in ('+', '*') or isinstance(current_action.nargs, int) or
                        isinstance(current_action, _AppendAction)):
                    # noinspection PyBroadException
                    try:
                        v = yaml.load(v.strip(), Loader=yaml.SafeLoader)
                        if not isinstance(v, (list, tuple)):
                            # we have no idea what happened, just put into a list.
                            v = [v] if v else None
                        # casting
                        if v:
                            if current_action.type:
                                v = [current_action.type(a) if str(a) != str(None) else None for a in v]
                            elif current_action.default:
                                v_type = type(current_action.default[0])
                                if v_type != type(None):  # noqa
                                    v = [v_type(a) if str(a) != str(None) else None for a in v]

                        if current_action.default is not None or v not in (None, ''):
                            arg_parser_arguments[k] = v
                    except Exception:
                        if self._task and self._task.log:
                            self._task.log.warning(
                                'Failed parsing task parameter {}="{}" keeping default {}={}'.format(
                                    k, v, k, current_action.default))
                        continue

                elif current_action and not current_action.type:
                    # cast manually if there is no type
                    var_type = type(current_action.default)
                    # if we have an int, we should cast to float, because it is more generic
                    if var_type == int:
                        var_type = float
                    elif var_type == type(None):  # noqa: E721 - do not change!
                        # because isinstance(var_type, type(None)) === False
                        var_type = str
                    elif var_type not in (bool, int, float, str, dict, list, tuple) and \
                            str(self.__cast_arg(current_action.default)) == str(v):
                        # there is nothing we can Do, just leave with the default
                        continue

                    # now we should try and cast the value if we can
                    # noinspection PyBroadException
                    try:
                        # Since we have no actual var type here, we check if the string presentation of the original
                        # default value and the new value are the same, if they are,
                        # we should just use the original default value
                        if str(v) == str(current_action.default):
                            v = current_action.default
                        else:
                            v = var_type(v)
                        # cast back to int if it's the same value
                        if type(current_action.default) == int and int(v) == v:
                            arg_parser_arguments[k] = v = int(v)
                        elif current_action.default is None and v in (None, ''):
                            # Do nothing, we should leave it as is.
                            pass
                        else:
                            arg_parser_arguments[k] = v
                    except Exception:
                        # if we failed, leave as string
                        arg_parser_arguments[k] = v
                elif current_action and current_action.type == bool:
                    # parser.set_defaults cannot cast string `False`/`True` to boolean properly,
                    # so we have to do it manually here
                    strip_v = str(v).lower().strip()
                    if strip_v == 'false' or not strip_v:
                        v = False
                    elif strip_v == 'true':
                        v = True
                    else:
                        # else, try to cast to integer
                        try:
                            v = int(strip_v)
                        except ValueError:
                            pass
                    if v not in (None, ''):
                        arg_parser_arguments[k] = v
                elif current_action and current_action.type:
                    # if we have an action type and value (v) is None, and cannot be casted, leave as is
                    if isfunction(current_action.type) and not v:  # noqa
                        # noinspection PyBroadException
                        try:
                            v = current_action.type(v)
                        except Exception:
                            continue
                    elif current_action.type == str and current_action.default is None and v in (None, ''):
                        # if the type is str and the default is None, and we stored empty string,
                        # do not change the value (i.e. leave it as None)
                        continue
                    elif current_action.type == str and isinstance(current_action.default, bool):
                        # this will take care of values that can be strings or boolean with default of boolean
                        # and the default value is kept the same
                        bool_value = cast_to_bool_int(v, strip=True)
                        if bool_value is not None and current_action.default == bool(bool_value):
                            continue
                    elif str(current_action.default) == v:
                        # if we changed nothing, leave it as is (i.e. default value)
                        v = current_action.default

                    arg_parser_arguments[k] = v
                    # noinspection PyBroadException
                    try:
                        if current_action.default is None and current_action.type != str and not v:
                            arg_parser_arguments[k] = v = None
                        elif (not isfunction(current_action.type)
                              and current_action.default == current_action.type(v)) \
                                or (isfunction(current_action.type) and
                                    str(self.__cast_arg(current_action.default)) == str(v)):
                            # this will make sure that if we have type float and default value int,
                            # we will keep the type as int, just like the original argparser
                            arg_parser_arguments[k] = v = current_action.default
                        else:
                            arg_parser_arguments[k] = v = current_action.type(v)
                    except Exception:
                        pass

                # add as default
                # noinspection PyBroadException
                try:
                    if current_action and isinstance(current_action, _SubParsersAction):
                        if v not in (None, '') or current_action.default not in (None, ''):
                            current_action.default = v
                        current_action.required = False
                    elif current_action and isinstance(current_action, _StoreAction):
                        if v not in (None, '') or current_action.default not in (None, ''):
                            current_action.default = v
                        current_action.required = False
                        # python2 doesn't support defaults for positional arguments, unless used with nargs=?
                        if PY2 and not current_action.nargs:
                            current_action.nargs = '?'
                    else:
                        # do not add parameters that do not exist in argparser, they might be the dict
                        pass
                except ArgumentError:
                    pass
                except Exception:
                    pass

        self._remove_req_flag_from_mutex_groups(parser)

        # if API supports sections, we can update back the Args section with all the missing default
        if Session.check_min_api_version('2.9') and not self._exclude_parser_args.get('*', None):
            # noinspection PyBroadException
            try:
                task_defaults, task_defaults_descriptions, task_defaults_types = \
                    self._get_defaults_from_argparse(parser, parsed_args=parsed_args)
                # select what's missing, and update it
                missing_keys = [k for k in task_defaults.keys() if k[len(self._prefix_args):] not in task_arguments]
                task_defaults = {k: v for k, v in task_defaults.items() if k in missing_keys}
                task_defaults_descriptions = {k: v for k, v in task_defaults_descriptions.items() if k in missing_keys}
                task_defaults_types = {k: v for k, v in task_defaults_types.items() if k in missing_keys}
                if task_defaults:
                    self._task.update_parameters(
                        task_defaults,
                        __parameters_descriptions=task_defaults_descriptions,
                        __parameters_types=task_defaults_types
                    )
            except Exception:
                pass

        # if we already have an instance of parsed args, we should update its values
        # this instance should already contain our defaults
        if parsed_args:
            for k, v in arg_parser_arguments.items():
                cur_v = getattr(parsed_args, k, None)
                # it should not happen...
                if cur_v != v and (cur_v is not None or v not in (None, '')):
                    setattr(parsed_args, k, v)
        parser.set_defaults(**arg_parser_arguments)

    def copy_from_dict(self, dictionary, prefix=None, descriptions=None, param_types=None):
        # add dict prefix
        prefix = prefix  # or self._prefix_dict
        if prefix:
            prefix = prefix.strip(self._prefix_sep) + self._prefix_sep
            if descriptions:
                descriptions = dict((prefix+k, v) for k, v in descriptions.items())
            if param_types:
                param_types = dict((prefix + k, v) for k, v in param_types.items())
            # this will only set the specific section
            self._task.update_parameters(
                dictionary,
                __parameters_prefix=prefix,
                __parameters_descriptions=descriptions,
                __parameters_types=param_types,
            )
        else:
            self._task.update_parameters(
                dictionary,
                __parameters_prefix=prefix,
                __parameters_descriptions=descriptions,
                __parameters_types=param_types,
            )
        if not isinstance(dictionary, self._ProxyDictWrite):
            return self._ProxyDictWrite(self, prefix, **dictionary)
        return dictionary

    def copy_to_dict(self, dictionary, prefix=None):
        # iterate over keys and merge values according to parameter type in dictionary
        # add dict prefix
        prefix = prefix  # or self._prefix_dict
        if prefix:
            prefix = prefix.strip(self._prefix_sep) + self._prefix_sep
            parameters = dict([(k[len(prefix):], v) for k, v in self._task.get_parameters().items()
                               if k.startswith(prefix)])
            # noinspection PyProtectedMember
            parameters_type = {
                k: p.type
                for k, p in ((self._task._get_task_property('hyperparams', raise_on_error=False) or {}).get(
                    prefix[:-len(self._prefix_sep)]) or {}).items()
                if p.type
            }
        else:
            parameters_type = {}
            parameters = dict([(k, v) for k, v in self._task.get_parameters().items()
                               if not k.startswith(self._prefix_tf_defines)])

        for k, v in dictionary.items():
            # if key is not present in the task's parameters, assume we didn't get this far when running
            # in non-remote mode, and just add it to the task's parameters
            if k not in parameters:
                self._task.set_parameter((prefix or '') + k, v)
                continue

            param = parameters.get(k, None)
            if param is None:
                continue

            # if default value is not specified, allow casting based on what we have on the Task
            if v is not None:
                v_type = type(v)
            elif parameters_type.get(k):
                v_type_str = parameters_type.get(k)
                v_type = get_type_from_basic_type_str(v_type_str)
            else:
                # this will be type(None), we deal with it later
                v_type = type(v)

            def warn_failed_parsing():
                self._task.log.warning(
                    "Failed parsing task parameter {}={} keeping default {}={}".format(k, param, k, v)
                )

            # if parameter is empty and default value is None, keep as None
            if param == '' and v is None:
                v_type = type(None)
            elif v_type == int:  # assume more general purpose type int -> float
                if v is not None and int(v) != float(v):
                    v_type = float
            elif v_type == bool:
                # cast based on string or int
                try:
                    param = bool(float(param))
                except ValueError:
                    try:
                        param = str(param).lower().strip() == 'true'
                    except ValueError:
                        warn_failed_parsing()
                        continue
            elif v_type == list:
                # noinspection PyBroadException
                try:
                    p = str(param).strip()
                    param = yaml.load(p, Loader=FloatSafeLoader)
                except Exception:
                    warn_failed_parsing()
                    continue
            elif v_type == tuple:
                # noinspection PyBroadException
                try:
                    p = str(param).strip().replace('(', '[', 1)[::-1].replace(')', ']', 1)[::-1]
                    param = tuple(yaml.load(p, Loader=FloatSafeLoader))
                except Exception:
                    warn_failed_parsing()
                    continue
            elif v_type == dict:
                # noinspection PyBroadException
                try:
                    p = str(param).strip()
                    param = yaml.load(p, Loader=FloatSafeLoader)
                except Exception:
                    warn_failed_parsing()
            elif issubclass(v_type, Enum):
                # noinspection PyBroadException
                try:
                    param = getattr(v_type, param).value
                except Exception:
                    warn_failed_parsing()
                    continue

            # noinspection PyBroadException
            try:
                # do not change this comparison because isinstance(v_type, type(None)) === False
                if v_type == type(None):  # noqa: E721
                    dictionary[k] = str(param) if param else None
                elif v_type == str:  # noqa: E721
                    dictionary[k] = v_type(param)
                else:
                    dictionary[k] = None if param == '' else v_type(param)
            except Exception:
                warn_failed_parsing()
                continue
        # add missing parameters to dictionary
        # for k, v in parameters.items():
        #     if k not in dictionary:
        #         dictionary[k] = v

        if not isinstance(dictionary, self._ProxyDictReadOnly):
            return self._ProxyDictReadOnly(self, prefix, **dictionary)
        return dictionary

    @classmethod
    def get_supported_types(cls, as_str=False):
        # type: (bool) -> Union[Type, Tuple[str]]
        """
        Return the basic types supported by Argument casting
        :param as_str: if True, return string cast of the types
        :return: List of type objects supported for auto casting (serializing to string)
        """
        supported_types = (int, float, bool, str, list, tuple, Enum)
        if as_str:
            return tuple([str(t) for t in supported_types])

        return supported_types

    @classmethod
    def __cast_arg(cls, arg, dtype=None):
        if arg is None or callable(arg):
            return ''
        # If this an instance, just store the type
        if str(hex(id(arg))) in str(arg):
            return str(type(arg))
        if dtype in (float, int) and isinstance(arg, list):
            return [None if a is None else dtype(a) for a in arg]
        return arg


# fix pyyaml Numbers in scientific notation without dot are parsed as string
# [#173](https://github.com/yaml/pyyaml/issues/173) [PR #174](https://github.com/yaml/pyyaml/pull/174)
# noinspection PyBroadException
try:
    import re
    from yaml.resolver import Resolver

    class FloatResolver(Resolver):
        pass

    FloatResolver.add_implicit_resolver(
        'tag:yaml.org,2002:float',
        re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
                        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+][0-9]+)
                        |\.[0-9_]+(?:[eE][-+][0-9]+)?
                        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                        |[-+]?\.(?:inf|Inf|INF)
                        |\.(?:nan|NaN|NAN))$''', re.X),
        list('-+0123456789.'))

    class FloatSafeLoader(yaml.SafeLoader, FloatResolver):
        def __init__(self, stream):
            super(FloatSafeLoader, self).__init__(stream)
            FloatResolver.__init__(self)

except Exception:
    FloatSafeLoader = yaml.SafeLoader
