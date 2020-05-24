import yaml

from six import PY2
from argparse import _StoreAction, ArgumentError, _StoreConstAction, _SubParsersAction, SUPPRESS
from copy import copy

from ...utilities.args import call_original_argparser


class _Arguments(object):
    _prefix_sep = '/'
    # TODO: separate dict and argparse after we add UI support
    _prefix_dict = 'dict' + _prefix_sep
    _prefix_args = 'argparse' + _prefix_sep
    _prefix_tf_defines = 'TF_DEFINE' + _prefix_sep

    class _ProxyDictWrite(dict):
        """ Dictionary wrapper that updates an arguments instance on any item set in the dictionary """

        def __init__(self, arguments, *args, **kwargs):
            super(_Arguments._ProxyDictWrite, self).__init__(*args, **kwargs)
            self._arguments = arguments

        def __setitem__(self, key, value):
            super(_Arguments._ProxyDictWrite, self).__setitem__(key, value)
            if self._arguments:
                self._arguments.copy_from_dict(self)

    class _ProxyDictReadOnly(dict):
        """ Dictionary wrapper that prevents modifications to the dictionary """

        def __init__(self, arguments, *args, **kwargs):
            super(_Arguments._ProxyDictReadOnly, self).__init__(*args, **kwargs)
            self._arguments = arguments

        def __setitem__(self, key, value):
            if self._arguments:
                param_dict = self._arguments.copy_to_dict({key: value})
                value = param_dict.get(key, value)
            super(_Arguments._ProxyDictReadOnly, self).__setitem__(key, value)

    def __init__(self, task):
        super(_Arguments, self).__init__()
        self._task = task
        self._exclude_parser_args = {}

    def exclude_parser_args(self, excluded_args):
        self._exclude_parser_args = excluded_args or {}

    def set_defaults(self, *dicts, **kwargs):
        self._task.set_parameters(*dicts, **kwargs)

    def add_argument(self, option_strings, type=None, default=None, help=None):
        if not option_strings:
            raise Exception('Expected at least one argument name (option string)')
        name = option_strings[0].strip('- \t') if isinstance(option_strings, list) else option_strings.strip('- \t')
        # TODO: add argparse prefix
        # name = self._prefix_args + name
        self._task.set_parameter(name=name, value=default, description=help)

    def connect(self, parser):
        self._task.connect_argparse(parser)

    @classmethod
    def _add_to_defaults(cls, a_parser, defaults, a_args=None, a_namespace=None, a_parsed_args=None):
        actions = [
            a for a in a_parser._actions
            if isinstance(a, _StoreAction) or isinstance(a, _StoreConstAction)
        ]
        args_dict = {}
        try:
            if isinstance(a_parsed_args, dict):
                args_dict = a_parsed_args
            else:
                if a_parsed_args:
                    args_dict = a_parsed_args.__dict__
                else:
                    args_dict = call_original_argparser(a_parser, args=a_args, namespace=a_namespace).__dict__
            defaults_ = {
                a.dest: args_dict.get(a.dest) if (args_dict.get(a.dest) is not None) else ''
                for a in actions
            }
        except Exception:
            # don't crash us if we failed parsing the inputs
            defaults_ = {
                a.dest: a.default if a.default is not None else ''
                for a in actions
            }

        full_args_dict = copy(defaults)
        full_args_dict.update(args_dict)
        defaults.update(defaults_)

        # deal with sub parsers
        sub_parsers = [
            a for a in a_parser._actions
            if isinstance(a, _SubParsersAction)
        ]
        for sub_parser in sub_parsers:
            if sub_parser.dest and sub_parser.dest != SUPPRESS:
                defaults[sub_parser.dest] = full_args_dict.get(sub_parser.dest) or ''
            for choice in sub_parser.choices.values():
                # recursively parse
                defaults = cls._add_to_defaults(
                    a_parser=choice,
                    defaults=defaults,
                    a_parsed_args=a_parsed_args or full_args_dict
                )

        return defaults

    def copy_defaults_from_argparse(self, parser, args=None, namespace=None, parsed_args=None):
        task_defaults = {}
        self._add_to_defaults(parser, task_defaults, args, namespace, parsed_args)

        # Make sure we didn't miss anything
        if parsed_args:
            for k, v in parsed_args.__dict__.items():
                if k not in task_defaults:
                    # do not change this comparison because isinstance(type(v), type(None)) === False
                    if type(v) == type(None):
                        task_defaults[k] = ''
                    elif type(v) in (str, int, float, bool, list):
                        task_defaults[k] = v

        # Verify arguments
        for k, v in task_defaults.items():
            try:
                if type(v) is list:
                    task_defaults[k] = str(v)
                elif type(v) not in (str, int, float, bool):
                    task_defaults[k] = str(v)
            except Exception:
                del task_defaults[k]

        # Skip excluded arguments, Add prefix, TODO: add argparse prefix
        # task_defaults = dict([(self._prefix_args + k, v) for k, v in task_defaults.items()
        #                       if k not in self._exclude_parser_args])
        task_defaults = dict([(k, v) for k, v in task_defaults.items() if self._exclude_parser_args.get(k, True)])
        # Store to task
        self._task.update_parameters(task_defaults)

    @classmethod
    def _find_parser_action(cls, a_parser, name):
        # find by name
        _actions = [(a_parser, a) for a in a_parser._actions if a.dest == name]
        if _actions:
            return _actions
        # iterate over subparsers
        _actions = []
        sub_parsers = [a for a in a_parser._actions if isinstance(a, _SubParsersAction)]
        for sub_parser in sub_parsers:
            for choice in sub_parser.choices.values():
                # recursively parse
                _action = cls._find_parser_action(choice, name)
                if _action:
                    _actions.extend(_action)
        return _actions

    def copy_to_parser(self, parser, parsed_args):
        # todo: change to argparse prefix only
        # task_arguments = dict([(k[len(self._prefix_args):], v) for k, v in self._task.get_parameters().items()
        #                        if k.startswith(self._prefix_args)])
        task_arguments = dict([(k, v) for k, v in self._task.get_parameters().items()
                               if not k.startswith(self._prefix_tf_defines) and self._exclude_parser_args.get(k, True)])
        arg_parser_argeuments = {}
        for k, v in task_arguments.items():
            # python2 unicode support
            try:
                v = str(v)
            except:
                pass

            # if we have a StoreTrueAction and the value is either False or Empty or 0 change the default to False
            # with the rest we have to make sure the type is correct
            matched_actions = self._find_parser_action(parser, k)
            for parent_parser, current_action in matched_actions:
                if current_action and isinstance(current_action, _StoreConstAction):
                    # make the default value boolean
                    # first check if False value
                    const_value = current_action.const if current_action.const is not None else (
                        current_action.default if current_action.default is not None else True)
                    const_type = type(const_value)
                    strip_v = str(v).lower().strip()
                    if const_type == bool:
                        if strip_v == 'false' or not strip_v:
                            const_value = False
                        elif strip_v == 'true':
                            const_value = True
                        else:
                            # first try to cast to integer
                            try:
                                const_value = int(strip_v)
                            except ValueError:
                                pass
                    else:
                        const_value = strip_v
                    # then cast to const type (might be boolean)
                    try:
                        const_value = const_type(const_value)
                        current_action.const = const_value
                    except ValueError:
                        pass
                    if current_action.default is not None or const_value not in (None, ''):
                        arg_parser_argeuments[k] = const_value
                elif current_action and (current_action.nargs in ('+', '*') or isinstance(current_action.nargs, int)):
                    try:
                        v = yaml.load(v.strip(), Loader=yaml.SafeLoader)
                        if not isinstance(v, (list, tuple)):
                            # do nothing, we have no idea what happened
                            pass
                        elif current_action.type:
                            v = [current_action.type(a) for a in v]
                        elif current_action.default:
                            v_type = type(current_action.default[0])
                            v = [v_type(a) for a in v]

                        if current_action.default is not None or v not in (None, ''):
                            arg_parser_argeuments[k] = v
                    except Exception:
                        pass
                elif current_action and not current_action.type:
                    # cast manually if there is no type
                    var_type = type(current_action.default)
                    # if we have an int, we should cast to float, because it is more generic
                    if var_type == int:
                        var_type = float
                    elif var_type == type(None):  # do not change! because isinstance(var_type, type(None)) === False
                        var_type = str
                    # now we should try and cast the value if we can
                    try:
                        v = var_type(v)
                        # cast back to int if it's the same value
                        if type(current_action.default) == int and int(v) == v:
                            arg_parser_argeuments[k] = v = int(v)
                        elif current_action.default is None and v in (None, ''):
                            # Do nothing, we should leave it as is.
                            pass
                        else:
                            arg_parser_argeuments[k] = v
                    except Exception:
                        # if we failed, leave as string
                        arg_parser_argeuments[k] = v

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
                        arg_parser_argeuments[k] = v
                elif current_action and current_action.type:
                    arg_parser_argeuments[k] = v
                    try:
                        if current_action.default is None and current_action.type != str and not v:
                            arg_parser_argeuments[k] = v = None
                        elif current_action.default == current_action.type(v):
                            # this will make sure that if we have type float and default value int,
                            # we will keep the type as int, just like the original argparser
                            arg_parser_argeuments[k] = v = current_action.default
                        else:
                            arg_parser_argeuments[k] = v = current_action.type(v)
                    except:
                        pass

                # add as default
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

        # if we already have an instance of parsed args, we should update its values
        if parsed_args:
            for k, v in arg_parser_argeuments.items():
                if parsed_args.get(k) is not None or v not in (None, ''):
                    setattr(parsed_args, k, v)
        parser.set_defaults(**arg_parser_argeuments)

    def copy_from_dict(self, dictionary, prefix=None):
        # TODO: add dict prefix
        prefix = prefix or ''  # self._prefix_dict
        if prefix:
            prefix_dictionary = dict([(prefix + k, v) for k, v in dictionary.items()])
            cur_params = dict([(k, v) for k, v in self._task.get_parameters().items() if not k.startswith(prefix)])
            cur_params.update(prefix_dictionary)
            self._task.set_parameters(cur_params)
        else:
            self._task.update_parameters(dictionary)
        if not isinstance(dictionary, self._ProxyDictWrite):
            return self._ProxyDictWrite(self, **dictionary)
        return dictionary

    def copy_to_dict(self, dictionary, prefix=None):
        # iterate over keys and merge values according to parameter type in dictionary
        # TODO: add dict prefix
        prefix = prefix or ''  # self._prefix_dict
        if prefix:
            parameters = dict([(k[len(prefix):], v) for k, v in self._task.get_parameters().items()
                               if k.startswith(prefix)])
        else:
            parameters = dict([(k, v) for k, v in self._task.get_parameters().items()
                               if not k.startswith(self._prefix_tf_defines)])

        for k, v in dictionary.items():
            # if key is not present in the task's parameters, assume we didn't get this far when running
            # in non-remote mode, and just add it to the task's parameters
            if k not in parameters:
                self._task.set_parameter(k, v)
                continue

            param = parameters.get(k, None)
            if param is None:
                continue
            v_type = type(v)
            # assume more general purpose type int -> float
            if v_type == int:
                if int(v) != float(v):
                    v_type = float
            elif v_type == bool:
                # cast based on string or int
                try:
                    param = bool(float(param))
                except ValueError:
                    try:
                        param = str(param).lower().strip() == 'true'
                    except ValueError:
                        self._task.log.warning('Failed parsing task parameter %s=%s keeping default %s=%s' %
                                               (str(k), str(param), str(k), str(v)))
                        continue
            elif v_type == list:
                try:
                    p = str(param).strip()
                    param = yaml.load(p, Loader=yaml.SafeLoader)
                except Exception:
                    self._task.log.warning('Failed parsing task parameter %s=%s keeping default %s=%s' %
                                           (str(k), str(param), str(k), str(v)))
                    continue
            elif v_type == tuple:
                try:
                    p = str(param).strip().replace('(', '[', 1)[::-1].replace(')', ']', 1)[::-1]
                    param = tuple(yaml.load(p, Loader=yaml.SafeLoader))
                except Exception:
                    self._task.log.warning('Failed parsing task parameter %s=%s keeping default %s=%s' %
                                           (str(k), str(param), str(k), str(v)))
                    continue
            elif v_type == dict:
                try:
                    p = str(param).strip()
                    param = yaml.load(p, Loader=yaml.SafeLoader)
                except Exception:
                    self._task.log.warning('Failed parsing task parameter %s=%s keeping default %s=%s' %
                                           (str(k), str(param), str(k), str(v)))

            try:
                # do not change this comparison because isinstance(v_type, type(None)) === False
                if v_type == type(None):
                    dictionary[k] = str(param) if param else None
                else:
                    dictionary[k] = v_type(param)
            except Exception:
                self._task.log.warning('Failed parsing task parameter %s=%s keeping default %s=%s' %
                                       (str(k), str(param), str(k), str(v)))
                continue
        # add missing parameters to dictionary
        # for k, v in parameters.items():
        #     if k not in dictionary:
        #         dictionary[k] = v

        if not isinstance(dictionary, self._ProxyDictReadOnly):
            return self._ProxyDictReadOnly(self, **dictionary)
        return dictionary
