""" jsonargparse binding utility functions """
from ..config import running_remotely


class PatchJsonArgParse(object):
    _original_parse_call = None
    _task = None

    @classmethod
    def update_current_task(cls, current_task):
        cls._task = current_task
        cls._patch_jsonargparse()

    @classmethod
    def _patch_jsonargparse(cls):
        # already patched
        if cls._original_parse_call:
            return

        # noinspection PyBroadException
        try:
            from jsonargparse import ArgumentParser  # noqa
            cls._original_parse_call = ArgumentParser._parse_common  # noqa
            ArgumentParser._parse_common = cls._patched_parse_known_args
        except Exception:
            # there is no jsonargparse
            pass

    @staticmethod
    def _patched_parse_known_args(self, *args, **kwargs):
        if not PatchJsonArgParse._task:
            return PatchJsonArgParse._original_parse_call(self, *args, **kwargs)

        try:
            from argparse import SUPPRESS
            from jsonargparse.typehints import ActionTypeHint
            from jsonargparse.actions import ActionConfigFile, _ActionSubCommands, \
                _ActionConfigLoad, filter_default_actions  # noqa
            from jsonargparse.util import get_key_value_from_flat_dict, update_key_value_in_flat_dict, \
                namespace_to_dict, _dict_to_flat_namespace  # noqa
        except ImportError:
            # something happened, let's just call the original
            return PatchJsonArgParse._original_parse_call(self, *args, **kwargs)

        def cleanup_actions(cfg, actions, prefix='', skip_none=False, cast_value=False):
            for action in filter_default_actions(actions):
                action_dest = prefix + action.dest
                if (action.help == SUPPRESS and not isinstance(action, _ActionConfigLoad)) or \
                   isinstance(action, ActionConfigFile) or \
                   (skip_none and action_dest in cfg and cfg[action_dest] is None):
                    cfg.pop(action_dest, None)
                elif isinstance(action, _ActionSubCommands):
                    for key, subparser in action.choices.items():
                        cleanup_actions(cfg, subparser._actions, prefix=prefix+key+'.',
                                        skip_none=skip_none, cast_value=cast_value)
                elif cast_value and isinstance(action, ActionTypeHint):
                    value = get_key_value_from_flat_dict(cfg, action_dest)
                    if value is not None and value != {}:
                        if value:
                            parsed_value = action._check_type(value)
                        else:
                            try:
                                parsed_value = action._check_type(None)
                            except TypeError:
                                # try with original empty text
                                parsed_value = action._check_type(value)

                        update_key_value_in_flat_dict(cfg, action_dest, parsed_value)
                elif cast_value and hasattr(action, 'type') and not isinstance(action, _ActionConfigLoad):
                    value = get_key_value_from_flat_dict(cfg, action_dest)
                    try:
                        parsed_value = action.type(value or None) if action.type != str else str(value)
                        update_key_value_in_flat_dict(cfg, action_dest, parsed_value)
                    except (ValueError, TypeError):
                        pass

        if not running_remotely():
            ret = PatchJsonArgParse._original_parse_call(self, *args, **kwargs)

            # noinspection PyBroadException
            try:
                cfg_dict = ret if isinstance(ret, dict) else namespace_to_dict(ret)
                cfg_dict = namespace_to_dict(_dict_to_flat_namespace(cfg_dict))
                cleanup_actions(cfg_dict, actions=self._actions, skip_none=False, cast_value=False)
            except Exception:
                cfg_dict = None

            # store / sync arguments
            if cfg_dict is not None:
                PatchJsonArgParse._task.connect(cfg_dict, name='Args')
        else:
            cfg_dict = PatchJsonArgParse._task.get_parameters_as_dict().get('Args', None)
            if cfg_dict is not None:
                if 'cfg' in kwargs:
                    cleanup_actions(cfg_dict, actions=self._actions, skip_none=False, cast_value=True)
                    kwargs['cfg'].update(cfg_dict)
                else:
                    print('Warning failed applying jsonargparse configuration')

            ret = PatchJsonArgParse._original_parse_call(self, *args, **kwargs)

        return ret
