""" absl-py FLAGS binding utility functions """
import six

from ..backend_interface.task.args import _Arguments
from ..config import running_remotely


class PatchAbsl(object):
    _original_DEFINE_flag = None
    _original_FLAGS_parse_call = None
    _task = None

    @classmethod
    def update_current_task(cls, current_task):
        cls._task = current_task
        cls._patch_absl()

    @classmethod
    def _patch_absl(cls):
        if cls._original_DEFINE_flag:
            return
        # noinspection PyBroadException
        try:
            from absl.flags import _defines
            if six.PY2:
                cls._original_DEFINE_flag = staticmethod(_defines.DEFINE_flag)
            else:
                cls._original_DEFINE_flag = _defines.DEFINE_flag
            _defines.DEFINE_flag = cls._patched_define_flag
        except Exception:
            # there is no absl
            pass

        try:
            from absl.flags._flagvalues import FlagValues
            if six.PY2:
                cls._original_FLAGS_parse_call = staticmethod(FlagValues.__call__)
            else:
                cls._original_FLAGS_parse_call = FlagValues.__call__
            FlagValues.__call__ = cls._patched_FLAGS_parse_call
        except Exception:
            # there is no absl
            pass

        if cls._original_DEFINE_flag:
            try:
                # if absl was already set, let's update our task params
                from absl import flags
                cls._update_current_flags(flags.FLAGS)
            except Exception:
                # there is no absl
                pass

    @staticmethod
    def _patched_define_flag(*args, **kwargs):
        if not PatchAbsl._task or not PatchAbsl._original_DEFINE_flag:
            if PatchAbsl._original_DEFINE_flag:
                return PatchAbsl._original_DEFINE_flag(*args, **kwargs)
            else:
                return None
        # noinspection PyBroadException
        try:
            flag = args[0] if len(args) >= 1 else None
            module_name = args[2] if len(args) >= 3 else None
            param_name = None
            if flag:
                param_name = ((module_name + _Arguments._prefix_sep) if module_name else '') + flag.name
        except Exception:
            flag = None
            param_name = None

        if running_remotely():
            # noinspection PyBroadException
            try:
                if param_name and flag:
                    param_dict = PatchAbsl._task._arguments.copy_to_dict(
                        {param_name: flag.value}, prefix=_Arguments._prefix_tf_defines)
                    flag.value = param_dict.get(param_name, flag.value)
            except Exception:
                pass
            ret = PatchAbsl._original_DEFINE_flag(*args, **kwargs)
        else:
            if flag and param_name:
                value = flag.value
                PatchAbsl._task.update_parameters({_Arguments._prefix_tf_defines + param_name: value}, )
            ret = PatchAbsl._original_DEFINE_flag(*args, **kwargs)
        return ret

    @staticmethod
    def _patched_FLAGS_parse_call(self, *args, **kwargs):
        ret = PatchAbsl._original_FLAGS_parse_call(self, *args, **kwargs)
        # noinspection PyBroadException
        try:
            PatchAbsl._update_current_flags(self)
        except Exception:
            pass
        return ret

    @classmethod
    def _update_current_flags(cls, FLAGS):
        if not cls._task:
            return
        # noinspection PyBroadException
        try:
            if running_remotely():
                param_dict = dict((k, FLAGS[k].value) for k in FLAGS)
                param_dict = cls._task._arguments.copy_to_dict(param_dict, prefix=_Arguments._prefix_tf_defines)
                for k, v in param_dict.items():
                    # noinspection PyBroadException
                    try:
                        parts = k.split(_Arguments._prefix_sep)
                        k = parts[0]
                        if k in FLAGS:
                            FLAGS[k].value = v
                    except Exception:
                        pass
            else:
                # clear previous parameters
                parameters = dict([(k, FLAGS[k].value) for k in FLAGS])
                # noinspection PyBroadException
                try:
                    descriptions = dict([(k, FLAGS[k].help or None) for k in FLAGS])
                except Exception:
                    descriptions = None
                # noinspection PyBroadException
                try:
                    param_types = dict([(k, FLAGS[k].flag_type() or None) for k in FLAGS])
                except Exception:
                    param_types = None
                cls._task._arguments.copy_from_dict(
                    parameters,
                    prefix=_Arguments._prefix_tf_defines,
                    descriptions=descriptions, param_types=param_types,
                )
        except Exception:
            pass
