import sys

import six
from pathlib2 import Path

from ..frameworks.base_bind import PatchBaseModelIO
from ..frameworks import _patched_call, WeightsFileHandler, _Empty
from ..import_bind import PostImportHookPatching
from ...config import running_remotely
from ...model import Framework


class PatchXGBoostModelIO(PatchBaseModelIO):
    __main_task = None
    __patched = None

    @staticmethod
    def update_current_task(task, **kwargs):
        PatchXGBoostModelIO.__main_task = task
        PatchXGBoostModelIO._patch_model_io()
        PostImportHookPatching.add_on_import('xgboost', PatchXGBoostModelIO._patch_model_io)

    @staticmethod
    def _patch_model_io():
        if PatchXGBoostModelIO.__patched:
            return

        if 'xgboost' not in sys.modules:
            return
        PatchXGBoostModelIO.__patched = True
        try:
            import xgboost as xgb
            bst = xgb.Booster
            bst.save_model = _patched_call(bst.save_model, PatchXGBoostModelIO._save)
            bst.load_model = _patched_call(bst.load_model, PatchXGBoostModelIO._load)
        except ImportError:
            pass
        except Exception:
            pass

    @staticmethod
    def _save(original_fn, obj, f, *args, **kwargs):
        ret = original_fn(obj, f, *args, **kwargs)
        if not PatchXGBoostModelIO.__main_task:
            return ret

        if isinstance(f, six.string_types):
            filename = f
        elif hasattr(f, 'name'):
            filename = f.name
            # noinspection PyBroadException
            try:
                f.flush()
            except Exception:
                pass
        else:
            filename = None

        # give the model a descriptive name based on the file name
        # noinspection PyBroadException
        try:
            model_name = Path(filename).stem
        except Exception:
            model_name = None
        WeightsFileHandler.create_output_model(obj, filename, Framework.xgboost, PatchXGBoostModelIO.__main_task,
                                               singlefile=True, model_name=model_name)
        return ret

    @staticmethod
    def _load(original_fn, f, *args, **kwargs):
        if isinstance(f, six.string_types):
            filename = f
        elif hasattr(f, 'name'):
            filename = f.name
        elif len(args) == 1 and isinstance(args[0], six.string_types):
            filename = args[0]
        else:
            filename = None

        if not PatchXGBoostModelIO.__main_task:
            return original_fn(f, *args, **kwargs)

        # register input model
        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            filename = WeightsFileHandler.restore_weights_file(empty, filename, Framework.xgboost,
                                                               PatchXGBoostModelIO.__main_task)
            model = original_fn(filename or f, *args, **kwargs)
        else:
            # try to load model before registering, in case we fail
            model = original_fn(f, *args, **kwargs)
            WeightsFileHandler.restore_weights_file(empty, filename, Framework.xgboost,
                                                    PatchXGBoostModelIO.__main_task)

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass
        return model
