import sys

import six
from pathlib2 import Path

from ..frameworks.base_bind import PatchBaseModelIO
from ..frameworks import _patched_call, WeightsFileHandler, _Empty
from ..import_bind import PostImportHookPatching
from ...config import running_remotely
from ...model import Framework


class PatchCatBoostModelIO(PatchBaseModelIO):
    __main_task = None
    __patched = None 


    @staticmethod
    def update_current_task(task, **kwargs):
        PatchCatBoostModelIO.__main_task = task
        PatchCatBoostModelIO._patch_model_io()
        PostImportHookPatching.add_on_import('catboost', PatchCatBoostModelIO._patch_model_io)

    @staticmethod
    def _patch_model_io():
        if PatchCatBoostModelIO.__patched:
            return
        if 'catboost' not in sys.modules:
            return
        PatchCatBoostModelIO.__patched = True
        try:
            import catboost
            bst = catboost.CatBoost
            bst.save_model = _patched_call(bst.save_model, PatchCatBoostModelIO._save)
            bst.load_model = _patched_call(bst.load_model, PatchCatBoostModelIO._load)
        except ImportError:
            pass
        except Exception:
            pass

    @staticmethod
    def _save(original_fn, obj, f, *args, **kwargs):
        # see https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
        ret = original_fn(obj, f, *args, **kwargs)
        if not PatchCatBoostModelIO.__main_task:
            return ret
        if isinstance(f, six.string_types):
            filename = f
        else:
            filename = None
        # give the model a descriptive name based on the file name
        # noinspection PyBroadException
        try:
            model_name = Path(filename).stem
        except Exception:
            model_name = None
        WeightsFileHandler.create_output_model(obj, filename, Framework.catboost, PatchCatBoostModelIO.__main_task,
                                               singlefile=True, model_name=model_name)
        return ret

    @staticmethod
    def _load(original_fn, obj, f, *args, **kwargs):
        # see https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
        if isinstance(f, six.string_types):
            filename = f
        else:
            filename = None

        if not PatchCatBoostModelIO.__main_task:
            return original_fn(obj, f, *args, **kwargs) 

        # register input model
        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            filename = WeightsFileHandler.restore_weights_file(empty, f, Framework.catboost,
                                                               PatchXGBoostModelIO.__main_task)
            model = original_fn(filename or f, *args, **kwargs)
        else:
            model = original_fn(obj, f, *args, **kwargs)
            WeightsFileHandler.restore_weights_file(empty, filename, Framework.catboost,
                PatchCatBoostModelIO.__main_task)
        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass
        return model
