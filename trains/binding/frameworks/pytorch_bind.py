import sys

import six
from pathlib2 import Path

from ...binding.frameworks.base_bind import PatchBaseModelIO
from ..frameworks import _patched_call, WeightsFileHandler, _Empty
from ..import_bind import PostImportHookPatching
from ...config import running_remotely
from ...model import Framework


class PatchPyTorchModelIO(PatchBaseModelIO):
    __main_task = None
    __patched = None

    @staticmethod
    def update_current_task(task, **_):
        PatchPyTorchModelIO.__main_task = task
        PatchPyTorchModelIO._patch_model_io()
        PostImportHookPatching.add_on_import('torch', PatchPyTorchModelIO._patch_model_io)

    @staticmethod
    def _patch_model_io():
        if PatchPyTorchModelIO.__patched:
            return

        if 'torch' not in sys.modules:
            return

        PatchPyTorchModelIO.__patched = True

        # noinspection PyBroadException
        try:
            # hack: make sure tensorflow.__init__ is called
            import torch
            torch.save = _patched_call(torch.save, PatchPyTorchModelIO._save)
            torch.load = _patched_call(torch.load, PatchPyTorchModelIO._load)
        except ImportError:
            pass
        except Exception:
            pass  # print('Failed patching pytorch')

    @staticmethod
    def _save(original_fn, obj, f, *args, **kwargs):
        ret = original_fn(obj, f, *args, **kwargs)
        if not PatchPyTorchModelIO.__main_task:
            return ret

        # noinspection PyBroadException
        try:
            if isinstance(f, six.string_types):
                filename = f
            elif hasattr(f, 'as_posix'):
                filename = f.as_posix()
            elif hasattr(f, 'name'):
                # noinspection PyBroadException
                try:
                    f.flush()
                except Exception:
                    pass

                if not isinstance(f.name, six.string_types):
                    # Probably a BufferedRandom object that has no meaningful name (still no harm flushing)
                    return ret

                filename = f.name
            else:
                filename = None
        except Exception:
            filename = None

        # give the model a descriptive name based on the file name
        # noinspection PyBroadException
        try:
            model_name = Path(filename).stem
        except Exception:
            model_name = None

        WeightsFileHandler.create_output_model(obj, filename, Framework.pytorch, PatchPyTorchModelIO.__main_task,
                                               singlefile=True, model_name=model_name)

        return ret

    @staticmethod
    def _load(original_fn, f, *args, **kwargs):
        if not PatchPyTorchModelIO.__main_task:
            return original_fn(f, *args, **kwargs)

        # noinspection PyBroadException
        try:
            if isinstance(f, six.string_types):
                filename = f
            elif hasattr(f, 'as_posix'):
                filename = f.as_posix()
            elif hasattr(f, 'name'):
                filename = f.name
            else:
                filename = None
        except Exception:
            filename = None

        # register input model
        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            filename = WeightsFileHandler.restore_weights_file(empty, filename, Framework.pytorch,
                                                               PatchPyTorchModelIO.__main_task)
            model = original_fn(filename or f, *args, **kwargs)
        else:
            # try to load model before registering, in case we fail
            model = original_fn(f, *args, **kwargs)
            WeightsFileHandler.restore_weights_file(empty, filename, Framework.pytorch,
                                                    PatchPyTorchModelIO.__main_task)

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass
        return model
