import sys

import six
from pathlib2 import Path

from ...binding.frameworks.base_bind import PatchBaseModelIO
from ..frameworks import _patched_call, WeightsFileHandler, _Empty
from ..import_bind import PostImportHookPatching
from ...config import running_remotely
from ...model import Framework


class PatchPyTorchModelIO(PatchBaseModelIO):
    _current_task = None
    __patched = None
    __patched_lightning = None

    @staticmethod
    def update_current_task(task, **_):
        PatchPyTorchModelIO._current_task = task
        if not task:
            return
        PatchPyTorchModelIO._patch_model_io()
        PatchPyTorchModelIO._patch_lightning_io()
        PostImportHookPatching.add_on_import('torch', PatchPyTorchModelIO._patch_model_io)
        PostImportHookPatching.add_on_import('pytorch_lightning', PatchPyTorchModelIO._patch_lightning_io)

    @staticmethod
    def _patch_model_io():
        if PatchPyTorchModelIO.__patched:
            return

        if 'torch' not in sys.modules:
            return

        PatchPyTorchModelIO.__patched = True

        # noinspection PyBroadException
        try:
            import torch  # noqa
            torch.save = _patched_call(torch.save, PatchPyTorchModelIO._save)
            torch.load = _patched_call(torch.load, PatchPyTorchModelIO._load)
            # noinspection PyBroadException
            try:
                # noinspection PyProtectedMember
                torch.jit._script.RecursiveScriptModule.save = _patched_call(torch.jit._script.RecursiveScriptModule.save, PatchPyTorchModelIO._save)
            except BaseException:
                pass

            # no need to worry about recursive calls, _patched_call takes care of that
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, '_save'):
                torch.serialization._save = _patched_call(
                    torch.serialization._save, PatchPyTorchModelIO._save)  # noqa
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, '_load'):
                torch.serialization._load = _patched_call(
                    torch.serialization._load, PatchPyTorchModelIO._load)  # noqa
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, '_legacy_save'):
                torch.serialization._legacy_save = _patched_call(
                    torch.serialization._legacy_save, PatchPyTorchModelIO._save)  # noqa
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, '_legacy_load'):
                torch.serialization._legacy_load = _patched_call(
                    torch.serialization._legacy_load, PatchPyTorchModelIO._load)  # noqa
        except ImportError:
            pass
        except Exception:
            pass  # print('Failed patching pytorch')

    @staticmethod
    def _patch_lightning_io():
        if PatchPyTorchModelIO.__patched_lightning:
            return

        if 'pytorch_lightning' not in sys.modules:
            return

        PatchPyTorchModelIO.__patched_lightning = True

        # noinspection PyBroadException
        try:
            import pytorch_lightning  # noqa

            pytorch_lightning.trainer.Trainer.save_checkpoint = _patched_call(
                pytorch_lightning.trainer.Trainer.save_checkpoint, PatchPyTorchModelIO._save)  # noqa

            pytorch_lightning.trainer.Trainer.restore = _patched_call(
                pytorch_lightning.trainer.Trainer.restore, PatchPyTorchModelIO._load_from_obj)  # noqa
        except ImportError:
            pass
        except Exception:
            pass

        # noinspection PyBroadException
        try:
            import pytorch_lightning  # noqa

            # noinspection PyUnresolvedReferences
            pytorch_lightning.trainer.connectors.checkpoint_connector.CheckpointConnector.save_checkpoint = \
                _patched_call(
                    pytorch_lightning.trainer.connectors.checkpoint_connector.CheckpointConnector.save_checkpoint,
                    PatchPyTorchModelIO._save)  # noqa

            # noinspection PyUnresolvedReferences
            pytorch_lightning.trainer.connectors.checkpoint_connector.CheckpointConnector.restore = \
                _patched_call(
                    pytorch_lightning.trainer.connectors.checkpoint_connector.CheckpointConnector.restore,
                    PatchPyTorchModelIO._load_from_obj)  # noqa
        except ImportError:
            pass
        except Exception:
            pass

    @staticmethod
    def _save(original_fn, obj, f, *args, **kwargs):
        ret = original_fn(obj, f, *args, **kwargs)

        # if there is no main task or this is a nested call
        if not PatchPyTorchModelIO._current_task:
            return ret

        # pytorch-lightning check if rank is zero
        if hasattr(obj, 'is_global_zero'):
            if not obj.is_global_zero:
                return ret
        elif hasattr(obj, 'trainer') and hasattr(obj.trainer, 'is_global_zero'):
            if not obj.trainer.is_global_zero:
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
            model_name = Path(filename).stem if filename is not None else None
        except Exception:
            model_name = None

        WeightsFileHandler.create_output_model(
            obj, filename, Framework.pytorch, PatchPyTorchModelIO._current_task, singlefile=True, model_name=model_name)

        return ret

    @staticmethod
    def _load(original_fn, f, *args, **kwargs):
        # if there is no main task or this is a nested call
        if not PatchPyTorchModelIO._current_task:
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
            filename = WeightsFileHandler.restore_weights_file(
                empty, filename, Framework.pytorch, PatchPyTorchModelIO._current_task)
            model = original_fn(filename or f, *args, **kwargs)
        else:
            # try to load model before registering, in case we fail
            model = original_fn(f, *args, **kwargs)
            WeightsFileHandler.restore_weights_file(
                empty, filename, Framework.pytorch, PatchPyTorchModelIO._current_task)

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass

        return model

    @staticmethod
    def _load_from_obj(original_fn, obj, f, *args, **kwargs):
        # if there is no main task or this is a nested call
        if not PatchPyTorchModelIO._current_task:
            return original_fn(obj, f, *args, **kwargs)

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
            filename = WeightsFileHandler.restore_weights_file(
                empty, filename, Framework.pytorch, PatchPyTorchModelIO._current_task)
            model = original_fn(obj, filename or f, *args, **kwargs)
        else:
            # try to load model before registering, in case we fail
            model = original_fn(obj, f, *args, **kwargs)
            WeightsFileHandler.restore_weights_file(
                empty, filename, Framework.pytorch, PatchPyTorchModelIO._current_task)

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass

        return model
