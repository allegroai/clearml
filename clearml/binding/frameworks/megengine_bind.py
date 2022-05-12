import sys

import six
from pathlib2 import Path

from ..frameworks.base_bind import PatchBaseModelIO
from ..frameworks import _patched_call, WeightsFileHandler, _Empty
from ..import_bind import PostImportHookPatching
from ...model import Framework


class PatchMegEngineModelIO(PatchBaseModelIO):
    _current_task = None
    __patched = None

    @staticmethod
    def update_current_task(task, **_):
        PatchMegEngineModelIO._current_task = task
        if not task:
            return
        PatchMegEngineModelIO._patch_model_io()
        PostImportHookPatching.add_on_import(
            'megengine', PatchMegEngineModelIO._patch_model_io
        )

    @staticmethod
    def _patch_model_io():
        if PatchMegEngineModelIO.__patched:
            return

        if 'megengine' not in sys.modules:
            return

        PatchMegEngineModelIO.__patched = True

        # noinspection PyBroadException
        try:
            import megengine as mge  # noqa
            mge.save = _patched_call(mge.save, PatchMegEngineModelIO._save)
            mge.load = _patched_call(mge.load, PatchMegEngineModelIO._load)

            # no need to worry about recursive calls, _patched_call takes care of that  # noqa
            if hasattr(mge, 'serialization') and hasattr(mge.serialization, 'save'):  # noqa
                mge.serialization.save = _patched_call(
                    mge.serialization.save, PatchMegEngineModelIO._save
                )
            if hasattr(mge, 'serialization') and hasattr(mge.serialization, 'load'):  # noqa
                mge.serialization.load = _patched_call(
                    mge.serialization.load, PatchMegEngineModelIO._load,
                )
        except ImportError:
            pass
        except Exception:
            pass

    @staticmethod
    def _save(original_fn, obj, f, *args, **kwargs):
        ret = original_fn(obj, f, *args, **kwargs)

        # if there is no main task or this is a nested call
        if not PatchMegEngineModelIO._current_task:
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
                    # Probably a BufferedRandom object that has no meaningful name (still no harm flushing)  # noqa
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
            obj, filename, Framework.megengine,
            PatchMegEngineModelIO._current_task,
            singlefile=True, model_name=model_name,
        )

        return ret

    @staticmethod
    def _load(original_fn, f, *args, **kwargs):
        # if there is no main task or this is a nested call
        if not PatchMegEngineModelIO._current_task:
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
        # try to load model before registering, in case we fail
        model = original_fn(f, *args, **kwargs)
        WeightsFileHandler.restore_weights_file(
            empty, filename, Framework.megengine,
            PatchMegEngineModelIO._current_task
        )

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
        if not PatchMegEngineModelIO._current_task:
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
        # try to load model before registering, in case we fail
        model = original_fn(obj, f, *args, **kwargs)
        WeightsFileHandler.restore_weights_file(
            empty, filename, Framework.megengine,
            PatchMegEngineModelIO._current_task,
        )

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass

        return model
