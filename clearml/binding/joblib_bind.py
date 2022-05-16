import sys
import warnings
from functools import partial

import six
from pathlib2 import Path

from .import_bind import PostImportHookPatching
from ..binding.frameworks import _patched_call, _Empty, WeightsFileHandler
from ..config import running_remotely
from ..debugging.log import LoggerRoot
from ..model import Framework
from ..utilities.lowlevel.file_access import get_filename_from_file_object, buffer_writer_close_cb


class PatchedJoblib(object):
    _patched_joblib = False
    _patched_sk_joblib = False
    _current_task = None

    @staticmethod
    def patch_joblib():
        # try manually
        PatchedJoblib._patch_joblib()
        # register callback
        PostImportHookPatching.add_on_import('joblib',
                                             PatchedJoblib._patch_joblib)
        PostImportHookPatching.add_on_import('sklearn',
                                             PatchedJoblib._patch_joblib)

    @staticmethod
    def _patch_joblib():
        # noinspection PyBroadException
        try:
            if not PatchedJoblib._patched_joblib and 'joblib' in sys.modules:
                PatchedJoblib._patched_joblib = True
                try:
                    import joblib
                except ImportError:
                    joblib = None

                if joblib:
                    joblib.numpy_pickle._write_fileobject = _patched_call(
                        joblib.numpy_pickle._write_fileobject,
                        partial(PatchedJoblib._write_fileobject, joblib.numpy_pickle))
                    joblib.numpy_pickle._read_fileobject = _patched_call(
                        joblib.numpy_pickle._read_fileobject, PatchedJoblib._load)
                    joblib.numpy_pickle.NumpyPickler.__init__ = _patched_call(
                        joblib.numpy_pickle.NumpyPickler.__init__,
                        PatchedJoblib._numpypickler)

            if not PatchedJoblib._patched_sk_joblib and 'sklearn' in sys.modules:
                PatchedJoblib._patched_sk_joblib = True
                try:
                    import sklearn  # noqa: F401
                    # avoid deprecation warning, we must import sklearn before, so we could catch it
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        from sklearn.externals import joblib as sk_joblib
                except ImportError:
                    sk_joblib = None

                if sk_joblib:
                    sk_joblib.numpy_pickle._write_fileobject = _patched_call(
                        sk_joblib.numpy_pickle._write_fileobject,
                        partial(PatchedJoblib._write_fileobject, sk_joblib.numpy_pickle))
                    sk_joblib.numpy_pickle._read_fileobject = _patched_call(
                        sk_joblib.numpy_pickle._read_fileobject, PatchedJoblib._load)
                    sk_joblib.numpy_pickle.NumpyPickler.__init__ = _patched_call(
                        sk_joblib.numpy_pickle.NumpyPickler.__init__,
                        PatchedJoblib._numpypickler)

        except Exception:
            return False
        return True

    @staticmethod
    def update_current_task(task):
        PatchedJoblib._current_task = task
        if not task:
            return
        PatchedJoblib.patch_joblib()

    @staticmethod
    def _dump(original_fn, obj, f, *args, **kwargs):
        ret = original_fn(obj, f, *args, **kwargs)
        if not PatchedJoblib._current_task:
            return ret
        PatchedJoblib._register_dump(obj, f)
        return ret

    @staticmethod
    def _numpypickler(original_fn, obj, f, *args, **kwargs):
        ret = original_fn(obj, f, *args, **kwargs)
        if not PatchedJoblib._current_task:
            return ret

        fname = f if isinstance(f, six.string_types) else None
        fileobj = ret if isinstance(f, six.string_types) else f

        if fileobj and hasattr(fileobj, 'close'):
            def callback(*_):
                PatchedJoblib._register_dump(obj, fname or fileobj)

            if isinstance(fname, six.string_types) or hasattr(fileobj, 'name'):
                buffer_writer_close_cb(fileobj, callback)
        else:
            PatchedJoblib._register_dump(obj, f)

        return ret

    @staticmethod
    def _write_fileobject(obj, original_fn, f, *args, **kwargs):
        ret = original_fn(f, *args, **kwargs)
        if not PatchedJoblib._current_task:
            return ret

        fname = f if isinstance(f, six.string_types) else None
        fileobj = ret if isinstance(f, six.string_types) else f

        if fileobj and hasattr(fileobj, 'close'):
            def callback(*_):
                PatchedJoblib._register_dump(obj, fname or fileobj)

            if isinstance(fname, six.string_types) or hasattr(fileobj, 'name'):
                buffer_writer_close_cb(fileobj, callback)
        else:
            PatchedJoblib._register_dump(obj, f)
        return ret

    @staticmethod
    def _register_dump(obj, f):
        filename = get_filename_from_file_object(f, flush=True)
        if not filename:
            return

        # give the model a descriptive name based on the file name
        # noinspection PyBroadException
        try:
            model_name = Path(filename).stem
        except Exception:
            model_name = None
        current_framework = PatchedJoblib.get_model_framework(obj)
        WeightsFileHandler.create_output_model(obj, filename, current_framework,
                                               PatchedJoblib._current_task, singlefile=True, model_name=model_name)

    @staticmethod
    def _load(original_fn, f, *args, **kwargs):
        if not PatchedJoblib._current_task:
            return original_fn(f, *args, **kwargs)

        filename = get_filename_from_file_object(f, flush=False)

        # register input model
        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            # we assume scikit-learn, for the time being
            current_framework = Framework.scikitlearn
            filename = WeightsFileHandler.restore_weights_file(empty, filename, current_framework,
                                                               PatchedJoblib._current_task)
            model = original_fn(filename or f, *args, **kwargs)
        else:
            # try to load model before registering, in case we fail
            model = original_fn(f, *args, **kwargs)
            current_framework = PatchedJoblib.get_model_framework(model)
            WeightsFileHandler.restore_weights_file(empty, filename, current_framework,
                                                    PatchedJoblib._current_task)

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass
        return model

    @staticmethod
    def get_model_framework(obj):
        framework = Framework.scikitlearn
        object_orig_module = None
        # noinspection PyBroadException
        try:
            object_orig_module = obj.__module__ if hasattr(obj, '__module__') else obj.__package__
            model = object_orig_module.partition(".")[0]
            if model == 'sklearn':
                framework = Framework.scikitlearn
            elif model == 'xgboost':
                framework = Framework.xgboost
            else:
                framework = Framework.scikitlearn
        except Exception:
            LoggerRoot.get_base_logger().debug(
                "Can't get model framework {}, model framework will be: {} ".format(object_orig_module, framework))
        finally:
            return framework
