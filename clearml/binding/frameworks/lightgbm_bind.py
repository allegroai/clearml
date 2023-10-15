import sys

import six
from pathlib2 import Path

from ..frameworks.base_bind import PatchBaseModelIO
from ..frameworks import _patched_call, WeightsFileHandler, _Empty
from ..import_bind import PostImportHookPatching
from ...config import running_remotely
from ...model import Framework


class PatchLIGHTgbmModelIO(PatchBaseModelIO):
    _current_task = None
    __patched = None

    @staticmethod
    def update_current_task(task, **kwargs):
        PatchLIGHTgbmModelIO._current_task = task
        if not task:
            return
        PatchLIGHTgbmModelIO._patch_model_io()
        PostImportHookPatching.add_on_import('lightgbm', PatchLIGHTgbmModelIO._patch_model_io)

    @staticmethod
    def _patch_model_io():
        if PatchLIGHTgbmModelIO.__patched:
            return

        if 'lightgbm' not in sys.modules:
            return
        PatchLIGHTgbmModelIO.__patched = True
        # noinspection PyBroadException
        try:
            import lightgbm as lgb  # noqa

            lgb.Booster.save_model = _patched_call(lgb.Booster.save_model, PatchLIGHTgbmModelIO._save)
            lgb.train = _patched_call(lgb.train, PatchLIGHTgbmModelIO._train)
            lgb.Booster = _patched_call(lgb.Booster, PatchLIGHTgbmModelIO._load)
        except ImportError:
            pass
        except Exception:
            pass

    @staticmethod
    def _save(original_fn, obj, f, *args, **kwargs):
        ret = original_fn(obj, f, *args, **kwargs)
        if not PatchLIGHTgbmModelIO._current_task:
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
        WeightsFileHandler.create_output_model(obj, filename, Framework.lightgbm, PatchLIGHTgbmModelIO._current_task,
                                               singlefile=True, model_name=model_name)
        return ret

    @staticmethod
    def _load(original_fn, model_file=None, *args, **kwargs):
        if not PatchLIGHTgbmModelIO._current_task:
            return original_fn(model_file, *args, **kwargs)

        if isinstance(model_file, six.string_types):
            filename = model_file
        elif hasattr(model_file, 'name'):
            filename = model_file.name
        elif len(args) == 1 and isinstance(args[0], six.string_types):
            filename = args[0]
        else:
            filename = None

        # register input model
        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            filename = WeightsFileHandler.restore_weights_file(empty, filename, Framework.xgboost,
                                                               PatchLIGHTgbmModelIO._current_task)
            model = original_fn(model_file=filename or model_file, *args, **kwargs)
        else:
            # try to load model before registering, in case we fail
            model = original_fn(model_file=model_file, *args, **kwargs)
            WeightsFileHandler.restore_weights_file(empty, filename, Framework.lightgbm,
                                                    PatchLIGHTgbmModelIO._current_task)

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass
        return model

    @staticmethod
    def _train(original_fn, *args, **kwargs):
        def trains_lightgbm_callback():
            def callback(env):
                # logging the results to scalars section
                # noinspection PyBroadException
                try:
                    logger = PatchLIGHTgbmModelIO._current_task.get_logger()
                    iteration = env.iteration
                    for data_title, data_series, value, _ in env.evaluation_result_list:
                        logger.report_scalar(title=data_title, series=data_series, value="{:.6f}".format(value),
                                             iteration=iteration)
                except Exception:
                    pass
            return callback

        kwargs.setdefault("callbacks", []).append(trains_lightgbm_callback())
        ret = original_fn(*args, **kwargs)
        if not PatchLIGHTgbmModelIO._current_task:
            return ret
        params = args[0] if args else kwargs.get('params', {})
        for k, v in params.items():
            if isinstance(v, set):
                params[k] = list(v)
        if params:
            PatchLIGHTgbmModelIO._current_task.connect(params)
        return ret
