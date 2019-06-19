import threading
import weakref
from logging import getLogger

import six
from pathlib2 import Path

from ...config import running_remotely
from ...model import InputModel, OutputModel

TrainsFrameworkAdapter = 'TrainsFrameworkAdapter'
_recursion_guard = {}


def _patched_call(original_fn, patched_fn):
    def _inner_patch(*args, **kwargs):
        ident = threading._get_ident() if six.PY2 else threading.get_ident()
        if ident in _recursion_guard:
            return original_fn(*args, **kwargs)
        _recursion_guard[ident] = 1
        ret = None
        try:
            ret = patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex
        finally:
            try:
                _recursion_guard.pop(ident)
            except KeyError:
                pass
        return ret

    return _inner_patch


class _Empty(object):
    def __init__(self):
        self.trains_in_model = None


class WeightsFileHandler(object):
    _model_out_store_lookup = {}
    _model_in_store_lookup = {}
    _model_store_lookup_lock = threading.Lock()

    @staticmethod
    def restore_weights_file(model, filepath, framework, task):
        if task is None:
            return filepath

        if not filepath:
            getLogger(TrainsFrameworkAdapter).warning("Could retrieve model location, model not restored")
            return filepath

        try:
            WeightsFileHandler._model_store_lookup_lock.acquire()

            # check if object already has InputModel
            trains_in_model, ref_model = WeightsFileHandler._model_in_store_lookup.get(id(model), (None, None))
            if ref_model is not None and model != ref_model():
                # old id pop it - it was probably reused because the object is dead
                WeightsFileHandler._model_in_store_lookup.pop(id(model))
                trains_in_model, ref_model = None, None

            # check if object already has InputModel
            model_name_id = getattr(model, 'name', '')
            # noinspection PyBroadException
            try:
                config_text = None
                config_dict = trains_in_model.config_dict if trains_in_model else None
            except Exception:
                config_dict = None
                # noinspection PyBroadException
                try:
                    config_text = trains_in_model.config_text if trains_in_model else None
                except Exception:
                    config_text = None
            trains_in_model = InputModel.import_model(
                weights_url=filepath,
                config_dict=config_dict,
                config_text=config_text,
                name=task.name + ' ' + model_name_id,
                label_enumeration=task.get_labels_enumeration(),
                framework=framework,
                create_as_published=False,
            )
            # noinspection PyBroadException
            try:
                ref_model = weakref.ref(model)
            except Exception:
                ref_model = None
            WeightsFileHandler._model_in_store_lookup[id(model)] = (trains_in_model, ref_model)
            # todo: support multiple models for the same task
            task.connect(trains_in_model)
            # if we are running remotely we should deserialize the object
            # because someone might have changed the config_dict
            if running_remotely():
                # reload the model
                model_config = trains_in_model.config_dict
                # verify that this is the same model so we are not deserializing a diff model
                if (config_dict and config_dict.get('config') and model_config and model_config.get('config') and
                    config_dict.get('config').get('name') == model_config.get('config').get('name')) or \
                        (not config_dict and not model_config):
                    filepath = trains_in_model.get_weights()
                    # update filepath to point to downloaded weights file
                    # actual model weights loading will be done outside the try/exception block
        except Exception as ex:
            getLogger(TrainsFrameworkAdapter).warning(str(ex))
        finally:
            WeightsFileHandler._model_store_lookup_lock.release()

        return filepath

    @staticmethod
    def create_output_model(model, saved_path, framework, task, singlefile=False, model_name=None):
        if task is None:
            return saved_path

        try:
            WeightsFileHandler._model_store_lookup_lock.acquire()

            # check if object already has InputModel
            trains_out_model, ref_model = WeightsFileHandler._model_out_store_lookup.get(id(model), (None, None))
            if ref_model is not None and model != ref_model():
                # old id pop it - it was probably reused because the object is dead
                WeightsFileHandler._model_out_store_lookup.pop(id(model))
                trains_out_model, ref_model = None, None

            # check if object already has InputModel
            if trains_out_model is None:
                trains_out_model = OutputModel(
                    task=task,
                    # config_dict=config,
                    name=(task.name + ' - ' + model_name) if model_name else None,
                    label_enumeration=task.get_labels_enumeration(),
                    framework=framework, )
                # noinspection PyBroadException
                try:
                    ref_model = weakref.ref(model)
                except Exception:
                    ref_model = None
                WeightsFileHandler._model_out_store_lookup[id(model)] = (trains_out_model, ref_model)

            if not saved_path:
                getLogger(TrainsFrameworkAdapter).warning("Could retrieve model location, stored as unknown ")
                return saved_path

            # check if we have output storage, and generate list of files to upload
            if trains_out_model.upload_storage_uri:
                if Path(saved_path).is_dir():
                    files = [str(f) for f in Path(saved_path).rglob('*') if f.is_file()]
                elif singlefile:
                    files = [str(Path(saved_path).absolute())]
                else:
                    files = [str(f) for f in Path(saved_path).parent.glob(str(Path(saved_path).name) + '.*')]
            else:
                files = None

            # upload files if we found them, or just register the original path
            if files:
                if len(files) > 1:
                    # noinspection PyBroadException
                    try:
                        target_filename = Path(saved_path).stem
                    except Exception:
                        target_filename = None
                    trains_out_model.update_weights_package(weights_filenames=files, auto_delete_file=False,
                                                            target_filename=target_filename)
                else:
                    trains_out_model.update_weights(weights_filename=files[0], auto_delete_file=False)
            else:
                trains_out_model.update_weights(weights_filename=None, register_uri=saved_path)
        except Exception as ex:
            getLogger(TrainsFrameworkAdapter).warning(str(ex))
        finally:
            WeightsFileHandler._model_store_lookup_lock.release()

        return saved_path
