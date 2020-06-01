import os
import shutil
import sys
import threading
import weakref
from random import randint
from tempfile import mkstemp
from typing import TYPE_CHECKING, Callable, Union

import six
from pathlib2 import Path

from ...backend_interface.model import Model
from ...config import running_remotely
from ...debugging.log import get_logger
from ...model import InputModel, OutputModel

if TYPE_CHECKING:
    from ...task import Task

TrainsFrameworkAdapter = 'frameworks'
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
    _model_pre_callbacks = {}
    _model_post_callbacks = {}

    @staticmethod
    def add_pre_callback(callback_function):
        # type: (Callable[[str, str, str, Task], str]) -> int

        # callback is Callable[[Union['load', 'save'], str, str, Task], str]
        if callback_function in WeightsFileHandler._model_pre_callbacks.values():
            return [k for k, v in WeightsFileHandler._model_pre_callbacks.items() if v == callback_function][0]

        while True:
            h = randint(0, 1 << 31)
            if h not in WeightsFileHandler._model_pre_callbacks:
                break
        WeightsFileHandler._model_pre_callbacks[h] = callback_function
        return h

    @staticmethod
    def add_post_callback(callback_function):
        # type: (Callable[[str, Model, str, str, str, Task], Model]) -> int

        # callback is Callable[[Union['load', 'save'], Model, str, str, str, Task], Model]
        if callback_function in WeightsFileHandler._model_post_callbacks.values():
            return [k for k, v in WeightsFileHandler._model_post_callbacks.items() if v == callback_function][0]

        while True:
            h = randint(0, 1 << 31)
            if h not in WeightsFileHandler._model_post_callbacks:
                break
        WeightsFileHandler._model_post_callbacks[h] = callback_function
        return h

    @staticmethod
    def remove_pre_callback(handle):
        # type: (int) -> bool
        if handle in WeightsFileHandler._model_pre_callbacks:
            WeightsFileHandler._model_pre_callbacks.pop(handle, None)
            return True
        return False

    @staticmethod
    def remove_post_callback(handle):
        # type: (int) -> bool
        if handle in WeightsFileHandler._model_post_callbacks:
            WeightsFileHandler._model_post_callbacks.pop(handle, None)
            return True
        return False

    @staticmethod
    def restore_weights_file(model, filepath, framework, task):
        if task is None:
            return filepath

        # call pre model callback functions
        for cb in WeightsFileHandler._model_pre_callbacks.values():
            filepath = cb('load', filepath, framework, task)

        if not filepath:
            get_logger(TrainsFrameworkAdapter).debug("Could not retrieve model file location, model is not logged")
            return filepath

        try:
            WeightsFileHandler._model_store_lookup_lock.acquire()

            # check if object already has InputModel
            trains_in_model, ref_model = WeightsFileHandler._model_in_store_lookup.get(
                id(model) if model is not None else None, (None, None))
            if ref_model is not None and model != ref_model():
                # old id pop it - it was probably reused because the object is dead
                WeightsFileHandler._model_in_store_lookup.pop(id(model))
                trains_in_model, ref_model = None, None

            # check if object already has InputModel
            model_name_id = getattr(model, 'name', '') if model else ''
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

            # check if we already have the model object:
            model_id, model_uri = Model._local_model_to_id_uri.get(filepath, (None, None))
            if model_id:
                # noinspection PyBroadException
                try:
                    trains_in_model = InputModel(model_id)
                except Exception:
                    model_id = None

            # if we do not, we need to import the model
            if not model_id:
                trains_in_model = InputModel.import_model(
                    weights_url=filepath,
                    config_dict=config_dict,
                    config_text=config_text,
                    name=task.name + (' ' + model_name_id) if model_name_id else '',
                    label_enumeration=task.get_labels_enumeration(),
                    framework=framework,
                    create_as_published=False,
                )

            # call post model callback functions
            for cb in WeightsFileHandler._model_post_callbacks.values():
                trains_in_model = cb('load', trains_in_model, filepath, filepath, framework, task)

            if model is not None:
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
            # Hack: disabled
            if False and running_remotely():
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
            get_logger(TrainsFrameworkAdapter).debug(str(ex))
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
            trains_out_model, ref_model = WeightsFileHandler._model_out_store_lookup.get(
                id(model) if model is not None else None, (None, None))
            # notice ref_model() is not an error/typo this is a weakref object call
            if ref_model is not None and model != ref_model():
                # old id pop it - it was probably reused because the object is dead
                WeightsFileHandler._model_out_store_lookup.pop(id(model))
                trains_out_model, ref_model = None, None

            if not saved_path:
                get_logger(TrainsFrameworkAdapter).warning(
                    "Could not retrieve model location, skipping auto model logging")
                return saved_path

            # check if we have output storage, and generate list of files to upload
            if Path(saved_path).is_dir():
                files = [str(f) for f in Path(saved_path).rglob('*') if f.is_file()]
            elif singlefile:
                files = [str(Path(saved_path).absolute())]
            else:
                files = [str(f) for f in Path(saved_path).parent.glob(str(Path(saved_path).name) + '.*')]

            target_filename = None
            if len(files) > 1:
                # noinspection PyBroadException
                try:
                    target_filename = Path(saved_path).stem
                except Exception:
                    pass
            else:
                target_filename = Path(files[0]).name

            # call pre model callback functions
            for cb in WeightsFileHandler._model_pre_callbacks.values():
                target_filename = cb('save', target_filename, framework, task)

            # check if object already has InputModel
            if trains_out_model is None:
                in_model_id, model_uri = Model._local_model_to_id_uri.get(saved_path, (None, None))

                if not in_model_id:
                    # if we are overwriting a local file, try to load registered model
                    # if there is an output_uri, then by definition we will not overwrite previously stored models.
                    if not task.output_uri:
                        try:
                            in_model_id = InputModel.load_model(weights_url=saved_path)
                            if in_model_id:
                                in_model_id = in_model_id.id

                                get_logger(TrainsFrameworkAdapter).info(
                                    "Found existing registered model id={} [{}] reusing it.".format(
                                        in_model_id, saved_path))
                        except:
                            in_model_id = None
                    else:
                        in_model_id = None

                trains_out_model = OutputModel(
                    task=task,
                    # config_dict=config,
                    name=(task.name + ' - ' + model_name) if model_name else None,
                    label_enumeration=task.get_labels_enumeration(),
                    framework=framework,
                    base_model_id=in_model_id
                )
                if model is not None:
                    # noinspection PyBroadException
                    try:
                        ref_model = weakref.ref(model)
                    except Exception:
                        ref_model = None
                    WeightsFileHandler._model_out_store_lookup[id(model)] = (trains_out_model, ref_model)

            # call post model callback functions
            for cb in WeightsFileHandler._model_post_callbacks.values():
                trains_out_model = cb('save', trains_out_model, target_filename, saved_path, framework, task)

            # upload files if we found them, or just register the original path
            if trains_out_model.upload_storage_uri:
                if len(files) > 1:
                    trains_out_model.update_weights_package(
                        weights_filenames=files, auto_delete_file=False, target_filename=target_filename)
                else:
                    # create a copy of the stored file,
                    # protect against someone deletes/renames the file before async upload finish is done

                    # HACK: if pytorch-lightning is used, remove the temp '.part' file extension
                    if sys.modules.get('pytorch_lightning') and target_filename.lower().endswith('.part'):
                        target_filename = target_filename[:-len('.part')]
                    fd, temp_file = mkstemp(prefix='.trains.upload_model_', suffix='.tmp')
                    os.close(fd)
                    shutil.copy(files[0], temp_file)
                    trains_out_model.update_weights(
                        weights_filename=temp_file, auto_delete_file=True, target_filename=target_filename)
            else:
                trains_out_model.update_weights(weights_filename=None, register_uri=saved_path)
        except Exception as ex:
            get_logger(TrainsFrameworkAdapter).debug(str(ex))
        finally:
            WeightsFileHandler._model_store_lookup_lock.release()

        return saved_path
