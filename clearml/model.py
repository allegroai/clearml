import abc
import os
import tarfile
import zipfile
from tempfile import mkdtemp, mkstemp

import six
from typing import List, Dict, Union, Optional, Mapping, TYPE_CHECKING, Sequence, Any

from .backend_api import Session
from .backend_api.services import models, projects
from pathlib2 import Path

from .utilities.config import config_dict_to_text, text_to_config_dict
from .utilities.proxy_object import cast_basic_type

from .backend_interface.util import (
    validate_dict, get_single_result, mutually_exclusive, exact_match_regex,
    get_or_create_project, )
from .debugging.log import get_logger
from .storage.cache import CacheManager
from .storage.helper import StorageHelper
from .storage.util import get_common_path
from .utilities.enum import Options
from .backend_interface import Task as _Task
from .backend_interface.model import create_dummy_model, Model as _Model
from .backend_interface.session import SendError
from .config import running_remotely, get_cache_dir


if TYPE_CHECKING:
    from .task import Task


class Framework(Options):
    """
    Optional frameworks for output model
    """
    tensorflow = 'TensorFlow'
    tensorflowjs = 'TensorFlow_js'
    tensorflowlite = 'TensorFlow_Lite'
    pytorch = 'PyTorch'
    torchscript = 'TorchScript'
    caffe = 'Caffe'
    caffe2 = 'Caffe2'
    onnx = 'ONNX'
    keras = 'Keras'
    mknet = 'MXNet'
    cntk = 'CNTK'
    torch = 'Torch'
    darknet = 'Darknet'
    paddlepaddle = 'PaddlePaddle'
    scikitlearn = 'ScikitLearn'
    xgboost = 'XGBoost'
    lightgbm = 'LightGBM'
    parquet = 'Parquet'
    megengine = 'MegEngine'
    catboost = 'CatBoost'
    tensorrt = 'TensorRT'
    openvino = 'OpenVINO'

    __file_extensions_mapping = {
        '.pb': (tensorflow, tensorflowjs, onnx, ),
        '.meta': (tensorflow, ),
        '.pbtxt': (tensorflow, onnx, ),
        '.zip': (tensorflow, ),
        '.tgz': (tensorflow, ),
        '.tar.gz': (tensorflow, ),
        'model.json': (tensorflowjs, ),
        '.tflite': (tensorflowlite, ),
        '.pth': (pytorch, ),
        '.pt': (pytorch, ),
        '.caffemodel': (caffe, ),
        '.prototxt': (caffe, ),
        'predict_net.pb': (caffe2, ),
        'predict_net.pbtxt': (caffe2, ),
        '.onnx': (onnx, ),
        '.h5': (keras, ),
        '.hdf5': (keras, ),
        '.keras': (keras, ),
        '.model': (mknet, cntk, xgboost),
        '-symbol.json': (mknet, ),
        '.cntk': (cntk, ),
        '.t7': (torch, ),
        '.cfg': (darknet, ),
        '__model__': (paddlepaddle, ),
        '.pkl': (scikitlearn, keras, xgboost, megengine),
        '.parquet': (parquet, ),
        '.cbm': (catboost, ),
        '.plan': (tensorrt,),
    }

    __parent_mapping = {
        "tensorflow": (
            tensorflow,
            tensorflowjs,
            tensorflowlite,
            keras,
        ),
        "pytorch": (pytorch,),
        "xgboost": (xgboost,),
        "lightgbm": (lightgbm,),
        "catboost": (catboost,),
        "joblib": (scikitlearn, xgboost)
    }

    @classmethod
    def get_framework_parents(cls, framework):
        if not framework:
            return []
        parents = []
        for k, v in cls.__parent_mapping.items():
            if framework in v:
                parents.append(k)
        return parents

    @classmethod
    def _get_file_ext(cls, framework, filename):
        mapping = cls.__file_extensions_mapping
        filename = filename.lower()

        def find_framework_by_ext(framework_selector):
            for ext, frameworks in mapping.items():
                if frameworks and filename.endswith(ext):
                    fw = framework_selector(frameworks)
                    if fw:
                        return fw, ext

        # If no framework, try finding first framework matching the extension, otherwise (or if no match) try matching
        # the given extension to the given framework. If no match return an empty extension
        return (
            (not framework and find_framework_by_ext(lambda frameworks_: frameworks_[0]))
            or find_framework_by_ext(lambda frameworks_: framework if framework in frameworks_ else None)
            or (framework, filename.split('.')[-1] if '.' in filename else '')
        )


@six.add_metaclass(abc.ABCMeta)
class BaseModel(object):
    # noinspection PyProtectedMember
    _archived_tag = _Task.archived_tag
    _package_tag = "package"

    @property
    def id(self):
        # type: () -> str
        """
        The Id (system UUID) of the model.

        :return: The model ID.
        """
        return self._get_model_data().id

    @property
    def name(self):
        # type: () -> str
        """
        The name of the model.

        :return: The model name.
        """
        return self._get_model_data().name

    @name.setter
    def name(self, value):
        # type: (str) -> None
        """
        Set the model name.

        :param str value: The model name.
        """
        self._get_base_model().update(name=value)

    @property
    def project(self):
        # type: () -> str
        """
        project id of the model.

        :return: project id (str).
        """
        data = self._get_model_data()
        return data.project

    @project.setter
    def project(self, value):
        # type: (str) -> None
        """
        Set the project ID of the model.

        :param value: project ID (str).

        :type value: str
        """
        self._get_base_model().update(project_id=value)

    @property
    def comment(self):
        # type: () -> str
        """
        The comment for the model. Also, use for a model description.

        :return: The model comment / description.
        """
        return self._get_model_data().comment

    @comment.setter
    def comment(self, value):
        # type: (str) -> None
        """
        Set comment for the model. Also, use for a model description.

        :param str value: The model comment/description.
        """
        self._get_base_model().update(comment=value)

    @property
    def tags(self):
        # type: () -> List[str]
        """
        A list of tags describing the model.

        :return: The list of tags.
        """
        return self._get_model_data().tags

    @tags.setter
    def tags(self, value):
        # type: (List[str]) -> None
        """
        Set the list of tags describing the model.

        :param value: The tags.

        :type value: list(str)
        """
        self._get_base_model().update(tags=value)

    @property
    def system_tags(self):
        # type: () -> List[str]
        """
        A list of system tags describing the model.

        :return: The list of tags.
        """
        data = self._get_model_data()
        return data.system_tags if Session.check_min_api_version('2.3') else data.tags

    @system_tags.setter
    def system_tags(self, value):
        # type: (List[str]) -> None
        """
        Set the list of system tags describing the model.

        :param value: The tags.

        :type value: list(str)
        """
        self._get_base_model().update(system_tags=value)

    @property
    def config_text(self):
        # type: () -> str
        """
        The configuration as a string. For example, prototxt, an ini file, or Python code to evaluate.

        :return: The configuration.
        """
        # noinspection PyProtectedMember
        return _Model._unwrap_design(self._get_model_data().design)

    @property
    def config_dict(self):
        # type: () -> dict
        """
        The configuration as a dictionary, parsed from the design text. This usually represents the model configuration.
        For example, prototxt, an ini file, or Python code to evaluate.

        :return: The configuration.
        """
        return self._text_to_config_dict(self.config_text)

    @property
    def labels(self):
        # type: () -> Dict[str, int]
        """
        The label enumeration of string (label) to integer (value) pairs.


        :return: A dictionary containing labels enumeration, where the keys are labels and the values as integers.
        """
        return self._get_model_data().labels

    @property
    def task(self):
        # type: () -> str
        """
        Return the creating task ID

        :return: The Task ID (str)
        """
        return self._task.id if self._task else self._get_base_model().task

    @property
    def url(self):
        # type: () -> str
        """
        Return the url of the model file (or archived files)

        :return: The model file URL.
        """
        return self._get_base_model().uri

    @property
    def published(self):
        # type: () -> bool
        return self._get_base_model().locked

    @property
    def framework(self):
        # type: () -> str
        return self._get_model_data().framework

    def __init__(self, task=None):
        # type: (Task) -> None
        super(BaseModel, self).__init__()
        self._log = get_logger()
        self._task = None
        self._set_task(task)

    def get_weights(self, raise_on_error=False):
        # type: (bool) -> str
        """
        Download the base model and return the locally stored filename.

        :param bool raise_on_error: If True and the artifact could not be downloaded,
            raise ValueError, otherwise return None on failure and output log warning.

        :return: The locally stored file.
        """
        # download model (synchronously) and return local file
        return self._get_base_model().download_model_weights(raise_on_error=raise_on_error)

    def get_weights_package(self, return_path=False, raise_on_error=False):
        # type: (bool, bool) -> Optional[Union[str, List[Path]]]
        """
        Download the base model package into a temporary directory (extract the files), or return a list of the
        locally stored filenames.

        :param bool return_path: Return the model weights or a list of filenames (Optional)

            - ``True`` - Download the model weights into a temporary directory, and return the temporary directory path.
            - ``False`` - Return a list of the locally stored filenames. (Default)

        :param bool raise_on_error: If True and the artifact could not be downloaded,
            raise ValueError, otherwise return None on failure and output log warning.

        :return: The model weights, or a list of the locally stored filenames.
            if raise_on_error=False, returns None on error.
        """
        # check if model was packaged
        if not self._is_package():
            raise ValueError('Model is not packaged')

        # download packaged model
        packed_file = self.get_weights(raise_on_error=raise_on_error)

        if not packed_file:
            if raise_on_error:
                raise ValueError('Model package \'{}\' could not be downloaded'.format(self.url))
            return None

        # unpack
        target_folder = mkdtemp(prefix='model_package_')
        if not target_folder:
            raise ValueError('cannot create temporary directory for packed weight files')

        for func in (zipfile.ZipFile, tarfile.open):
            try:
                obj = func(packed_file)
                obj.extractall(path=target_folder)
                break
            except (zipfile.BadZipfile, tarfile.ReadError):
                pass
        else:
            raise ValueError('cannot extract files from packaged model at %s', packed_file)

        if return_path:
            return target_folder

        target_files = list(Path(target_folder).glob('*'))
        return target_files

    def publish(self):
        # type: () -> ()
        """
        Set the model to the status ``published`` and for public use. If the model's status is already ``published``,
        then this method is a no-op.
        """

        if not self.published:
            self._get_base_model().publish()

    def _running_remotely(self):
        # type: () -> ()
        return bool(running_remotely() and self._task is not None)

    def _set_task(self, value):
        # type: (_Task) -> ()
        if value is not None and not isinstance(value, _Task):
            raise ValueError('task argument must be of Task type')
        self._task = value

    @abc.abstractmethod
    def _get_model_data(self):
        pass

    @abc.abstractmethod
    def _get_base_model(self):
        pass

    def _set_package_tag(self):
        if self._package_tag not in self.system_tags:
            self.system_tags.append(self._package_tag)
            self._get_base_model().edit(system_tags=self.system_tags)

    def _is_package(self):
        return self._package_tag in (self.system_tags or [])

    @staticmethod
    def _config_dict_to_text(config):
        if not isinstance(config, six.string_types) and not isinstance(config, dict):
            raise ValueError("Model configuration only supports dictionary or string objects")
        return config_dict_to_text(config)

    @staticmethod
    def _text_to_config_dict(text):
        if not isinstance(text, six.string_types):
            raise ValueError("Model configuration parsing only supports string")
        return text_to_config_dict(text)

    @staticmethod
    def _resolve_config(config_text=None, config_dict=None):
        mutually_exclusive(config_text=config_text, config_dict=config_dict, _require_at_least_one=False)
        if config_dict:
            return InputModel._config_dict_to_text(config_dict)

        return config_text

    def set_metadata(self, key, value, v_type=None):
        # type: (str, str, Optional[str]) -> bool
        """
        Set one metadata entry. All parameters must be strings or castable to strings

        :param key: Key of the metadata entry
        :param value: Value of the metadata entry
        :param v_type: Type of the metadata entry

        :return: True if the metadata was set and False otherwise
        """
        self._reload_required = (
            _Model._get_default_session()
            .send(
                models.AddOrUpdateMetadataRequest(
                    metadata=[{
                        "key": str(key),
                        "value": str(value),
                        "type": str(v_type)
                        if str(v_type) in ('float', 'int', 'bool', 'str', 'basestring', 'list', 'tuple', 'dict') else
                        str(None)
                    }],
                    model=self.id,
                    replace_metadata=False,
                )
            )
            .ok()
        )
        return self._reload_required

    def get_metadata(self, key):
        # type: (str) -> Optional[str]
        """
        Get one metadata entry value (as a string) based on its key. See `Model.get_metadata_casted`
        if you wish to cast the value to its type (if possible)

        :param key: Key of the metadata entry you want to get

        :return: String representation of the value of the metadata entry or None if the entry was not found
        """
        self._reload_if_required()
        return self.get_all_metadata().get(str(key), {}).get("value")

    def get_metadata_casted(self, key):
        # type: (str) -> Optional[str]
        """
        Get one metadata entry based on its key, casted to its type if possible

        :param key: Key of the metadata entry you want to get

        :return: The value of the metadata entry, casted to its type (if not possible,
            the string representation will be returned) or None if the entry was not found
        """
        key = str(key)
        metadata = self.get_all_metadata()
        if key not in metadata:
            return None
        return cast_basic_type(metadata[key].get("value"), metadata[key].get("type"))

    def get_all_metadata(self):
        # type: () -> Dict[str, Dict[str, str]]
        """
        See `Model.get_all_metadata_casted` if you wish to cast the value to its type (if possible)

        :return: Get all metadata as a dictionary of format Dict[key, Dict[value, type]]. The key, value and type
            entries are all strings. Note that each entry might have an additional 'key' entry, repeating the key
        """
        self._reload_if_required()
        return self._get_model_data().metadata or {}

    def get_all_metadata_casted(self):
        # type: () -> Dict[str, Dict[str, Any]]
        """
        :return: Get all metadata as a dictionary of format Dict[key, Dict[value, type]]. The key and type
            entries are strings. The value is cast to its type if possible. Note that each entry might
            have an additional 'key' entry, repeating the key
        """
        self._reload_if_required()
        result = {}
        metadata = self.get_all_metadata()
        for key, metadata_entry in metadata.items():
            result[key] = cast_basic_type(metadata_entry.get("value"), metadata_entry.get("type"))
        return result

    def set_all_metadata(self, metadata, replace=True):
        # type: (Dict[str, Dict[str, str]], bool) -> bool
        """
        Set metadata based on the given parameters. Allows replacing all entries or updating the current entries.

        :param metadata: A dictionary of format Dict[key, Dict[value, type]] representing the metadata you want to set
        :param replace: If True, replace all metadata with the entries in the `metadata` parameter. If False,
            keep the old metadata and update it with the entries in the `metadata` parameter (add or change it)

        :return: True if the metadata was set and False otherwise
        """
        metadata_array = [
            {"key": str(k), "value": str(v_t.get("value")), "type": str(v_t.get("type"))} for k, v_t in metadata.items()
        ]
        self._reload_required = (
            _Model._get_default_session()
            .send(models.AddOrUpdateMetadataRequest(metadata=metadata_array, model=self.id, replace_metadata=replace))
            .ok()
        )
        return self._reload_required

    def _reload_if_required(self):
        if not self._reload_required:
            return
        self._get_base_model().reload()
        self._reload_required = False


class Model(BaseModel):
    """
    Represent an existing model in the system, search by model id.
    The Model will be read-only and can be used to pre initialize a network
    """

    def __init__(self, model_id):
        # type: (str) ->None
        """
        Load model based on id, returned object is read-only and can be connected to a task

        Notice, we can override the input model when running remotely

        :param model_id: id (string)
        """
        super(Model, self).__init__()
        self._base_model_id = model_id
        self._base_model = None

    def get_local_copy(self, extract_archive=True, raise_on_error=False):
        # type: (bool, bool) -> str
        """
        Retrieve a valid link to the model file(s).
        If the model URL is a file system link, it will be returned directly.
        If the model URL is points to a remote location (http/s3/gs etc.),
        it will download the file(s) and return the temporary location of the downloaded model.

        :param bool extract_archive: If True and the model is of type 'packaged' (e.g. TensorFlow compressed folder)
            The returned path will be a temporary folder containing the archive content
        :param bool raise_on_error: If True and the artifact could not be downloaded,
            raise ValueError, otherwise return None on failure and output log warning.

        :return: A local path to the model (or a downloaded copy of it).
        """
        if extract_archive and self._is_package():
            return self.get_weights_package(return_path=True, raise_on_error=raise_on_error)
        return self.get_weights(raise_on_error=raise_on_error)

    def _get_base_model(self):
        if self._base_model:
            return self._base_model

        if not self._base_model_id:
            # this shouldn't actually happen
            raise Exception('Missing model ID, cannot create an empty model')
        self._base_model = _Model(
            upload_storage_uri=None,
            cache_dir=get_cache_dir(),
            model_id=self._base_model_id,
        )
        return self._base_model

    def _get_model_data(self):
        return self._get_base_model().data

    @classmethod
    def query_models(
            cls,
            project_name=None,  # type: Optional[str]
            model_name=None,  # type: Optional[str]
            tags=None,  # type: Optional[Sequence[str]]
            only_published=False,  # type: bool
            include_archived=False,  # type: bool
            max_results=None,  # type: Optional[int]
            metadata=None  # type: Optional[Dict[str, str]]
    ):
        # type: (...) -> List[Model]
        """
        Return Model objects from the project artifactory.
        Filter based on project-name / model-name / tags.
        List is always returned sorted by descending last update time (i.e. latest model is the first in the list)

        :param project_name: Optional, filter based project name string, if not given query models from all projects
        :param model_name: Optional Model name as shown in the model artifactory
        :param tags: Optional filter models based on list of tags, example: ['production', 'verified', '-qa']
            Notice use '-' prefix to filter out tags.
        :param only_published: If True only return published models.
        :param include_archived: If True return archived models.
        :param max_results: Optional return the last X models,
            sorted by last update time (from the most recent to the least).
        :param metadata: Filter based on metadata. This parameter is a dictionary. Notice that the type of the
            metadata field is not required.

        :return: ModeList of Models objects
        """
        if project_name:
            # noinspection PyProtectedMember
            res = _Model._get_default_session().send(
                projects.GetAllRequest(
                    name=exact_match_regex(project_name),
                    only_fields=['id', 'name', 'last_update']
                )
            )
            project = get_single_result(entity='project', query=project_name, results=res.response.projects)
        else:
            project = None

        only_fields = ['id', 'created', 'system_tags']

        extra_fields = {"metadata.{}.value".format(k): v for k, v in (metadata or {}).items()}

        models_fetched = []

        page = 0
        page_size = 500
        results_left = max_results if max_results is not None else float("inf")
        while True:
            # noinspection PyProtectedMember
            res = _Model._get_default_session().send(
                models.GetAllRequest(
                    project=[project.id] if project else None,
                    name=exact_match_regex(model_name) if model_name is not None else None,
                    only_fields=only_fields,
                    tags=tags or None,
                    system_tags=["-" + cls._archived_tag] if not include_archived else None,
                    ready=True if only_published else None,
                    order_by=['-created'],
                    page=page,
                    page_size=page_size if results_left > page_size else results_left,
                    _allow_extra_fields_=True,
                    **extra_fields
                )
            )
            if not res.response.models:
                break
            models_fetched.extend(res.response.models)
            results_left -= len(res.response.models)
            if results_left <= 0 or len(res.response.models) < page_size:
                break

            page += 1

        return [Model(model_id=m.id) for m in models_fetched]

    @property
    def id(self):
        # type: () -> str
        return self._base_model_id if self._base_model_id else super(Model, self).id

    @classmethod
    def remove(cls, model, delete_weights_file=True, force=False, raise_on_errors=False):
        # type: (Union[str, Model], bool, bool, bool) -> bool
        """
        Remove a model from the model repository.
        Optional, delete the model weights file from the remote storage.

        :param model: Model ID or Model object to remove
        :param delete_weights_file: If True (default) delete the weights file from the remote storage
        :param force: If True, remove model even if other Tasks are using this model. default False.
        :param raise_on_errors: If True, throw ValueError if something went wrong, default False.
        :return: True if Model was removed successfully
            partial removal returns False, i.e. Model was deleted but weights file deletion failed
        """
        if isinstance(model, str):
            model = Model(model_id=model)

        # noinspection PyBroadException
        try:
            weights_url = model.url
        except Exception:
            if raise_on_errors:
                raise ValueError("Could not find model id={}".format(model.id))
            return False

        try:
            # noinspection PyProtectedMember
            res = _Model._get_default_session().send(
                models.DeleteRequest(model.id, force=force),
            )
            response = res.wait()
            if not response.ok():
                if raise_on_errors:
                    raise ValueError("Could not remove model id={}: {}".format(model.id, response.meta))
                return False
        except SendError as ex:
            if raise_on_errors:
                raise ValueError("Could not remove model id={}: {}".format(model.id, ex))
            return False
        except ValueError:
            if raise_on_errors:
                raise
            return False
        except Exception as ex:
            if raise_on_errors:
                raise ValueError("Could not remove model id={}: {}".format(model.id, ex))
            return False

        if not delete_weights_file:
            return True

        helper = StorageHelper.get(url=weights_url)
        try:
            if not helper.delete(weights_url):
                if raise_on_errors:
                    raise ValueError("Could not remove model id={} weights file: {}".format(model.id, weights_url))
                return False
        except Exception as ex:
            if raise_on_errors:
                raise ValueError("Could not remove model id={} weights file \'{}\': {}".format(
                    model.id, weights_url, ex))
            return False

        return True


class InputModel(Model):
    """
    Load an existing model in the system, search by model id.
    The Model will be read-only and can be used to pre initialize a network
    We can connect the model to a task as input model, then when running remotely override it with the UI.
    """

    # noinspection PyProtectedMember
    _EMPTY_MODEL_ID = _Model._EMPTY_MODEL_ID

    @classmethod
    def import_model(
        cls,
        weights_url,  # type: str
        config_text=None,  # type: Optional[str]
        config_dict=None,  # type: Optional[dict]
        label_enumeration=None,  # type: Optional[Mapping[str, int]]
        name=None,  # type: Optional[str]
        project=None,  # type: Optional[str]
        tags=None,  # type: Optional[List[str]]
        comment=None,  # type: Optional[str]
        is_package=False,  # type: bool
        create_as_published=False,  # type: bool
        framework=None,  # type: Optional[str]
    ):
        # type: (...) -> InputModel
        """
        Create an InputModel object from a pre-trained model by specifying the URL of an initial weight files.
        Optionally, input a configuration, label enumeration, name for the model, tags describing the model,
        comment as a description of the model, indicate whether the model is a package, specify the model's
        framework, and indicate whether to immediately set the model's status to ``Published``.
        The model is read-only.

        The **ClearML Server** (backend) may already store the model's URL. If the input model's URL is not
        stored, meaning the model is new, then it is imported and ClearML stores its metadata.
        If the URL is already stored, the import process stops, ClearML issues a warning message, and ClearML
        reuses the model.

        In your Python experiment script, after importing the model, you can connect it to the main execution
        Task as an input model using :meth:`InputModel.connect` or :meth:`.Task.connect`. That initializes the
        network.

        .. note::
           Using the **ClearML Web-App** (user interface), you can reuse imported models and switch models in
           experiments.

        :param str weights_url: A valid URL for the initial weights file. If the **ClearML Web-App** (backend)
            already stores the metadata of a model with the same URL, that existing model is returned
            and ClearML ignores all other parameters.

            For example:

            - ``https://domain.com/file.bin``
            - ``s3://bucket/file.bin``
            - ``file:///home/user/file.bin``

        :param str config_text: The configuration as a string. This is usually the content of a configuration
            dictionary file. Specify ``config_text`` or ``config_dict``, but not both.
        :type config_text: unconstrained text string
        :param dict config_dict: The configuration as a dictionary. Specify ``config_text`` or ``config_dict``,
            but not both.
        :param dict label_enumeration: Optional label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }
        :param str name: The name of the newly imported model. (Optional)
        :param str project: The project name to add the model into. (Optional)
        :param tags: The list of tags which describe the model. (Optional)
        :type tags: list(str)
        :param str comment: A comment / description for the model. (Optional)
        :type comment: str
        :param is_package: Is the imported weights file is a package (Optional)

            - ``True`` - Is a package. Add a package tag to the model.
            - ``False`` - Is not a package. Do not add a package tag. (Default)

        :type is_package: bool
        :param bool create_as_published: Set the model's status to Published (Optional)

            - ``True`` - Set the status to Published.
            - ``False`` - Do not set the status to Published. The status will be Draft. (Default)

        :param str framework: The framework of the model. (Optional)
        :type framework: str or Framework object

        :return: The imported model or existing model (see above).
        """
        config_text = cls._resolve_config(config_text=config_text, config_dict=config_dict)
        weights_url = StorageHelper.conform_url(weights_url)
        if not weights_url:
            raise ValueError("Please provide a valid weights_url parameter")
        # convert local to file to remote one
        weights_url = CacheManager.get_remote_url(weights_url)

        extra = {'system_tags': ["-" + cls._archived_tag]} \
            if Session.check_min_api_version('2.3') else {'tags': ["-" + cls._archived_tag]}
        # noinspection PyProtectedMember
        result = _Model._get_default_session().send(models.GetAllRequest(
            uri=[weights_url],
            only_fields=["id", "name", "created"],
            **extra
        ))

        if result.response.models:
            logger = get_logger()

            logger.debug('A model with uri "{}" already exists. Selecting it'.format(weights_url))

            model = get_single_result(
                entity='model',
                query=weights_url,
                results=result.response.models,
                log=logger,
                raise_on_error=False,
            )

            logger.info("Selected model id: {}".format(model.id))

            return InputModel(model_id=model.id)

        base_model = _Model(
            upload_storage_uri=None,
            cache_dir=get_cache_dir(),
        )

        from .task import Task
        task = Task.current_task()
        if task:
            comment = 'Imported by task id: {}'.format(task.id) + ('\n' + comment if comment else '')
            project_id = task.project
            name = name or 'Imported by {}'.format(task.name or '')
            # do not register the Task, because we do not want it listed after as "output model",
            # the Task never actually created the Model
            task_id = None
        else:
            project_id = None
            task_id = None

        if project:
            project_id = get_or_create_project(
                session=task.session if task else Task._get_default_session(),
                project_name=project
            )

        if not framework:
            # noinspection PyProtectedMember
            framework, file_ext = Framework._get_file_ext(
                framework=framework,
                filename=weights_url
            )

        base_model.update(
            design=config_text,
            labels=label_enumeration,
            name=name,
            comment=comment,
            tags=tags,
            uri=weights_url,
            framework=framework,
            project_id=project_id,
            task_id=task_id,
        )

        this_model = InputModel(model_id=base_model.id)
        this_model._base_model = base_model

        if is_package:
            this_model._set_package_tag()

        if create_as_published:
            this_model.publish()

        return this_model

    @classmethod
    def load_model(cls, weights_url, load_archived=False):
        # type: (str, bool) -> InputModel
        """
        Load an already registered model based on a pre-existing model file (link must be valid). If the url to the
        weights file already exists, the returned object is a Model representing the loaded Model. If no registered
        model with the specified url is found, ``None`` is returned.

        :param weights_url: The valid url for the weights file (string).

            Examples:

            .. code-block:: py

                "https://domain.com/file.bin" or "s3://bucket/file.bin" or "file:///home/user/file.bin".

            .. note::
                If a model with the exact same URL exists, it will be used, and all other arguments will be ignored.

        :param bool load_archived: Load archived models

            - ``True`` - Load the registered Model, if it is archived.
            - ``False`` - Ignore archive models.

        :return: The InputModel object, or None if no model could be found.
        """
        weights_url = StorageHelper.conform_url(weights_url)
        if not weights_url:
            raise ValueError("Please provide a valid weights_url parameter")

        # convert local to file to remote one
        weights_url = CacheManager.get_remote_url(weights_url)

        if not load_archived:
            # noinspection PyTypeChecker
            extra = {'system_tags': ["-" + _Task.archived_tag]} \
                if Session.check_min_api_version('2.3') else {'tags': ["-" + cls._archived_tag]}
        else:
            extra = {}

        # noinspection PyProtectedMember
        result = _Model._get_default_session().send(models.GetAllRequest(
            uri=[weights_url],
            only_fields=["id", "name", "created"],
            **extra
        ))

        if not result or not result.response or not result.response.models:
            return None

        logger = get_logger()
        model = get_single_result(
            entity='model',
            query=weights_url,
            results=result.response.models,
            log=logger,
            raise_on_error=False,
        )

        return InputModel(model_id=model.id)

    @classmethod
    def empty(cls, config_text=None, config_dict=None, label_enumeration=None):
        # type: (Optional[str], Optional[dict], Optional[Mapping[str, int]]) -> InputModel
        """
        Create an empty model object. Later, you can assign a model to the empty model object.

        :param config_text: The model configuration as a string. This is usually the content of a configuration
            dictionary file. Specify ``config_text`` or ``config_dict``, but not both.
        :type config_text: unconstrained text string
        :param dict config_dict: The model configuration as a dictionary. Specify ``config_text`` or ``config_dict``,
            but not both.
        :param dict label_enumeration: The label enumeration dictionary of string (label) to integer (value) pairs.
            (Optional)

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        :return: An empty model object.
        """
        design = cls._resolve_config(config_text=config_text, config_dict=config_dict)

        this_model = InputModel(model_id=cls._EMPTY_MODEL_ID)
        this_model._base_model = m = _Model(
            cache_dir=None,
            upload_storage_uri=None,
            model_id=cls._EMPTY_MODEL_ID,
        )
        # noinspection PyProtectedMember
        m._data.design = _Model._wrap_design(design)
        # noinspection PyProtectedMember
        m._data.labels = label_enumeration
        return this_model

    def __init__(self, model_id=None, name=None, project=None, tags=None, only_published=False):
        # type: (Optional[str], Optional[str], Optional[str], Optional[Sequence[str]], bool) -> None
        """
        Load a model from the Model artifactory,
        based on model_id (uuid) or a model name/projects/tags combination.

        :param model_id: The ClearML Id (system UUID) of the input model whose metadata the **ClearML Server**
            (backend) stores. If provided all other arguments are ignored
        :param name: Model name to search and load
        :param project: Model project name to search model in
        :param tags: Model tags list to filter by
        :param only_published: If True filter out non-published (draft) models
        """
        if not model_id:
            found_models = self.query_models(
                project_name=project, model_name=name, tags=tags, only_published=only_published)
            if not found_models:
                raise ValueError("Could not locate model with project={} name={} tags={} published={}".format(
                    project, name, tags, only_published))
            model_id = found_models[0].id
        super(InputModel, self).__init__(model_id)

    @property
    def id(self):
        # type: () -> str
        return self._base_model_id

    def connect(self, task, name=None):
        # type: (Task, Optional[str]) -> None
        """
        Connect the current model to a Task object, if the model is preexisting. Preexisting models include:

        - Imported models (InputModel objects created using the :meth:`Logger.import_model` method).
        - Models whose metadata is already in the ClearML platform, meaning the InputModel object is instantiated
          from the ``InputModel`` class specifying the model's ClearML Id as an argument.
        - Models whose origin is not ClearML that are used to create an InputModel object. For example,
          models created using TensorFlow models.

        When the experiment is executed remotely in a worker, the input model already specified in the experiment is
        used.

        .. note::
           The **ClearML Web-App** allows you to switch one input model for another and then enqueue the experiment
           to execute in a worker.

        :param object task: A Task object.
        :param str name: The model name to be stored on the Task
            (default the filename, of the model weights, without the file extension)
        """
        self._set_task(task)

        model_id = None
        # noinspection PyProtectedMember
        if running_remotely() and (task.is_main_task() or task._is_remote_main_task()):
            input_models = task.input_models_id
            # noinspection PyBroadException
            try:
                # TODO: (temp fix) At the moment, the UI changes the key of the model hparam
                # when modifying its value... There is no way to tell which model was changed
                # so just take the first one in case `name` is not in `input_models`
                model_id = input_models.get(name, next(iter(input_models.values())))
                self._base_model_id = model_id
                self._base_model = InputModel(model_id=model_id)._get_base_model()
            except Exception:
                model_id = None

        if not model_id:
            # we should set the task input model to point to us
            model = self._get_base_model()
            # try to store the input model id, if it is not empty
            # (Empty Should not happen)
            if model.id != self._EMPTY_MODEL_ID:
                task.set_input_model(model_id=model.id, name=name)
            # only copy the model design if the task has no design to begin with
            # noinspection PyProtectedMember
            if not self._task._get_model_config_text() and model.model_design:
                # noinspection PyProtectedMember
                task._set_model_config(config_text=model.model_design)
            if not self._task.get_labels_enumeration() and model.data.labels:
                task.set_model_label_enumeration(model.data.labels)


class OutputModel(BaseModel):
    """
    Create an output model for a Task (experiment) to store the training results.

    The OutputModel object is always connected to a Task object, because it is instantiated with a Task object
    as an argument. It is, therefore, automatically registered as the Task's (experiment's) output model.

    The OutputModel object is read-write.

    A common use case is to reuse the OutputModel object, and override the weights after storing a model snapshot.
    Another use case is to create multiple OutputModel objects for a Task (experiment), and after a new high score
    is found, store a model snapshot.

    If the model configuration and / or the model's label enumeration
    are ``None``, then the output model is initialized with the values from the Task object's input model.

    .. note::
       When executing a Task (experiment) remotely in a worker, you can modify the model configuration and / or model's
       label enumeration using the **ClearML Web-App**.
    """

    _default_output_uri = None

    @property
    def published(self):
        # type: () -> bool
        """
        Get the published state of this model.

        :return:

        """
        if not self.id:
            return False
        return self._get_base_model().locked

    @property
    def config_text(self):
        # type: () -> str
        """
        Get the configuration as a string. For example, prototxt, an ini file, or Python code to evaluate.

        :return: The configuration.
        """
        # noinspection PyProtectedMember
        return _Model._unwrap_design(self._get_model_data().design)

    @config_text.setter
    def config_text(self, value):
        # type: (str) -> None
        """
        Set the configuration. Store a blob of text for custom usage.
        """
        self.update_design(config_text=value)

    @property
    def config_dict(self):
        # type: () -> dict
        """
        Get the configuration as a dictionary parsed from the ``config_text`` text. This usually represents the model
        configuration. For example, from prototxt to ini file or python code to evaluate.

        :return: The configuration.
        """
        return self._text_to_config_dict(self.config_text)

    @config_dict.setter
    def config_dict(self, value):
        # type: (dict) -> None
        """
        Set the configuration. Saved in the model object.

        :param dict value: The configuration parameters.
        """
        self.update_design(config_dict=value)

    @property
    def labels(self):
        # type: () -> Dict[str, int]
        """
        Get the label enumeration as a dictionary of string (label) to integer (value) pairs.

        For example:

        .. code-block:: javascript

           {
                "background": 0,
                "person": 1
           }

        :return: The label enumeration.
        """
        return self._get_model_data().labels

    @labels.setter
    def labels(self, value):
        # type: (Mapping[str, int]) -> None
        """
        Set the label enumeration.

        :param dict value: The label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        """
        self.update_labels(labels=value)

    @property
    def upload_storage_uri(self):
        # type: () -> str
        return self._get_base_model().upload_storage_uri

    def __init__(
        self,
        task=None,  # type: Optional[Task]
        config_text=None,  # type: Optional[str]
        config_dict=None,  # type: Optional[dict]
        label_enumeration=None,  # type: Optional[Mapping[str, int]]
        name=None,  # type: Optional[str]
        tags=None,  # type: Optional[List[str]]
        comment=None,  # type: Optional[str]
        framework=None,  # type: Optional[Union[str, Framework]]
        base_model_id=None,  # type: Optional[str]
    ):
        """
        Create a new model and immediately connect it to a task.

        We do not allow for Model creation without a task, so we always keep track on how we created the models
        In remote execution, Model parameters can be overridden by the Task
        (such as model configuration & label enumerator)

        :param task: The Task object with which the OutputModel object is associated.
        :type task: Task
        :param config_text: The configuration as a string. This is usually the content of a configuration
            dictionary file. Specify ``config_text`` or ``config_dict``, but not both.
        :type config_text: unconstrained text string
        :param dict config_dict: The configuration as a dictionary.
            Specify ``config_dict`` or ``config_text``, but not both.
        :param dict label_enumeration: The label enumeration dictionary of string (label) to integer (value) pairs.
            (Optional)

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        :param str name: The name for the newly created model. (Optional)
        :param list(str) tags: A list of strings which are tags for the model. (Optional)
        :param str comment: A comment / description for the model. (Optional)
        :param framework: The framework of the model or a Framework object. (Optional)
        :type framework: str or Framework object
        :param base_model_id: optional, model id to be reused
        """
        if not task:
            from .task import Task
            task = Task.current_task()
            if not task:
                raise ValueError("task object was not provided, and no current task was found")

        super(OutputModel, self).__init__(task=task)

        config_text = self._resolve_config(config_text=config_text, config_dict=config_dict)

        self._model_local_filename = None
        self._last_uploaded_url = None
        self._base_model = None
        self._base_model_id = None
        self._task_connect_name = None
        # noinspection PyProtectedMember
        self._floating_data = create_dummy_model(
            design=_Model._wrap_design(config_text),
            labels=label_enumeration or task.get_labels_enumeration(),
            name=name or self._task.name,
            tags=tags,
            comment='{} by task id: {}'.format('Created' if not base_model_id else 'Overwritten', task.id) +
                    ('\n' + comment if comment else ''),
            framework=framework,
            upload_storage_uri=task.output_uri,
        )
        # If we have no real model ID, we are done
        if not base_model_id:
            return

        # noinspection PyBroadException
        try:
            # noinspection PyProtectedMember
            _base_model = self._task._get_output_model(model_id=base_model_id)
            _base_model.update(
                labels=self._floating_data.labels,
                design=self._floating_data.design,
                task_id=self._task.id,
                project_id=self._task.project,
                name=self._floating_data.name or self._task.name,
                comment=('{}\n{}'.format(_base_model.comment, self._floating_data.comment)
                         if (_base_model.comment and self._floating_data.comment and
                             self._floating_data.comment not in _base_model.comment)
                         else (_base_model.comment or self._floating_data.comment)),
                tags=self._floating_data.tags,
                framework=self._floating_data.framework,
                upload_storage_uri=self._floating_data.upload_storage_uri
            )
            self._base_model = _base_model
            self._floating_data = None
            name = self._task_connect_name or Path(_base_model.uri).stem
        except Exception:
            pass
        self.connect(task, name=name)

    def connect(self, task, name=None):
        # type: (Task, Optional[str]) -> None
        """
        Connect the current model to a Task object, if the model is a preexisting model. Preexisting models include:

        - Imported models.
        - Models whose metadata the **ClearML Server** (backend) is already storing.
        - Models from another source, such as frameworks like TensorFlow.

        :param object task: A Task object.
        :param str name: The model name as it would appear on the Task object.
            The model object itself can have a different name,
            this is designed to support multiple models used/created by a single Task.
            Use examples would be GANs or model ensemble
        """
        if self._task != task:
            raise ValueError('Can only connect preexisting model to task, but this is a fresh model')

        if name:
            self._task_connect_name = name

        # we should set the task input model to point to us
        model = self._get_base_model()

        # only copy the model design if the task has no design to begin with
        # noinspection PyProtectedMember
        if not self._task._get_model_config_text():
            # noinspection PyProtectedMember
            task._set_model_config(
                config_text=model.model_design if hasattr(model, "model_design") else model.design.get("design", "")
            )
        if not self._task.get_labels_enumeration():
            task.set_model_label_enumeration(model.data.labels if hasattr(model, "data") else model.labels)

        if self._base_model:
            self._base_model.update_for_task(
                task_id=self._task.id, model_id=self.id, type_="output", name=self._task_connect_name)

    def set_upload_destination(self, uri):
        # type: (str) -> None
        """
        Set the URI of the storage destination for uploaded model weight files.
        Supported storage destinations include S3, Google Cloud Storage), and file locations.

        Using this method, files uploads are separate and then a link to each is stored in the model object.

        .. note::
           For storage requiring credentials, the credentials are stored in the ClearML configuration file,
           ``~/clearml.conf``.

        :param str uri: The URI of the upload storage destination.

            For example:

            - ``s3://bucket/directory/``
            - ``file:///tmp/debug/``

        :return bool: The status of whether the storage destination schema is supported.

            - ``True`` - The storage destination scheme is supported.
            - ``False`` - The storage destination scheme is not supported.
        """
        if not uri:
            return

        # Test if we can update the model.
        self._validate_update()

        # Create the storage helper
        storage = StorageHelper.get(uri)

        # Verify that we can upload to this destination
        try:
            uri = storage.verify_upload(folder_uri=uri)
        except Exception:
            raise ValueError("Could not set destination uri to: %s [Check write permissions]" % uri)

        # store default uri
        self._get_base_model().upload_storage_uri = uri

    def update_weights(
        self,
        weights_filename=None,  # type: Optional[str]
        upload_uri=None,  # type: Optional[str]
        target_filename=None,  # type: Optional[str]
        auto_delete_file=True,  # type: bool
        register_uri=None,  # type: Optional[str]
        iteration=None,  # type: Optional[int]
        update_comment=True,  # type: bool
        is_package=False,  # type: bool
    ):
        # type: (...) -> str
        """
        Update the model weights from a locally stored model filename.

        .. note::
           Uploading the model is a background process. A call to this method returns immediately.

        :param str weights_filename: The name of the locally stored weights file to upload.
            Specify ``weights_filename`` or ``register_uri``, but not both.
        :param str upload_uri: The URI of the storage destination for model weights upload. The default value
            is the previously used URI. (Optional)
        :param str target_filename: The newly created filename in the storage destination location. The default value
            is the ``weights_filename`` value. (Optional)
        :param bool auto_delete_file: Delete the temporary file after uploading (Optional)

            - ``True`` - Delete (Default)
            - ``False`` - Do not delete

        :param str register_uri: The URI of an already uploaded weights file. The URI must be valid. Specify
            ``register_uri`` or ``weights_filename``, but not both.
        :param int iteration: The iteration number.
        :param bool update_comment: Update the model comment with the local weights file name (to maintain
            provenance) (Optional)

            - ``True`` - Update model comment (Default)
            - ``False`` - Do not update
        :param bool is_package: Mark the weights file as compressed package, usually a zip file.

        :return: The uploaded URI.
        """

        def delete_previous_weights_file(filename=weights_filename):
            try:
                if filename:
                    os.remove(filename)
            except OSError:
                self._log.debug('Failed removing temporary file %s' % filename)

        # test if we can update the model
        if self.id and self.published:
            raise ValueError('Model is published and cannot be changed')

        if (not weights_filename and not register_uri) or (weights_filename and register_uri):
            raise ValueError('Model update must have either local weights file to upload, '
                             'or pre-uploaded register_uri, never both')

        # only upload if we are connected to a task
        if not self._task:
            raise Exception('Missing a task for this model')

        if weights_filename is not None:
            # Check if weights_filename is a folder, is package upload
            if Path(weights_filename).is_dir():
                return self.update_weights_package(
                    weights_path=weights_filename,
                    upload_uri=upload_uri,
                    target_filename=target_filename or Path(weights_filename).name,
                    auto_delete_file=auto_delete_file,
                    iteration=iteration)

            # make sure we delete the previous file, if it exists
            if self._model_local_filename != weights_filename:
                delete_previous_weights_file(self._model_local_filename)
            # store temp filename for deletion next time, if needed
            if auto_delete_file:
                self._model_local_filename = weights_filename

        # make sure the created model is updated:
        out_model_file_name = target_filename or weights_filename or register_uri
        name = Path(out_model_file_name).stem if out_model_file_name else (self._task_connect_name or "Output Model")
        model = self._get_force_base_model(task_model_entry=name)
        if not model:
            raise ValueError('Failed creating internal output model')

        # select the correct file extension based on the framework,
        # or update the framework based on the file extension
        # noinspection PyProtectedMember
        framework, file_ext = Framework._get_file_ext(
            framework=self._get_model_data().framework,
            filename=target_filename or weights_filename or register_uri
        )

        if weights_filename:
            target_filename = target_filename or Path(weights_filename).name
            if not target_filename.lower().endswith(file_ext):
                target_filename += file_ext

        # set target uri for upload (if specified)
        if upload_uri:
            self.set_upload_destination(upload_uri)

        # let us know the iteration number, we put it in the comment section for now.
        if update_comment:
            comment = self.comment or ''
            iteration_msg = 'snapshot {} stored'.format(weights_filename or register_uri)
            if not comment.startswith('\n'):
                comment = '\n' + comment
            comment = iteration_msg + comment
        else:
            comment = None

        # if we have no output destination, just register the local model file
        if weights_filename and not self.upload_storage_uri and not self._task.storage_uri:
            register_uri = weights_filename
            weights_filename = None
            auto_delete_file = False
            self._log.info('No output storage destination defined, registering local model %s' % register_uri)

        # start the upload
        if weights_filename:
            if not model.upload_storage_uri:
                self.set_upload_destination(self.upload_storage_uri or self._task.storage_uri)

            output_uri = model.update_and_upload(
                model_file=weights_filename,
                task_id=self._task.id,
                async_enable=True,
                target_filename=target_filename,
                framework=self.framework or framework,
                comment=comment,
                cb=delete_previous_weights_file if auto_delete_file else None,
                iteration=iteration or self._task.get_last_iteration(),
            )
        elif register_uri:
            register_uri = StorageHelper.conform_url(register_uri)
            output_uri = model.update(uri=register_uri, task_id=self._task.id, framework=framework, comment=comment)
        else:
            output_uri = None

        self._last_uploaded_url = output_uri

        if is_package:
            self._set_package_tag()

        return output_uri

    def update_weights_package(
        self,
        weights_filenames=None,  # type: Optional[Sequence[str]]
        weights_path=None,  # type: Optional[str]
        upload_uri=None,  # type: Optional[str]
        target_filename=None,  # type: Optional[str]
        auto_delete_file=True,  # type: bool
        iteration=None  # type: Optional[int]
    ):
        # type: (...) -> str
        """
        Update the model weights from locally stored model files, or from directory containing multiple files.

        .. note::
           Uploading the model weights is a background process. A call to this method returns immediately.

        :param weights_filenames: The file names of the locally stored model files. Specify ``weights_filenames``,
            or ``weights_path``, but not both.
        :type weights_filenames: list(str)
        :param weights_path: The directory path to a package. All the files in the directory will be uploaded.
            Specify ``weights_path`` or ``weights_filenames``, but not both.
        :type weights_path: str
        :param str upload_uri: The URI of the storage destination for the model weights upload. The default
            is the previously used URI. (Optional)
        :param str target_filename: The newly created filename in the storage destination URI location. The default
            is the value specified in the ``weights_filename`` parameter.  (Optional)
        :param bool auto_delete_file: Delete temporary file after uploading  (Optional)

            - ``True`` - Delete (Default)
            - ``False`` - Do not delete

        :param int iteration: The iteration number.

        :return: The uploaded URI for the weights package.
        """
        # create list of files
        if (not weights_filenames and not weights_path) or (weights_filenames and weights_path):
            raise ValueError('Model update weights package should get either '
                             'directory path to pack or a list of files')

        if not weights_filenames:
            weights_filenames = list(map(six.text_type, Path(weights_path).rglob('*')))
        elif weights_filenames and len(weights_filenames) > 1:
            weights_path = get_common_path(weights_filenames)

        # create packed model from all the files
        fd, zip_file = mkstemp(prefix='model_package.', suffix='.zip')
        try:
            with zipfile.ZipFile(zip_file, 'w', allowZip64=True, compression=zipfile.ZIP_STORED) as zf:
                for filename in weights_filenames:
                    relative_file_name = Path(filename).name if not weights_path else \
                        Path(filename).absolute().relative_to(Path(weights_path).absolute()).as_posix()
                    zf.write(filename, arcname=relative_file_name)
        finally:
            os.close(fd)

        # now we can delete the files (or path if provided)
        if auto_delete_file:
            def safe_remove(path, is_dir=False):
                try:
                    (os.rmdir if is_dir else os.remove)(path)
                except OSError:
                    self._log.info('Failed removing temporary {}'.format(path))

            for filename in weights_filenames:
                safe_remove(filename)
            if weights_path:
                safe_remove(weights_path, is_dir=True)

        if target_filename and not target_filename.lower().endswith('.zip'):
            target_filename += '.zip'

        # and now we should upload the file, always delete the temporary zip file
        comment = self.comment or ''
        iteration_msg = 'snapshot {} stored'.format(str(weights_filenames))
        if not comment.startswith('\n'):
            comment = '\n' + comment
        comment = iteration_msg + comment
        self.comment = comment
        uploaded_uri = self.update_weights(weights_filename=zip_file, auto_delete_file=True, upload_uri=upload_uri,
                                           target_filename=target_filename or 'model_package.zip',
                                           iteration=iteration, update_comment=False)
        # set the model tag (by now we should have a model object) so we know we have packaged file
        self._set_package_tag()
        return uploaded_uri

    def update_design(self, config_text=None, config_dict=None):
        # type: (Optional[str], Optional[dict]) -> bool
        """
        Update the model configuration. Store a blob of text for custom usage.

        .. note::
           This method's behavior is lazy. The design update is only forced when the weights
           are updated.

        :param config_text: The configuration as a string. This is usually the content of a configuration
            dictionary file. Specify ``config_text`` or ``config_dict``, but not both.
        :type config_text: unconstrained text string
        :param dict config_dict: The configuration as a dictionary. Specify ``config_text`` or ``config_dict``,
            but not both.

        :return: True, update successful. False, update not successful.
        """
        if not self._validate_update():
            return False

        config_text = self._resolve_config(config_text=config_text, config_dict=config_dict)

        if self._task and not self._task.get_model_config_text():
            self._task.set_model_config(config_text=config_text)

        if self.id:
            # update the model object (this will happen if we resumed a training task)
            result = self._get_force_base_model().edit(design=config_text)
        else:
            # noinspection PyProtectedMember
            self._floating_data.design = _Model._wrap_design(config_text)
            result = Waitable()

        # you can wait on this object
        return result

    def update_labels(self, labels):
        # type: (Mapping[str, int]) -> Optional[Waitable]
        """
        Update the label enumeration.

        :param dict labels: The label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        :return:
        """
        validate_dict(labels, key_types=six.string_types, value_types=six.integer_types, desc='label enumeration')

        if not self._validate_update():
            return

        if self._task:
            self._task.set_model_label_enumeration(labels)

        if self.id:
            # update the model object (this will happen if we resumed a training task)
            result = self._get_force_base_model().edit(labels=labels)
        else:
            self._floating_data.labels = labels
            result = Waitable()

        # you can wait on this object
        return result

    @classmethod
    def wait_for_uploads(cls, timeout=None, max_num_uploads=None):
        # type: (Optional[float], Optional[int]) -> None
        """
        Wait for any pending or in-progress model uploads to complete. If no uploads are pending or in-progress,
        then the ``wait_for_uploads`` returns immediately.

        :param float timeout: The timeout interval to wait for uploads (seconds). (Optional).
        :param int max_num_uploads: The maximum number of uploads to wait for. (Optional).
        """
        _Model.wait_for_results(timeout=timeout, max_num_uploads=max_num_uploads)

    @classmethod
    def set_default_upload_uri(cls, output_uri):
        # type: (Optional[str]) -> None
        """
        Set the default upload uri for all OutputModels

        :param output_uri: URL for uploading models. examples:
            https://demofiles.demo.clear.ml, s3://bucket/, gs://bucket/, azure://bucket/, file:///mnt/shared/nfs
        """
        cls._default_output_uri = str(output_uri) if output_uri else None

    def _get_force_base_model(self, model_name=None, task_model_entry=None):
        if self._base_model:
            return self._base_model

        # create a new model from the task
        # noinspection PyProtectedMember
        self._base_model = self._task._get_output_model(model_id=None)
        # update the model from the task inputs
        labels = self._task.get_labels_enumeration()
        # noinspection PyProtectedMember
        config_text = self._task._get_model_config_text()
        model_name = model_name or self._floating_data.name or self._task.name
        task_model_entry = task_model_entry or self._task_connect_name or Path(self._get_model_data().uri).stem
        parent = self._task.input_models_id.get(task_model_entry)
        self._base_model.update(
            labels=self._floating_data.labels or labels,
            design=self._floating_data.design or config_text,
            task_id=self._task.id,
            project_id=self._task.project,
            parent_id=parent,
            name=model_name,
            comment=self._floating_data.comment,
            tags=self._floating_data.tags,
            framework=self._floating_data.framework,
            upload_storage_uri=self._floating_data.upload_storage_uri
        )

        # remove model floating change set, by now they should have matched the task.
        self._floating_data = None

        # now we have to update the creator task so it points to us
        if str(self._task.status) not in (
                str(self._task.TaskStatusEnum.created), str(self._task.TaskStatusEnum.in_progress)):
            self._log.warning('Could not update last created model in Task {}, '
                              'Task status \'{}\' cannot be updated'.format(self._task.id, self._task.status))
        else:
            self._base_model.update_for_task(
                task_id=self._task.id, model_id=self.id, type_="output", name=task_model_entry)

        return self._base_model

    def _get_base_model(self):
        if self._floating_data:
            return self._floating_data
        return self._get_force_base_model()

    def _get_model_data(self):
        if self._base_model:
            return self._base_model.data
        return self._floating_data

    def _validate_update(self):
        # test if we can update the model
        if self.id and self.published:
            raise ValueError('Model is published and cannot be changed')

        return True

    def _get_last_uploaded_filename(self):
        if not self._last_uploaded_url and not self.url:
            return None
        return Path(self._last_uploaded_url or self.url).name


class Waitable(object):
    def wait(self, *_, **__):
        return True
