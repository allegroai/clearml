import abc
import os
import tarfile
import zipfile
from tempfile import mkdtemp, mkstemp

import pyparsing
import six
from typing import List, Dict, Union, Optional, TYPE_CHECKING, Sequence

from .backend_api import Session
from .backend_api.services import models
from pathlib2 import Path
from .utilities.pyhocon import ConfigFactory, HOCONConverter

from .backend_interface.util import validate_dict, get_single_result, mutually_exclusive
from .debugging.log import get_logger
from .storage.helper import StorageHelper
from .utilities.enum import Options
from .backend_interface import Task as _Task
from .backend_interface.model import create_dummy_model, Model as _Model
from .config import running_remotely, get_cache_dir


if TYPE_CHECKING:
    from .task import Task

ARCHIVED_TAG = "archived"


class Framework(Options):
    """
    Optional frameworks for output model
    """
    tensorflow = 'TensorFlow'
    tensorflowjs = 'TensorFlow_js'
    tensorflowlite = 'TensorFlow_Lite'
    pytorch = 'PyTorch'
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
    parquet = 'Parquet'

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
        '.pkl': (scikitlearn, keras, xgboost),
        '.parquet': (parquet),
    }

    @classmethod
    def _get_file_ext(cls, framework, filename):
        mapping = cls.__file_extensions_mapping
        filename = filename.lower()

        def find_framework_by_ext(framework_selector):
            for ext, frameworks in mapping.items():
                if frameworks and filename.endswith(ext):
                    fw = framework_selector(frameworks)
                    if fw:
                        return (fw, ext)

        # If no framework, try finding first framework matching the extension, otherwise (or if no match) try matching
        # the given extension to the given framework. If no match return an empty extension
        return (
            (not framework and find_framework_by_ext(lambda frameworks_: frameworks_[0]))
            or find_framework_by_ext(lambda frameworks_: framework if framework in frameworks_ else None)
            or (framework, filename.split('.')[-1] if '.' in filename else '')
        )


@six.add_metaclass(abc.ABCMeta)
class BaseModel(object):
    _package_tag = "package"

    @property
    def id(self):
        # type: () -> str
        """
        The Id (system UUID) of the model.

        :return str: The model id.
        """
        return self._get_model_data().id

    @property
    def name(self):
        # type: () -> str
        """
        The name of the model.

        :return str: The model name.
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
    def comment(self):
        # type: () -> str
        """
        The comment for the model. Also, use for a model description.

        :return str: The model comment / description.
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

        :return list(str): The list of tags.
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
    def config_text(self):
        # type: () -> str
        """
        The configuration as a string. For example, prototxt, an ini file, or Python code to evaluate.

        :return str: The configuration.
        """
        return _Model._unwrap_design(self._get_model_data().design)

    @property
    def config_dict(self):
        # type: () -> dict
        """
        The configuration as a dictionary, parsed from the design text. This usually represents the model configuration.
        For example, prototxt, an ini file, or Python code to evaluate.

        :return str: The configuration.
        """
        return self._text_to_config_dict(self.config_text)

    @property
    def labels(self):
        # type: () -> Dict[str, int]
        """
        The label enumeration of string (label) to integer (value) pairs.


        :return dict: A dictionary containing labels enumeration, where the keys are labels and the values as integers.
        """
        return self._get_model_data().labels

    @property
    def task(self):
        # type: () -> str
        """
        Return the creating task id (str)

        :return str: Task ID
        """
        return self._task or self._get_base_model().task

    @property
    def url(self):
        # type: () -> str
        """
        Return the url of the model file (or archived files)

        :return str: Model file URL
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

        :return str: The locally stored file.
        """
        # download model (synchronously) and return local file
        return self._get_base_model().download_model_weights(raise_on_error=raise_on_error)

    def get_weights_package(self, return_path=False, raise_on_error=False):
        # type: (bool, bool) -> Union[str, List[Path]]
        """
        Download the base model package into a temporary directory (extract the files), or return a list of the
        locally stored filenames.

        :param bool return_path: Return the model weights or a list of filenames? (Optional)

            - ``True`` - Download the model weights into a temporary directory, and return the temporary directory path.
            - ``False`` - Return a list of the locally stored filenames. (Default)

        :param bool raise_on_error: If True and the artifact could not be downloaded,
            raise ValueError, otherwise return None on failure and output log warning.

        :return: The model weights, or a list of the locally stored filenames.
        """
        # check if model was packaged
        if self._package_tag not in self._get_model_data().tags:
            raise ValueError('Model is not packaged')

        # download packaged model
        packed_file = self.get_weights(raise_on_error=raise_on_error)

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
        if self._package_tag not in self.tags:
            self.tags.append(self._package_tag)
            self._get_base_model().edit(tags=self.tags)

    @staticmethod
    def _config_dict_to_text(config):
        # if already string return as is
        if isinstance(config, six.string_types):
            return config
        if not isinstance(config, dict):
            raise ValueError("Model configuration only supports dictionary objects")
        try:
            try:
                text = HOCONConverter.to_hocon(ConfigFactory.from_dict(config))
            except Exception:
                # fallback json+pyhocon
                # hack, pyhocon is not very good with dict conversion so we pass through json
                import json
                text = json.dumps(config)
                text = HOCONConverter.to_hocon(ConfigFactory.parse_string(text))

        except Exception:
            raise ValueError("Could not serialize configuration dictionary:\n", config)
        return text

    @staticmethod
    def _text_to_config_dict(text):
        if not isinstance(text, six.string_types):
            raise ValueError("Model configuration parsing only supports string")
        try:
            return ConfigFactory.parse_string(text).as_plain_ordered_dict()
        except pyparsing.ParseBaseException as ex:
            pos = "at char {}, line:{}, col:{}".format(ex.loc, ex.lineno, ex.column)
            six.raise_from(ValueError("Could not parse configuration text ({}):\n{}".format(pos, text)), None)
        except Exception:
            six.raise_from(ValueError("Could not parse configuration text:\n{}".format(text)), None)

    @staticmethod
    def _resolve_config(config_text=None, config_dict=None):
        mutually_exclusive(config_text=config_text, config_dict=config_dict, _require_at_least_one=False)
        if config_dict:
            return InputModel._config_dict_to_text(config_dict)

        return config_text


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
        :return str: a local path to the model (or a downloaded copy of it)
        """
        if extract_archive and self._package_tag in self.tags:
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


class InputModel(Model):
    """
    Load an existing model in the system, search by model id.
    The Model will be read-only and can be used to pre initialize a network
    We can connect the model to a task as input model, then when running remotely override it with the UI.
    """

    _EMPTY_MODEL_ID = _Model._EMPTY_MODEL_ID

    @classmethod
    def import_model(
        cls,
        weights_url,  # type: str
        config_text=None,  # type: Optional[str]
        config_dict=None,  # type: Optional[dict]
        label_enumeration=None,  # type: Optional[Dict[str, int]]
        name=None,  # type: Optional[str]
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

        The **Trains Server** (backend) may already store the model's URL. If the input model's URL is not
        stored, meaning the model is new, then it is imported and Trains stores its metadata.
        If the URL is already stored, the import process stops, Trains issues a warning message, and Trains
        reuses the model.

        In your Python experiment script, after importing the model, you can connect it to the main execution
        Task as an input model using :meth:`InputModel.connect` or :meth:`.Task.connect`. That initializes the
        network.

        .. note::
           Using the **Trains Web-App** (user interface), you can reuse imported models and switch models in
           experiments.

        :param str weights_url: A valid URL for the initial weights file. If the **Trains Web-App** (backend)
            already stores the metadata of a model with the same URL, that existing model is returned
            and Trains ignores all other parameters.

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
                    'background': 0,
                    'person': 1
               }
        :param str name: The name of the newly imported model. (Optional)
        :param tags: The list of tags which describe the model. (Optional)
        :type tags: list(str)
        :param str comment: A comment / description for the model. (Optional)
        :type comment str:
        :param is_package: Is the imported weights file is a package? (Optional)

            - ``True`` - Is a package. Add a package tag to the model.
            - ``False`` - Is not a package. Do not add a package tag. (Default)

        :type is_package: bool
        :param bool create_as_published: Set the model's status to Published? (Optional)

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
        extra = {'system_tags': ["-" + ARCHIVED_TAG]} \
            if Session.check_min_api_version('2.3') else {'tags': ["-" + ARCHIVED_TAG]}
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
            task_id = task.id
        else:
            project_id = None
            task_id = None

        if not framework:
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
        Load an already registered model based on a pre-existing model file (link must be valid).

        If the url to the weights file already exists, the returned object is a Model representing the loaded Model
        If there could not be found any registered model Model with the specified url, None is returned.

        :param weights_url: valid url for the weights file (string).
            examples: "https://domain.com/file.bin" or "s3://bucket/file.bin" or "file:///home/user/file.bin".
            NOTE: if a model with the exact same URL exists, it will be used, and all other arguments will be ignored.
        :param bool load_archived: If True return registered Model with even if they are archived,
            otherwise archived models are ignored,
        :return Model: InputModel object or None if no model could be found
        """
        weights_url = StorageHelper.conform_url(weights_url)
        if not weights_url:
            raise ValueError("Please provide a valid weights_url parameter")
        if not load_archived:
            extra = {'system_tags': ["-" + ARCHIVED_TAG]} \
                if Session.check_min_api_version('2.3') else {'tags': ["-" + ARCHIVED_TAG]}
        else:
            extra = {}

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
        # type: (Optional[str], Optional[dict], Optional[Dict[str, int]]) -> InputModel
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
                    'background': 0,
                    'person': 1
               }
        """
        design = cls._resolve_config(config_text=config_text, config_dict=config_dict)

        this_model = InputModel(model_id=cls._EMPTY_MODEL_ID)
        this_model._base_model = m = _Model(
            cache_dir=None,
            upload_storage_uri=None,
            model_id=cls._EMPTY_MODEL_ID,
        )
        m._data.design = _Model._wrap_design(design)
        m._data.labels = label_enumeration
        return this_model

    def __init__(self, model_id):
        # type: (str) -> None
        """
        :param str model_id: The Trains Id (system UUID) of the input model whose metadata the **Trains Server**
            (backend) stores.
        """
        super(InputModel, self).__init__(model_id)

    @property
    def id(self):
        # type: () -> str
        return self._base_model_id

    def connect(self, task):
        # type: (Task) -> None
        """
        Connect the current model to a Task object, if the model is preexisting. Preexisting models include:

        - Imported models (InputModel objects created using the :meth:`Logger.import_model` method).
        - Models whose metadata is already in the Trains platform, meaning the InputModel object is instantiated
          from the ``InputModel`` class specifying the the model's Trains Id as an argument.
        - Models whose origin is not Trains that are used to create an InputModel object. For example,
          models created using TensorFlow models.

        When the experiment is executed remotely in a worker, the input model already specified in the experiment is
        used.

        .. note::
           The **Trains Web-App** allows you to switch one input model for another and then enqueue the experiment
           to execute in a worker.

        :param object task: A Task object.
        """
        self._set_task(task)

        if running_remotely() and task.input_model and task.is_main_task():
            self._base_model = task.input_model
            self._base_model_id = task.input_model.id
        else:
            # we should set the task input model to point to us
            model = self._get_base_model()
            # try to store the input model id, if it is not empty
            if model.id != self._EMPTY_MODEL_ID:
                task.set_input_model(model_id=model.id)
            # only copy the model design if the task has no design to begin with
            if not self._task._get_model_config_text():
                task._set_model_config(config_text=model.model_design)
            if not self._task.get_labels_enumeration():
                task.set_model_label_enumeration(model.data.labels)

        # If there was an output model connected, it may need to be updated by
        # the newly connected input model
        self.task._reconnect_output_model()


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
       label enumeration using the **Trains Web-App**.
    """

    @property
    def published(self):
        # type: () -> bool
        """
        Get the published state of this model.

        :return bool: ``True`` if the model is published, ``False`` otherwise.
        """
        if not self.id:
            return False
        return self._get_base_model().locked

    @property
    def config_text(self):
        # type: () -> str
        """
        Get the configuration as a string. For example, prototxt, an ini file, or Python code to evaluate.

        :return str: The configuration.
        """
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

        :return dict: The configuration.
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
                'background': 0,
                'person': 1
           }

        :return dict: The label enumeration.
        """
        return self._get_model_data().labels

    @labels.setter
    def labels(self, value):
        # type: (Dict[str, int]) -> None
        """
        Set the label enumeration.

        :param dict value: The label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    'background': 0,
                    'person': 1
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
        label_enumeration=None,  # type: Optional[Dict[str, int]]
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
                    'background': 0,
                    'person': 1
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
        self._base_model = None
        self._floating_data = create_dummy_model(
            design=_Model._wrap_design(config_text),
            labels=label_enumeration or task.get_labels_enumeration(),
            name=name,
            tags=tags,
            comment='{} by task id: {}'.format('Created' if not base_model_id else 'Overwritten', task.id) +
                    ('\n' + comment if comment else ''),
            framework=framework,
            upload_storage_uri=task.output_uri,
        )
        if base_model_id:
            try:
                _base_model = InputModel(base_model_id)._get_base_model()
                _base_model.update(
                    labels=self._floating_data.labels,
                    design=self._floating_data.design,
                    task_id=self._task.id,
                    project_id=self._task.project,
                    name=self._floating_data.name or task.name,
                    comment=('{}\n{}'.format(_base_model.comment, self._floating_data.comment)
                             if _base_model.comment and self._floating_data.comment else
                             (_base_model.comment or self._floating_data.comment)),
                    tags=self._floating_data.tags,
                    framework=self._floating_data.framework,
                    upload_storage_uri=self._floating_data.upload_storage_uri
                )
                self._base_model = _base_model
                self._floating_data = None
                self._base_model.update_for_task(task_id=self._task.id, override_model_id=self.id)
            except Exception:
                pass
        self.connect(task)

    def connect(self, task):
        # type: (Task) -> None
        """
        Connect the current model to a Task object, if the model is a preexisting model. Preexisting models include:

        - Imported models.
        - Models whose metadata the **Trains Server** (backend) is already storing.
        - Models from another source, such as frameworks like TensorFlow.

        :param object task: A Task object.
        """
        if self._task != task:
            raise ValueError('Can only connect preexisting model to task, but this is a fresh model')

        if running_remotely() and task.is_main_task():
            if self._floating_data:
                self._floating_data.design = _Model._wrap_design(self._task._get_model_config_text()) or \
                    self._floating_data.design
                self._floating_data.labels = self._task.get_labels_enumeration() or \
                    self._floating_data.labels
            elif self._base_model:
                self._base_model.update(design=_Model._wrap_design(self._task._get_model_config_text()) or
                                        self._base_model.design)
                self._base_model.update(labels=self._task.get_labels_enumeration() or self._base_model.labels)

        elif self._floating_data is not None:
            # we copy configuration / labels if they exist, obviously someone wants them as the output base model
            if _Model._unwrap_design(self._floating_data.design):
                if not task._get_model_config_text():
                    task._set_model_config(config_text=self._floating_data.design)
            else:
                self._floating_data.design = _Model._wrap_design(self._task._get_model_config_text())

            if self._floating_data.labels:
                task.set_model_label_enumeration(self._floating_data.labels)
            else:
                self._floating_data.labels = self._task.get_labels_enumeration()

        self.task._save_output_model(self)

    def set_upload_destination(self, uri):
        # type: (str) -> None
        """
        Set the URI of the storage destination for uploaded model weight files.
        Supported storage destinations include S3, Google Cloud Storage), and file locations.

        Using this method, files uploads are separate and then a link to each is stored in the model object.

        .. note::
           For storage requiring credentials, the credentials are stored in the Trains configuration file,
           ``~/trains.conf``.

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
        update_comment=True  # type: bool
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
        :param bool auto_delete_file: Delete the temporary file after uploading? (Optional)

            - ``True`` - Delete (Default)
            - ``False`` - Do not delete

        :param str register_uri: The URI of an already uploaded weights file. The URI must be valid. Specify
            ``register_uri`` or ``weights_filename``, but not both.
        :param int iteration: The iteration number.
        :param bool update_comment: Update the model comment with the local weights file name (to maintain
            provenance)? (Optional)

            - ``True`` - Update model comment (Default)
            - ``False`` - Do not update

        :return str: The uploaded URI.
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
            # make sure we delete the previous file, if it exists
            if self._model_local_filename != weights_filename:
                delete_previous_weights_file(self._model_local_filename)
            # store temp filename for deletion next time, if needed
            if auto_delete_file:
                self._model_local_filename = weights_filename

        # make sure the created model is updated:
        model = self._get_force_base_model()
        if not model:
            raise ValueError('Failed creating internal output model')

        # select the correct file extension based on the framework,
        # or update the framework based on the file extension
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

        # make sure that if we are in dev move we report that we are training (not debugging)
        self._task._output_model_updated()

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

        :param weights_filenames: The file names of the locally stored model files. Specify ``weights_filenames``
            or ``weights_path``, but not both.
        :type weights_filenames: list(str)
        :param weights_path: The directory path to a package. All the files in the directory will be uploaded.
            Specify ``weights_path`` or ``weights_filenames``, but not both.
        :type weights_path: str
        :param str upload_uri: The URI of the storage destination for the model weights upload. The default
            is the previously used URI. (Optional)
        :param str target_filename: The newly created filename in the storage destination URI location. The default
            is the value specified in the ``weights_filename`` parameter.  (Optional)
        :param bool auto_delete_file: Delete temporary file after uploading?  (Optional)

            - ``True`` - Delete (Default)
            - ``False`` - Do not delete

        :param int iteration: The iteration number.

        :return str: The uploaded URI for the weights package.
        """
        # create list of files
        if (not weights_filenames and not weights_path) or (weights_filenames and weights_path):
            raise ValueError('Model update weights package should get either '
                             'directory path to pack or a list of files')

        if not weights_filenames:
            weights_filenames = list(map(six.text_type, Path(weights_path).rglob('*')))

        # create packed model from all the files
        fd, zip_file = mkstemp(prefix='model_package.', suffix='.zip')
        try:
            with zipfile.ZipFile(zip_file, 'w', allowZip64=True, compression=zipfile.ZIP_STORED) as zf:
                for filename in weights_filenames:
                    zf.write(filename, arcname=Path(filename).name)
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

        :return bool: The status of the update.

            - ``True`` - Update successful.
            - ``False`` - Update not successful.
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
            self._floating_data.design = _Model._wrap_design(config_text)
            result = Waitable()

        # you can wait on this object
        return result

    def update_labels(self, labels):
        # type: (Dict[str, int]) -> Optional[Waitable]
        """
        Update the label enumeration.

        :param dict labels: The label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    'background': 0,
                    'person': 1
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

    def _get_force_base_model(self):
        if self._base_model:
            return self._base_model

        # create a new model from the task
        self._base_model = self._task.create_output_model()
        # update the model from the task inputs
        labels = self._task.get_labels_enumeration()
        config_text = self._task._get_model_config_text()
        parent = self._task.output_model_id or self._task.input_model_id
        self._base_model.update(
            labels=self._floating_data.labels or labels,
            design=self._floating_data.design or config_text,
            task_id=self._task.id,
            project_id=self._task.project,
            parent_id=parent,
            name=self._floating_data.name or self._task.name,
            comment=self._floating_data.comment,
            tags=self._floating_data.tags,
            framework=self._floating_data.framework,
            upload_storage_uri=self._floating_data.upload_storage_uri
        )

        # remove model floating change set, by now they should have matched the task.
        self._floating_data = None

        # now we have to update the creator task so it points to us
        if self._task.status not in (self._task.TaskStatusEnum.created, self._task.TaskStatusEnum.in_progress):
            self._log.warning('Could not update last created model in Task {}, '
                              'Task status \'{}\' cannot be updated'.format(self._task.id, self._task.status))
        else:
            self._base_model.update_for_task(task_id=self._task.id, override_model_id=self.id)

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


class Waitable(object):
    def wait(self, *_, **__):
        return True
