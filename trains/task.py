import atexit
import os
import signal
import sys
import threading
import time
from argparse import ArgumentParser
from tempfile import mkstemp

try:
    # noinspection PyCompatibility
    from collections.abc import Callable, Sequence as CollectionsSequence
except ImportError:
    from collections import Callable, Sequence as CollectionsSequence

from typing import Optional, Union, Mapping, Sequence, Any, Dict, TYPE_CHECKING

import psutil
import six
from pathlib2 import Path

from .backend_api.services import tasks, projects, queues
from .backend_api.session.session import Session, ENV_ACCESS_KEY, ENV_SECRET_KEY
from .backend_interface.metrics import Metrics
from .backend_interface.model import Model as BackendModel
from .backend_interface.task import Task as _Task
from .backend_interface.task.development.worker import DevWorker
from .backend_interface.task.repo import ScriptInfo
from .backend_interface.util import get_single_result, exact_match_regex, make_message, mutually_exclusive
from .binding.absl_bind import PatchAbsl
from .binding.artifacts import Artifacts, Artifact
from .binding.environ_bind import EnvironmentBind, PatchOsFork
from .binding.frameworks.pytorch_bind import PatchPyTorchModelIO
from .binding.frameworks.tensorflow_bind import TensorflowBinding
from .binding.frameworks.xgboost_bind import PatchXGBoostModelIO
from .binding.joblib_bind import PatchedJoblib
from .binding.matplotlib_bind import PatchedMatplotlib
from .config import config, DEV_TASK_NO_REUSE, get_is_master_node
from .config import running_remotely, get_remote_task_id
from .config.cache import SessionCache
from .debugging.log import LoggerRoot
from .errors import UsageError
from .logger import Logger
from .model import Model, InputModel, OutputModel, ARCHIVED_TAG
from .task_parameters import TaskParameters
from .utilities.args import argparser_parseargs_called, get_argparser_last_args, \
    argparser_update_currenttask
from .utilities.dicts import ReadOnlyDict
from .utilities.proxy_object import ProxyDictPreWrite, ProxyDictPostWrite, flatten_dictionary, \
    nested_from_flat_dictionary, naive_nested_from_flat_dictionary
from .utilities.resource_monitor import ResourceMonitor
from .utilities.seed import make_deterministic
# noinspection PyProtectedMember
from .backend_interface.task.args import _Arguments


if TYPE_CHECKING:
    import pandas
    import numpy
    from PIL import Image


class Task(_Task):
    """
    The ``Task`` class is a code template for a Task object which, together with its connected experiment components,
    represents the current running experiment. These connected components include hyperparameters, loggers,
    configuration, label enumeration, models, and other artifacts.

    The term "main execution Task" refers to the Task context for current running experiment. Python experiment scripts
    can create one, and only one, main execution Task. It is a traceable, and after a script runs and Trains stores
    the Task in the **Trains Server** (backend), it is modifiable, reproducible, executable by a worker, and you
    can duplicate it for further experimentation.

    The ``Task`` class and its methods allow you to create and manage experiments, as well as perform
    advanced experimentation functions, such as autoML.

    .. warning::
       Do not construct Task objects directly. Use one of the methods listed below to create experiments or
       reference existing experiments.

    For detailed information about creating Task objects, see the following methods:

    - :meth:`Task.init` - Create a new reproducible Task, or reuse one.
    - :meth:`Task.create` - Create a new non-reproducible Task.
    - :meth:`Task.current_task` - Get the current running Task.
    - :meth:`Task.get_task` - Get another Task (whose metadata the **Trains Server** maintains).

    .. note::
       The **Trains** documentation often refers to a Task as, "Task (experiment)".

       "Task" refers to the class in the Trains Python Client Package, the object in your Python experiment script,
       and the entity with which **Trains Server** and **Trains Agent** work.

       "Experiment" refers to your deep learning solution, including its connected components, inputs, and outputs,
       and is the experiment you can view, analyze, compare, modify, duplicate, and manage using the Trains
       **Web-App** (UI).

       Therefore, a "Task" is effectively an "experiment", and "Task (experiment)" encompasses its usage throughout
       the Trains.

        The exception to this Task behavior is sub-tasks (non-reproducible Tasks), which do not use the main execution
        Task. Creating a sub-task always creates a new Task with a new Task Id.
    """

    TaskTypes = _Task.TaskTypes

    NotSet = object()

    __create_protection = object()
    __main_task = None  # type: Optional[Task]
    __exit_hook = None
    __forked_proc_main_pid = None
    __task_id_reuse_time_window_in_hours = float(config.get('development.task_reuse_time_window_in_hours', 24.0))
    __detect_repo_async = config.get('development.vcs_repo_detect_async', False)
    __default_output_uri = config.get('development.default_output_uri', None)

    class _ConnectedParametersType(object):
        argparse = "argument_parser"
        dictionary = "dictionary"
        task_parameters = "task_parameters"

        @classmethod
        def _options(cls):
            return {
                var for var, val in vars(cls).items()
                if isinstance(val, six.string_types)
            }

    def __init__(self, private=None, **kwargs):
        """
        .. warning::
            **Do not construct Task manually!**
            Please use :meth:`Task.init` or :meth:`Task.get_task`
        """
        if private is not Task.__create_protection:
            raise UsageError(
                'Task object cannot be instantiated externally, use Task.current_task() or Task.get_task(...)')
        self._repo_detect_lock = threading.RLock()

        super(Task, self).__init__(**kwargs)
        self._arguments = _Arguments(self)
        self._logger = None
        self._last_input_model_id = None
        self._connected_output_model = None
        self._dev_worker = None
        self._connected_parameter_type = None
        self._detect_repo_async_thread = None
        self._resource_monitor = None
        self._artifacts_manager = Artifacts(self)
        self._calling_filename = None
        # register atexit, so that we mark the task as stopped
        self._at_exit_called = False

    @classmethod
    def current_task(cls):
        # type: () -> Task
        """
        Get the current running Task (experiment). This is the main execution Task (task context) returned as a Task
        object.

        :return: The current running Task (experiment).
        """
        return cls.__main_task

    @classmethod
    def init(
        cls,
        project_name=None,  # type: Optional[str]
        task_name=None,  # type: Optional[str]
        task_type=TaskTypes.training,  # type: Task.TaskTypes
        reuse_last_task_id=True,  # type: bool
        output_uri=None,  # type: Optional[str]
        auto_connect_arg_parser=True,  # type: Union[bool, Mapping[str, bool]]
        auto_connect_frameworks=True,  # type: Union[bool, Mapping[str, bool]]
        auto_resource_monitoring=True,  # type: bool
    ):
        # type: (...) -> Task
        """
        Creates a new Task (experiment), or returns the existing Task, depending upon the following:

        - if **any** of the following are true, Trains creates a new Task and a new Task Id:

          - a Task in the same project with same name does not exist, **or**
          - a Task in the same project with same name does exist and its status is ``Published``, **or**
          - the :paramref:`~.init.reuse_last_task_id` parameter is assigned ``False``.

        - if **all** of the following are true, Trains returns the existing Task with the existing Task Id:

          - a Task in the same project with the same name does exist, **and**
          - the Task's status is ``Draft``, ``Completed``, ``Failed``, or ``Aborted``, **and**
          - the ``reuse_last_task_id`` parameter is the default value of ``True``.

            .. warning::
               When a Python experiment script runs using an existing Task, it overwrites previous experiment output.

        :param str project_name: The name of the project in which the experiment will be created. If the project does
            not exist, it is created. If ``project_name`` is ``None``, the repository name is used. (Optional)
        :param str task_name: The name of Task (experiment). If ``task_name`` is ``None``, the Python experiment
            script's file name is used. (Optional)
        :param task_type: The task type.

            The values are:

            - ``TaskTypes.training`` (Default)
            - ``TaskTypes.train``
            - ``TaskTypes.testing``
            - ``TaskTypes.inference``
        :type task_type: TaskTypeEnum(value)
        :param bool reuse_last_task_id: Force a new Task (experiment) with a new Task Id, but
            the same project and Task names.

            .. note::
               Trains creates the new Task Id using the previous Id, which is stored in the data cache folder.

            The values are:

            - ``True`` - Reuse the last Task Id. (Default)
            - ``False`` - Force a new Task (experiment).
            - A string - In addition to a boolean, you can use a string to set a specific value for Task Id
              (instead of the system generated UUID).

        :param str output_uri: The default location for output models and other artifacts. In the default location,
            Trains creates a subfolder for the output. The subfolder structure is the following:

                <output destination name> / <project name> / <task name>.<Task Id>

            The following are examples of ``output_uri`` values for the supported locations:

            - A shared folder: ``/mnt/share/folder``
            - S3: ``s3://bucket/folder``
            - Google Cloud Storage: ``gs://bucket-name/folder``
            - Azure Storage: ``azure://company.blob.core.windows.net/folder/``

            .. important::
               For cloud storage, you must install the **Trains** package for your cloud storage type,
               and then configure your storage credentials. For detailed information, see
               `Trains Python Client Extras <./references/trains_extras_storage/>`_ in the "Trains Python Client
               Reference" section.

        :param auto_connect_arg_parser: Automatically connect an argparse object to the Task?

            The values are:

            - ``True`` - Automatically connect. (Default)
            -  ``False`` - Do not automatically connect.
            - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of connected
              arguments. The dictionary keys are argparse variable names and the values are booleans,
              False value will exclude the specified argument from the Task's parameter section.
              Keys missing from the dictionary default to ``True``, and an empty dictionary defaults to ``False``.

            For example:

            .. code-block:: py

               auto_connect_arg_parser={'do_not_include_me': False, }

            .. note::
               To manually connect an argparse, use :meth:`Task.connect`.

        :param auto_connect_frameworks: Automatically connect frameworks? This includes patching MatplotLib, XGBoost,
            scikit-learn, Keras callbacks, and TensorBoard/X to serialize plots, graphs, and the model location to
            the **Trains Server** (backend), in addition to original output destination.

            The values are:

            - ``True`` - Automatically connect (Default)
            -  ``False`` - Do not automatically connect
            - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of connected
              frameworks. The dictionary keys are frameworks and the values are booleans.
              Keys missing from the dictionary default to ``True``, and an empty dictionary defaults to ``False``.

            For example:

            .. code-block:: py

               auto_connect_frameworks={'matplotlib': True, 'tensorflow': True, 'pytorch': True,
                    'xgboost': True, 'scikit': True}

        :type auto_connect_frameworks: bool or dict
        :param bool auto_resource_monitoring: Automatically create machine resource monitoring plots?
            These plots appear in in the **Trains Web-App (UI)**, **RESULTS** tab, **SCALARS** sub-tab,
            with a title of **:resource monitor:**.

            The values are:

            - ``True`` - Automatically create resource monitoring plots. (Default)
            - ``False`` - Do not automatically create.

        :return: The main execution Task (Task context).
        """

        def verify_defaults_match():
            validate = [
                ('project name', project_name, cls.__main_task.get_project_name()),
                ('task name', task_name, cls.__main_task.name),
                ('task type', str(task_type), str(cls.__main_task.task_type)),
            ]

            for field, default, current in validate:
                if default is not None and default != current:
                    raise UsageError(
                        "Current task already created "
                        "and requested {field} '{default}' does not match current {field} '{current}'. "
                        "If you wish to create additional tasks use `Task.create`".format(
                            field=field,
                            default=default,
                            current=current,
                        )
                    )

        if cls.__main_task is not None:
            # if this is a subprocess, regardless of what the init was called for,
            # we have to fix the main task hooks and stdout bindings
            if cls.__forked_proc_main_pid != os.getpid() and cls.__is_subprocess():
                if task_type is None:
                    task_type = cls.__main_task.task_type
                # make sure we only do it once per process
                cls.__forked_proc_main_pid = os.getpid()
                # make sure we do not wait for the repo detect thread
                cls.__main_task._detect_repo_async_thread = None
                cls.__main_task._dev_worker = None
                cls.__main_task._resource_monitor = None
                # remove the logger from the previous process
                logger = cls.__main_task.get_logger()
                logger.set_flush_period(None)
                # create a new logger (to catch stdout/err)
                cls.__main_task._logger = None
                cls.__main_task._reporter = None
                cls.__main_task.get_logger()
                cls.__main_task._artifacts_manager = Artifacts(cls.__main_task)
                # unregister signal hooks, they cause subprocess to hang
                # noinspection PyProtectedMember
                cls.__main_task.__register_at_exit(cls.__main_task._at_exit)
                # TODO: Check if the signal handler method is safe enough, for the time being, do not unhook
                # cls.__main_task.__register_at_exit(None, only_remove_signal_and_exception_hooks=True)

            if not running_remotely():
                verify_defaults_match()

            return cls.__main_task

        is_sub_process_task_id = None
        # check that we are not a child process, in that case do nothing.
        # we should not get here unless this is Windows platform, all others support fork
        if cls.__is_subprocess():
            class _TaskStub(object):
                def __call__(self, *args, **kwargs):
                    return self

                def __getattr__(self, attr):
                    return self

                def __setattr__(self, attr, val):
                    pass

            is_sub_process_task_id = cls.__get_master_id_task_id()
            # we could not find a task ID, revert to old stub behaviour
            if not is_sub_process_task_id:
                return _TaskStub()
        elif running_remotely() and not get_is_master_node():
            # make sure we only do it once per process
            cls.__forked_proc_main_pid = os.getpid()
            # make sure everyone understands we should act as if we are a subprocess (fake pid 1)
            cls.__update_master_pid_task(pid=1, task=get_remote_task_id())
        else:
            # set us as master process (without task ID)
            cls.__update_master_pid_task()
            is_sub_process_task_id = None

        if task_type is None:
            # Backwards compatibility: if called from Task.current_task and task_type
            # was not specified, keep legacy default value of TaskTypes.training
            task_type = cls.TaskTypes.training
        elif isinstance(task_type, six.string_types):
            if task_type not in Task.TaskTypes.__members__:
                raise ValueError("Task type '{}' not supported, options are: {}".format(
                    task_type, Task.TaskTypes.__members__.keys()))
            task_type = Task.TaskTypes.__members__[str(task_type)]

        try:
            if not running_remotely():
                # if this is the main process, create the task
                if not is_sub_process_task_id:
                    task = cls._create_dev_task(
                        project_name,
                        task_name,
                        task_type,
                        reuse_last_task_id,
                        detect_repo=False if (isinstance(auto_connect_frameworks, dict) and
                                              not auto_connect_frameworks.get('detect_repository', True)) else True
                    )
                    # set defaults
                    if output_uri:
                        task.output_uri = output_uri
                    elif cls.__default_output_uri:
                        task.output_uri = cls.__default_output_uri
                    # store new task ID
                    cls.__update_master_pid_task(task=task)
                else:
                    # subprocess should get back the task info
                    task = Task.get_task(task_id=is_sub_process_task_id)
            else:
                # if this is the main process, create the task
                if not is_sub_process_task_id:
                    task = cls(
                        private=cls.__create_protection,
                        task_id=get_remote_task_id(),
                        log_to_backend=False,
                    )
                    if cls.__default_output_uri and not task.output_uri:
                        task.output_uri = cls.__default_output_uri
                    # store new task ID
                    cls.__update_master_pid_task(task=task)
                    # make sure we are started
                    task.started(ignore_errors=True)
                else:
                    # subprocess should get back the task info
                    task = Task.get_task(task_id=is_sub_process_task_id)
        except Exception:
            raise
        else:
            Task.__main_task = task
            # register the main task for at exit hooks (there should only be one)
            task.__register_at_exit(task._at_exit)
            # patch OS forking
            PatchOsFork.patch_fork()
            if auto_connect_frameworks:
                is_auto_connect_frameworks_bool = not isinstance(auto_connect_frameworks, dict)
                if is_auto_connect_frameworks_bool or auto_connect_frameworks.get('scikit', True):
                    PatchedJoblib.update_current_task(task)
                if is_auto_connect_frameworks_bool or auto_connect_frameworks.get('matplotlib', True):
                    PatchedMatplotlib.update_current_task(Task.__main_task)
                if is_auto_connect_frameworks_bool or auto_connect_frameworks.get('tensorflow', True):
                    PatchAbsl.update_current_task(Task.__main_task)
                    TensorflowBinding.update_current_task(task)
                if is_auto_connect_frameworks_bool or auto_connect_frameworks.get('pytorch', True):
                    PatchPyTorchModelIO.update_current_task(task)
                if is_auto_connect_frameworks_bool or auto_connect_frameworks.get('xgboost', True):
                    PatchXGBoostModelIO.update_current_task(task)
            if auto_resource_monitoring and not is_sub_process_task_id:
                task._resource_monitor = ResourceMonitor(
                    task, report_mem_used_per_process=not config.get(
                        'development.worker.report_global_mem_used', False))
                task._resource_monitor.start()

            # make sure all random generators are initialized with new seed
            make_deterministic(task.get_random_seed())

            if auto_connect_arg_parser:
                EnvironmentBind.update_current_task(Task.__main_task)

                # Patch ArgParser to be aware of the current task
                argparser_update_currenttask(Task.__main_task)

                # set excluded arguments
                if isinstance(auto_connect_arg_parser, dict):
                    task._arguments.exclude_parser_args(auto_connect_arg_parser)

                # Check if parse args already called. If so, sync task parameters with parser
                if argparser_parseargs_called():
                    parser, parsed_args = get_argparser_last_args()
                    task._connect_argparse(parser=parser, parsed_args=parsed_args)
            elif argparser_parseargs_called():
                # actually we have nothing to do, in remote running, the argparser will ignore
                # all non argparser parameters, only caveat if parameter connected with the same name
                # as the argparser this will be solved once sections are introduced to parameters
                pass

        # Make sure we start the logger, it will patch the main logging object and pipe all output
        # if we are running locally and using development mode worker, we will pipe all stdout to logger.
        # The logger will automatically take care of all patching (we just need to make sure to initialize it)
        logger = task.get_logger()
        # show the debug metrics page in the log, it is very convenient
        if not is_sub_process_task_id:
            logger.report_text(
                'TRAINS results page: {}'.format(task.get_output_log_web_page()),
            )
        # Make sure we start the dev worker if required, otherwise it will only be started when we write
        # something to the log.
        task._dev_mode_task_start()

        return task

    @classmethod
    def create(cls, project_name=None, task_name=None, task_type=TaskTypes.training):
        # type: (Optional[str], Optional[str], TaskTypes) -> Task
        """
        Create a new, non-reproducible Task (experiment). This is called a sub-task.

        .. note::
           - This method always creates a new Task.
           - To create reproducible Tasks, use the :meth:`Task.init` method.

        :param str project_name: The name of the project in which the experiment will be created.
            If ``project_name`` is ``None``, and the main execution Task is initialized (see :meth:`Task.init`),
            then the main execution Task's project is used. Otherwise, if the project does
            not exist, it is created. (Optional)
        :param str task_name: The name of Task (experiment).
        :param task_type: The task type. (Optional)

            The values are:

            - ``TaskTypes.training`` (default)
            - ``TaskTypes.testing``
            - ``TaskTypes.inference``
            - ``TaskTypes.data_processing``
            - ``TaskTypes.application``
            - ``TaskTypes.monitor``
            - ``TaskTypes.controller``
            - ``TaskTypes.optimizer``
            - ``TaskTypes.service``
            - ``TaskTypes.qc``
            - ``TaskTypes.custom``

        :type task_type: TaskTypeEnum(value)
        :return: A new experiment.
        """
        if not project_name:
            if not cls.__main_task:
                raise ValueError("Please provide project_name, no global task context found "
                                 "(Task.current_task hasn't been called)")
            project_name = cls.__main_task.get_project_name()

        try:
            task = cls(
                private=cls.__create_protection,
                project_name=project_name,
                task_name=task_name,
                task_type=task_type,
                log_to_backend=False,
                force_create=True,
            )
        except Exception:
            raise
        return task

    @classmethod
    def get_task(cls, task_id=None, project_name=None, task_name=None):
        # type: (Optional[str], Optional[str], Optional[str]) -> Task
        """
        Get a Task by Id, or project name / task name combination.

        :param str task_id: The Id (system UUID) of the experiment to get.
            If specified, ``project_name`` and ``task_name`` are ignored.
        :param str project_name: The project name of the Task to get.
        :param str task_name: The name of the Task within ``project_name`` to get.

        :return: The Task specified by Id, or project name / experiment name combination.
        """
        return cls.__get_task(task_id=task_id, project_name=project_name, task_name=task_name)

    @classmethod
    def get_tasks(cls, task_ids=None, project_name=None, task_name=None, task_filter=None):
        # type: (Optional[Sequence[str]], Optional[str], Optional[str], Optional[Dict]) -> Sequence[Task]
        """
        Get a list of Tasks by one of the following:

        - A list of specific Task Ids.
        - All Tasks in a project matching a full or partial Task name.
        - All Tasks in any project matching a full or partial Task name.

        :param list(str) task_ids: The Ids (system UUID) of experiments to get.
            If ``task_ids`` specified, then ``project_name`` and ``task_name`` are ignored.

        :param str project_name: The project name of the Tasks to get. To get the experiment
            in all projects, use the default value of ``None``. (Optional)
        :param str task_name: The full name or partial name of the Tasks to match within the specified
            ``project_name`` (or all projects if ``project_name`` is ``None``).
            This method supports regular expressions for name matching. (Optional)

        :param list(str) task_ids: list of unique task id string (if exists other parameters are ignored)
        :param str project_name: project name (str) the task belongs to (use None for all projects)
        :param str task_name: task name (str) in within the selected project
            Return any partial match of task_name, regular expressions matching is also supported
            If None is passed, returns all tasks within the project
        :param dict task_filter: filter and order Tasks. See service.tasks.GetAllRequest for details
        :return: The Tasks specified by the parameter combinations (see the parameters).
        """
        return cls.__get_tasks(task_ids=task_ids, project_name=project_name,
                               task_name=task_name, **(task_filter or {}))

    @property
    def output_uri(self):
        # type: () -> str
        return self.storage_uri

    @output_uri.setter
    def output_uri(self, value):
        # type: (str) -> None
        # check if we have the correct packages / configuration
        if value and value != self.storage_uri:
            from .storage.helper import StorageHelper
            helper = StorageHelper.get(value)
            if not helper:
                raise ValueError("Could not get access credentials for '{}' "
                                 ", check configuration file ~/trains.conf".format(value))
            helper.check_write_permissions(value)
        self.storage_uri = value

    @property
    def artifacts(self):
        # type: () -> Dict[str, Artifact]
        """
        A read-only dictionary of Task artifacts (name, artifact).

        :return: The artifacts.
        """
        if not Session.check_min_api_version('2.3'):
            return ReadOnlyDict()
        artifacts_pairs = []
        if self.data.execution and self.data.execution.artifacts:
            artifacts_pairs = [(a.key, Artifact(a)) for a in self.data.execution.artifacts]
        if self._artifacts_manager:
            artifacts_pairs += list(self._artifacts_manager.registered_artifacts.items())
        return ReadOnlyDict(artifacts_pairs)

    @property
    def models(self):
        # type: () -> Dict[str, Sequence[Model]]
        """
        Read-only dictionary of the Task's loaded/stored models

        :return: dictionary of models loaded/stored {'input': list(Model), 'output': list(Model)}
        """
        return self.get_models()

    @classmethod
    def clone(
            cls,
            source_task=None,  # type: Optional[Union[Task, str]]
            name=None,  # type: Optional[str]
            comment=None,  # type: Optional[str]
            parent=None,  # type: Optional[str]
            project=None,  # type: Optional[str]
    ):
        # type: (...) -> Task
        """
        Create a duplicate (a clone) of a Task (experiment). The status of the cloned Task is ``Draft``
        and modifiable.

        Use this method to manage experiments and for autoML.

        :param str source_task: The Task to clone. Specify a Task object or a Task Id. (Optional)
        :param str name: The name of the new cloned Task. (Optional)
        :param str comment: A comment / description for the new cloned Task. (Optional)
        :param str parent: The Id of the parent Task of the new Task.

            - If ``parent`` is not specified, then ``parent`` is set to ``source_task.parent``.
            - If ``parent`` is not specified and ``source_task.parent`` is not available, then
              ``parent`` set to to ``source_task``.

        :param str project: The Id of the project in which to create the new Task.
            If ``None``, the new task inherits the original Task's project. (Optional)

        :return: The new cloned Task (experiment).
        """
        assert isinstance(source_task, (six.string_types, Task))
        if not Session.check_min_api_version('2.4'):
            raise ValueError("Trains-server does not support DevOps features, "
                             "upgrade trains-server to 0.12.0 or above")

        task_id = source_task if isinstance(source_task, six.string_types) else source_task.id
        if not parent:
            if isinstance(source_task, six.string_types):
                source_task = cls.get_task(task_id=source_task)
            parent = source_task.id if not source_task.parent else source_task.parent
        elif isinstance(parent, Task):
            parent = parent.id

        cloned_task_id = cls._clone_task(cloned_task_id=task_id, name=name, comment=comment,
                                         parent=parent, project=project)
        cloned_task = cls.get_task(task_id=cloned_task_id)
        return cloned_task

    @classmethod
    def enqueue(cls, task, queue_name=None, queue_id=None):
        # type: (Union[Task, str], Optional[str], Optional[str]) -> Any
        """
        Enqueue a Task for execution, by adding it to an execution queue.

        .. note::
           A worker daemon must be listening at the queue for the worker to fetch the Task and execute it,
           see `Use Case Examples <../trains_agent_ref/#use-case-examples>`_ on the "Trains Agent
           Reference page.

        :param task: The Task to enqueue. Specify a Task object or Task Id.
        :type task: Task object / str
        :param str queue_name: The name of the queue. If not specified, then ``queue_id`` must be specified.
        :param str queue_id: The Id of the queue. If not specified, then ``queue_name`` must be specified.

        :return: An enqueue JSON response.

            .. code-block:: javascript

               {
                    "queued": 1,
                    "updated": 1,
                    "fields": {
                        "status": "queued",
                        "status_reason": "",
                        "status_message": "",
                        "status_changed": "2020-02-24T15:05:35.426770+00:00",
                        "last_update": "2020-02-24T15:05:35.426770+00:00",
                        "execution.queue": "2bd96ab2d9e54b578cc2fb195e52c7cf"
                        }
                }

            - ``queued``  - The number of Tasks enqueued (an integer or ``null``).
            - ``updated`` - The number of Tasks updated (an integer or ``null``).
            - ``fields``

              - ``status`` - The status of the experiment.
              - ``status_reason`` - The reason for the last status change.
              - ``status_message`` - Information about the status.
              - ``status_changed`` - The last status change date and time (ISO 8601 format).
              - ``last_update`` - The last Task update time, including Task creation, update, change, or events for
                this task (ISO 8601 format).
              - ``execution.queue`` - The Id of the queue where the Task is enqueued. ``null`` indicates not enqueued.
        """
        assert isinstance(task, (six.string_types, Task))
        if not Session.check_min_api_version('2.4'):
            raise ValueError("Trains-server does not support DevOps features, "
                             "upgrade trains-server to 0.12.0 or above")

        # make sure we have wither name ot id
        mutually_exclusive(queue_name=queue_name, queue_id=queue_id)

        task_id = task if isinstance(task, six.string_types) else task.id
        session = cls._get_default_session()
        if not queue_id:
            req = queues.GetAllRequest(name=exact_match_regex(queue_name), only_fields=["id"])
            res = cls._send(session=session, req=req)
            if not res.response.queues:
                raise ValueError('Could not find queue named "{}"'.format(queue_name))
            queue_id = res.response.queues[0].id
            if len(res.response.queues) > 1:
                LoggerRoot.get_base_logger().info("Multiple queues with name={}, selecting queue id={}".format(
                    queue_name, queue_id))

        req = tasks.EnqueueRequest(task=task_id, queue=queue_id)
        res = cls._send(session=session, req=req)
        resp = res.response
        return resp

    @classmethod
    def dequeue(cls, task):
        # type: (Union[Task, str]) -> Any
        """
        Dequeue (remove) a Task from an execution queue.

        :param task: The Task to dequeue. Specify a Task object or Task Id.
        :type task: Task object / str

        :return: A dequeue JSON response.

        .. code-block:: javascript

           {
                "dequeued": 1,
                "updated": 1,
                "fields": {
                    "status": "created",
                    "status_reason": "",
                    "status_message": "",
                    "status_changed": "2020-02-24T16:43:43.057320+00:00",
                    "last_update": "2020-02-24T16:43:43.057320+00:00",
                    "execution.queue": null
                    }
            }

        - ``dequeued``  - The number of Tasks enqueued (an integer or ``null``).
        - ``fields``

          - ``status`` - The status of the experiment.
          - ``status_reason`` - The reason for the last status change.
          - ``status_message`` - Information about the status.
          - ``status_changed`` - The last status change date and time in ISO 8601 format.
          - ``last_update`` - The last time the Task was created, updated,
                changed or events for this task were reported.
          - ``execution.queue`` - The Id of the queue where the Task is enqueued. ``null`` indicates not enqueued.

        - ``updated`` - The number of Tasks updated (an integer or ``null``).
        """
        assert isinstance(task, (six.string_types, Task))
        if not Session.check_min_api_version('2.4'):
            raise ValueError("Trains-server does not support DevOps features, "
                             "upgrade trains-server to 0.12.0 or above")

        task_id = task if isinstance(task, six.string_types) else task.id
        session = cls._get_default_session()
        req = tasks.DequeueRequest(task=task_id)
        res = cls._send(session=session, req=req)
        resp = res.response
        return resp

    def add_tags(self, tags):
        # type: (Union[Sequence[str], str]) -> None
        """
        Add Tags to this task. Old tags are not deleted. When executing a Task (experiment) remotely,
        this method has no effect).

        :param tags: A list of tags which describe the Task to add.
        :type tags: str or iterable of str
        """

        if not running_remotely() or not self.is_main_task():
            if isinstance(tags, six.string_types):
                tags = tags.split(" ")

            self.data.tags.extend(tags)
            self._edit(tags=list(set(self.data.tags)))

    def connect(self, mutable):
        # type: (Any) -> Any
        """
        Connect an object to a Task object. This connects an experiment component (part of an experiment) to the
        experiment. For example, connect hyperparameters or models.

        :param object mutable: The experiment component to connect. The object can be any object Task supports
            integrating, including:

            - argparse - An argparse object for parameters.
            - dict - A dictionary for parameters.
            - TaskParameters - A TaskParameters object.
            - model - A model object for initial model warmup, or for model update/snapshot uploading.

        :return: The result returned when connecting the object, if supported.

        :raise: Raise an exception on unsupported objects.
        """

        dispatch = (
            (OutputModel, self._connect_output_model),
            (InputModel, self._connect_input_model),
            (ArgumentParser, self._connect_argparse),
            (dict, self._connect_dictionary),
            (TaskParameters, self._connect_task_parameters),
        )

        for mutable_type, method in dispatch:
            if isinstance(mutable, mutable_type):
                return method(mutable)

        raise Exception('Unsupported mutable type %s: no connect function found' % type(mutable).__name__)

    def connect_configuration(self, configuration):
        # type: (Union[Mapping, Path, str]) -> Union[Mapping, Path, str]
        """
        Connect a configuration dictionary or configuration file (pathlib.Path / str) to a Task object.
        This method should be called before reading the configuration file.

        Later, when creating an output model, the model will include the contents of the configuration dictionary
        or file.

        For example, a local file:

        .. code-block:: py

           config_file = task.connect_configuration(config_file)
           my_params = json.load(open(config_file,'rt'))

        A parameter dictionary:

        .. code-block:: py

           my_params = task.connect_configuration(my_params)

        :param configuration: The configuration. This is usually the configuration used in the model training process.
            Specify one of the following:

            - A dictionary - A dictionary containing the configuration. Trains stores the configuration in
              the **Trains Server** (backend), in a HOCON format (JSON-like format) which is editable.
            - A ``pathlib2.Path`` string - A path to the configuration file. Trains stores the content of the file.
              A local path must be relative path. When executing a Task remotely in a worker, the contents brought
              from the **Trains Server** (backend) overwrites the contents of the file.

        :type configuration: dict, pathlib.Path/str

        :return: If a dictionary is specified, then a dictionary is returned. If pathlib2.Path / string is
            specified, then a path to a local configuration file is returned. Configuration object.
        """
        if not isinstance(configuration, (dict, Path, six.string_types)):
            raise ValueError("connect_configuration supports `dict`, `str` and 'Path' types, "
                             "{} is not supported".format(type(configuration)))

        # parameter dictionary
        if isinstance(configuration, dict):
            def _update_config_dict(task, config_dict):
                # noinspection PyProtectedMember
                task._set_model_config(config_dict=config_dict)

            if not running_remotely() or not self.is_main_task():
                self._set_model_config(config_dict=configuration)
                configuration = ProxyDictPostWrite(self, _update_config_dict, **configuration)
            else:
                configuration.clear()
                configuration.update(self._get_model_config_dict())
                configuration = ProxyDictPreWrite(False, False, **configuration)
            return configuration

        # it is a path to a local file
        if not running_remotely() or not self.is_main_task():
            # check if not absolute path
            configuration_path = Path(configuration)
            if not configuration_path.is_file():
                ValueError("Configuration file does not exist")
            try:
                with open(configuration_path.as_posix(), 'rt') as f:
                    configuration_text = f.read()
            except Exception:
                raise ValueError("Could not connect configuration file {}, file could not be read".format(
                    configuration_path.as_posix()))
            self._set_model_config(config_text=configuration_text)
            return configuration
        else:
            configuration_text = self._get_model_config_text()
            configuration_path = Path(configuration)
            fd, local_filename = mkstemp(prefix='trains_task_config_',
                                         suffix=configuration_path.suffixes[-1] if
                                         configuration_path.suffixes else '.txt')
            os.write(fd, configuration_text.encode('utf-8'))
            os.close(fd)
            return Path(local_filename) if isinstance(configuration, Path) else local_filename

    def connect_label_enumeration(self, enumeration):
        # type: (Dict[str, int]) -> Dict[str, int]
        """
        Connect a label enumeration dictionary to a Task (experiment) object.

        Later, when creating an output model, the model will include the label enumeration dictionary.

        :param dict enumeration: A label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    'background': 0,
                    'person': 1
               }

        :return: The label enumeration dictionary (JSON).
        """
        if not isinstance(enumeration, dict):
            raise ValueError("connect_label_enumeration supports only `dict` type, "
                             "{} is not supported".format(type(enumeration)))

        if not running_remotely() or not self.is_main_task():
            self.set_model_label_enumeration(enumeration)
        else:
            # pop everything
            enumeration.clear()
            enumeration.update(self.get_labels_enumeration())
        return enumeration

    def get_logger(self):
        # type: () -> Logger
        """
        Get a Logger object for reporting, for this task context. You can view all Logger report output associated with
        the Task for which this method is called, including metrics, plots, text, tables, and images, in the
        **Trains Web-App (UI)**.

        :return: The Logger for the Task (experiment).
        """
        return self._get_logger()

    def mark_started(self):
        """
        Manually mark a Task as started (happens automatically)
        """
        # UI won't let us see metrics if we're not started
        self.started()
        self.reload()

    def mark_stopped(self):
        """
        Manually mark a Task as stopped (also used in :meth:`_at_exit`)
        """
        # flush any outstanding logs
        self.flush(wait_for_uploads=True)
        # mark task as stopped
        self.stopped()

    def flush(self, wait_for_uploads=False):
        # type: (bool) -> bool
        """
        Flush any outstanding reports or console logs.

        :param bool wait_for_uploads: Wait for all outstanding uploads to complete before existing the flush?

            - ``True`` - Wait
            - ``False`` - Do not wait (Default)
        """

        # make sure model upload is done
        if BackendModel.get_num_results() > 0 and wait_for_uploads:
            BackendModel.wait_for_results()

        # flush any outstanding logs
        if self._logger:
            # noinspection PyProtectedMember
            self._logger._flush_stdout_handler()
        if self._reporter:
            self.reporter.flush()
        LoggerRoot.flush()

        return True

    def reset(self, set_started_on_success=False, force=False):
        # type: (bool, bool) -> None
        """
        Reset a Task. Trains reloads a Task after a successful reset.
        When a worker executes a Task remotely, the Task does not reset unless
        the ``force`` parameter is set to ``True`` (this avoids accidentally clearing logs and metrics).

        :param bool set_started_on_success: If successful, automatically set the Task to started?

            - ``True`` - If successful, set to started.
            - ``False`` - If successful, do not set to started. (Default)

        :param bool force: Force a Task reset, even when executing the Task (experiment) remotely in a worker?

            - ``True`` - Force
            - ``False`` - Do not force (Default)
        """
        if not running_remotely() or not self.is_main_task() or force:
            super(Task, self).reset(set_started_on_success=set_started_on_success)

    def close(self):
        """
        Close the current Task. Enables you to manually shutdown the task.

        .. warning::
           Only call :meth:`Task.close` if you are certain the Task is not needed.
        """
        if self._at_exit_called:
            return

        # store is main before we call at_exit, because will will Null it
        is_main = self.is_main_task()

        # wait for repository detection (5 minutes should be reasonable time to detect all packages)
        if self._logger and not self.__is_subprocess():
            self._wait_for_repo_detection(timeout=300.)

        self.__shutdown()
        # unregister atexit callbacks and signal hooks, if we are the main task
        if is_main:
            self.__register_at_exit(None)

    def register_artifact(self, name, artifact, metadata=None, uniqueness_columns=True):
        # type: (str, pandas.DataFrame, Dict, Union[bool, Sequence[str]]) -> None
        """
        Register (add) an artifact for the current Task. Registered artifacts are dynamically sychronized with the
        **Trains Server** (backend). If a registered artifact is updated, the update is stored in the
        **Trains Server** (backend). Registered artifacts are primarily used for Data Audition.

        The currently supported registered artifact object type is a pandas.DataFrame.

        See also :meth:`Task.unregister_artifact` and :meth:`Task.get_registered_artifacts`.

        .. note::
           Trains also supports uploaded artifacts which are one-time uploads of static artifacts that are not
           dynamically sychronized with the **Trains Server** (backend). These static artifacts include
           additional object types. For more information, see :meth:`Task.upload_artifact`.

        :param str name: The name of the artifact.

         .. warning::
            If an artifact with the same name was previously registered, it is overwritten.
        :param object artifact: The artifact object.
        :param dict metadata: A dictionary of key-value pairs for any metadata. This dictionary appears with the
            experiment in the **Trains Web-App (UI)**, **ARTIFACTS** tab.
        :param uniqueness_columns: A Sequence of columns for artifact uniqueness comparison criteria, or the default
            value of ``True``. If ``True``, the artifact uniqueness comparison criteria is all the columns,
            which is the same as ``artifact.columns``.
        :type uniqueness_columns: sequence, str, ``True``
        """
        if not isinstance(uniqueness_columns, CollectionsSequence) and uniqueness_columns is not True:
            raise ValueError('uniqueness_columns should be a List (sequence) or True')
        if isinstance(uniqueness_columns, str):
            uniqueness_columns = [uniqueness_columns]
        self._artifacts_manager.register_artifact(
            name=name, artifact=artifact, metadata=metadata, uniqueness_columns=uniqueness_columns)

    def unregister_artifact(self, name):
        # type: (str) -> None
        """
        Unregister (remove) a registered artifact. This removes the artifact from the watch list that Trains uses
        to synchronize artifacts with the **Trains Server** (backend).

        .. important::
           - Calling this method does not remove the artifact from a Task. It only stops Trains from
             monitoring the artifact.
           - When this method is called, Trains immediately takes the last snapshot of the artifact.
        """
        self._artifacts_manager.unregister_artifact(name=name)

    def get_registered_artifacts(self):
        # type: () -> Dict[str, Artifact]
        """
        Get a dictionary containing the Task's registered (dynamically synchronized) artifacts (name, artifact object).

        .. note::
           After calling ``get_registered_artifacts``, you can still modify the registered artifacts.

        :return: The registered (dynamically synchronized) artifacts.
        """
        return self._artifacts_manager.registered_artifacts

    def upload_artifact(
        self,
        name,  # type: str
        artifact_object,  # type: Union[str, Mapping, pandas.DataFrame, numpy.ndarray, Image.Image]
        metadata=None,  # type: Optional[Mapping]
        delete_after_upload=False  # type: bool
    ):
        # type: (...) -> bool
        """
        Upload (add) a static artifact to a Task object. The artifact is uploaded in the background.

        The currently supported upload (static) artifact types include:

        - string / pathlib2.Path - A path to artifact file. If a wildcard or a folder is specified, then Trains
          creates and uploads a ZIP file.
        - dict - Trains stores a dictionary as ``.json`` file and uploads it.
        - pandas.DataFrame - Trains stores a pandas.DataFrame as ``.csv.gz`` (compressed CSV) file and uploads it.
        - numpy.ndarray - Trains stores a numpy.ndarray as ``.npz`` file and uploads it.
        - PIL.Image - Trains stores a PIL.Image as ``.png`` file and uploads it.

        :param str name: The artifact name.

            .. warning::
               If an artifact with the same name was previously uploaded, then it is overwritten.

        :param object artifact_object:  The artifact object.
        :param dict metadata: A dictionary of key-value pairs for any metadata. This dictionary appears with the
            experiment in the **Trains Web-App (UI)**, **ARTIFACTS** tab.
        :param bool delete_after_upload: After the upload, delete the local copy of the artifact?

            - ``True`` - Delete the local copy of the artifact.
            - ``False`` - Do not delete. (Default)

        :return: The status of the upload.

        - ``True`` - Upload succeeded.
        - ``False`` - Upload failed.

        :raise: If the artifact object type is not supported, raise a ``ValueError``.
        """
        return self._artifacts_manager.upload_artifact(name=name, artifact_object=artifact_object,
                                                       metadata=metadata, delete_after_upload=delete_after_upload)

    def get_models(self):
        # type: () -> Dict[str, Sequence[Model]]
        """
        Return a dictionary with {'input': [], 'output': []} loaded/stored models of the current Task
        Input models are files loaded in the task, either manually or automatically logged
        Output models are files stored in the task, either manually or automatically logged
        Automatically logged frameworks are for example: TensorFlow, Keras, PyTorch, ScikitLearn(joblib) etc.

        :return dict: dict with keys input/output, each is list of Model objects.
            Example: {'input': [trains.Model()], 'output': [trains.Model()]}
        """
        task_models = {'input': self._get_models(model_type='input'),
                       'output': self._get_models(model_type='output')}
        return task_models

    def is_current_task(self):
        # type: () -> bool
        """
        .. deprecated:: 0.13.0
           This method is deprecated. Use :meth:`Task.is_main_task` instead.

        Is this Task object the main execution Task (initially returned by :meth:`Task.init`)?

        :return: Is this Task object the main execution Task?

            - ``True`` - Is the main execution Task.
            - ``False`` - Is not the main execution Task.
        """
        return self.is_main_task()

    def is_main_task(self):
        # type: () -> bool
        """
        Is this Task object the main execution Task (initially returned by :meth:`Task.init`)?

        .. note::
           If :meth:`Task.init` was never called, this method will *not* create
           it, making this test more efficient than:

           .. code-block:: py

              Task.init() == task

        :return: Is this Task object the main execution Task?

            - ``True`` - Is the main execution Task.
            - ``False`` - Is not the main execution Task.
        """
        return self is self.__main_task

    def set_model_config(self, config_text=None, config_dict=None):
        # type: (Optional[str], Optional[Mapping]) -> None
        """
        .. deprecated:: 0.14.1
            Use :meth:`Task.connect_configuration` instead
        """
        self._set_model_config(config_text=config_text, config_dict=config_dict)

    def get_model_config_text(self):
        # type: () -> str
        """
        .. deprecated:: 0.14.1
            Use :meth:`Task.connect_configuration` instead
        """
        return self._get_model_config_text()

    def get_model_config_dict(self):
        # type: () -> Dict
        """
        .. deprecated:: 0.14.1
            Use :meth:`Task.connect_configuration` instead
        """
        return self._get_model_config_dict()

    def set_model_label_enumeration(self, enumeration=None):
        # type: (Optional[Mapping[str, int]]) -> ()
        """
        Set the label enumeration for the Task object before creating an output model.
        Later, when creating an output model, the model will inherit these properties.

        :param dict enumeration: A label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    'background': 0,
                    'person': 1
               }
        """
        super(Task, self).set_model_label_enumeration(enumeration=enumeration)

    def get_last_iteration(self):
        # type: () -> int
        """
        Get the last reported iteration, which is the last iteration for which the Task reported a metric.

        .. note::
           The maximum reported iteration is not in the local cache. This method
           sends a request to the **Trains Server** (backend).

        :return: The last reported iteration number.
        """
        self._reload_last_iteration()
        return max(self.data.last_iteration, self._reporter.max_iteration if self._reporter else 0)

    def set_last_iteration(self, last_iteration):
        # type: (int) -> None
        """
        Forcefully set the last reported iteration, which is the last iteration for which the Task reported a metric.

        :param last_iteration: The last reported iteration number.
        :type last_iteration: int
        """
        self.data.last_iteration = int(last_iteration)
        self._edit(last_iteration=self.data.last_iteration)

    def set_initial_iteration(self, offset=0):
        # type: (int) -> int
        """
        Set initial iteration, instead of zero. Useful when continuing training from previous checkpoints

        :param int offset: Initial iteration (at starting point)
        :return: newly set initial offset
        """
        return super(Task, self).set_initial_iteration(offset=offset)

    def get_initial_iteration(self):
        # type: () -> int
        """
        Return the initial iteration offset, default is 0
        Useful when continuing training from previous checkpoints

        :return int: initial iteration offset
        """
        return super(Task, self).get_initial_iteration()

    def get_last_scalar_metrics(self):
        # type: () -> Dict[str, Dict[str, Dict[str, float]]]
        """
        Get the last scalar metrics which the Task reported. This is a nested dictionary, ordered by title and series.

        For example:

        .. code-block:: javascript

           {
            'title': {
                'series': {
                    'last': 0.5,
                    'min': 0.1,
                    'max': 0.9
                    }
                }
            }

        :return: The last scalar metrics.
        """
        self.reload()
        metrics = self.data.last_metrics
        scalar_metrics = dict()
        for i in metrics.values():
            for j in i.values():
                scalar_metrics.setdefault(j['metric'], {}).setdefault(
                    j['variant'], {'last': j['value'], 'min': j['min_value'], 'max': j['max_value']})
        return scalar_metrics

    def get_parameters_as_dict(self):
        # type: () -> Dict
        """
        Get the Task parameters as a raw nested dictionary.

        .. note::
           The values are not parsed. They are returned as is.
        """
        return naive_nested_from_flat_dictionary(self.get_parameters())

    def set_parameters_as_dict(self, dictionary):
        # type: (Dict) -> None
        """
        Set the parameters for the Task object from a dictionary. The dictionary can be nested.
        This does not link the dictionary to the Task object. It does a one-time update. This
        is the same behavior as the :meth:`Task.connect` method.
        """
        self._arguments.copy_from_dict(flatten_dictionary(dictionary))

    @classmethod
    def set_credentials(cls, api_host=None, web_host=None, files_host=None, key=None, secret=None, host=None):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]) -> ()
        """
        Set new default **Trains Server** (backend) host and credentials.

        These credentials will be overridden by either OS environment variables, or the Trains configuration
        file, ``trains.conf``.

        .. warning::
           Credentials must be set before initializing a Task object.

        For example, to set credentials for a remote computer:

        .. code-block:: py

           Task.set_credentials(api_host='http://localhost:8008', web_host='http://localhost:8080',
            files_host='http://localhost:8081',  key='optional_credentials',  secret='optional_credentials')
            task = Task.init('project name', 'experiment name')

        :param str api_host: The API server url. For example, ``host='http://localhost:8008'``
        :param str web_host: The Web server url. For example, ``host='http://localhost:8080'``
        :param str files_host: The file server url. For example, ``host='http://localhost:8081'``
        :param str key: The user key (in the key/secret pair). For example, ``key='thisisakey123'``
        :param str secret: The user secret (in the key/secret pair). For example, ``secret='thisisseceret123'``
        :param str host: The host URL (overrides api_host). For example, ``host='http://localhost:8008'``
        """
        if api_host:
            Session.default_host = api_host
        if web_host:
            Session.default_web = web_host
        if files_host:
            Session.default_files = files_host
        if key:
            Session.default_key = key
            if not running_remotely():
                ENV_ACCESS_KEY.set(key)
        if secret:
            Session.default_secret = secret
            if not running_remotely():
                ENV_SECRET_KEY.set(secret)
        if host:
            Session.default_host = host
            Session.default_web = web_host or ''
            Session.default_files = files_host or ''

    def _set_model_config(self, config_text=None, config_dict=None):
        # type: (Optional[str], Optional[Mapping]) -> None
        """
        Set Task model configuration text/dict

        :param config_text: model configuration (unconstrained text string). usually the content
            of a configuration file. If `config_text` is not None, `config_dict` must not be provided.
        :param config_dict: model configuration parameters dictionary.
            If `config_dict` is not None, `config_text` must not be provided.
        """
        # noinspection PyProtectedMember
        design = OutputModel._resolve_config(config_text=config_text, config_dict=config_dict)
        super(Task, self)._set_model_design(design=design)

    def _get_model_config_text(self):
        # type: () -> str
        """
        Get Task model configuration text (before creating an output model)
        When an output model is created it will inherit these properties

        :return: model config_text (unconstrained text string)
        """
        return super(Task, self).get_model_design()

    def _get_model_config_dict(self):
        # type: () -> Dict
        """
        Get Task model configuration dictionary (before creating an output model)
        When an output model is created it will inherit these properties

        :return: config_dict: model configuration parameters dictionary
        """
        config_text = self._get_model_config_text()
        # noinspection PyProtectedMember
        return OutputModel._text_to_config_dict(config_text)

    @classmethod
    def _reset_current_task_obj(cls):
        if not cls.__main_task:
            return
        task = cls.__main_task
        cls.__main_task = None
        if task._dev_worker:
            task._dev_worker.unregister()
            task._dev_worker = None

    @classmethod
    def _create_dev_task(
        cls, default_project_name, default_task_name, default_task_type, reuse_last_task_id, detect_repo=True
    ):
        if not default_project_name or not default_task_name:
            # get project name and task name from repository name and entry_point
            result, _ = ScriptInfo.get(create_requirements=False, check_uncommitted=False)
            if not default_project_name:
                # noinspection PyBroadException
                try:
                    parts = result.script['repository'].split('/')
                    default_project_name = (parts[-1] or parts[-2]).replace('.git', '') or 'Untitled'
                except Exception:
                    default_project_name = 'Untitled'
            if not default_task_name:
                # noinspection PyBroadException
                try:
                    default_task_name = os.path.splitext(os.path.basename(result.script['entry_point']))[0]
                except Exception:
                    pass

        # if we force no task reuse from os environment
        if DEV_TASK_NO_REUSE.get() or not reuse_last_task_id:
            default_task = None
        else:
            # if we have a previous session to use, get the task id from it
            default_task = cls.__get_last_used_task_id(
                default_project_name,
                default_task_name,
                default_task_type.value,
            )

        closed_old_task = False
        default_task_id = None
        task = None
        in_dev_mode = not running_remotely()

        if in_dev_mode:
            if isinstance(reuse_last_task_id, str) and reuse_last_task_id:
                default_task_id = reuse_last_task_id
            elif not reuse_last_task_id or not cls.__task_is_relevant(default_task):
                default_task_id = None
            else:
                default_task_id = default_task.get('id') if default_task else None

            if default_task_id:
                try:
                    task = cls(
                        private=cls.__create_protection,
                        task_id=default_task_id,
                        log_to_backend=True,
                    )
                    task_tags = task.data.system_tags if hasattr(task.data, 'system_tags') else task.data.tags
                    task_artifacts = task.data.execution.artifacts \
                        if hasattr(task.data.execution, 'artifacts') else None
                    if ((str(task._status) in (str(tasks.TaskStatusEnum.published), str(tasks.TaskStatusEnum.closed)))
                            or task.output_model_id or (ARCHIVED_TAG in task_tags)
                            or (cls._development_tag not in task_tags)
                            or task_artifacts):
                        # If the task is published or closed, we shouldn't reset it so we can't use it in dev mode
                        # If the task is archived, or already has an output model,
                        #  we shouldn't use it in development mode either
                        default_task_id = None
                        task = None
                    else:
                        with task._edit_lock:
                            # from now on, there is no need to reload, we just clear stuff,
                            # this flag will be cleared off once we actually refresh at the end of the function
                            task._reload_skip_flag = True
                            # reset the task, so we can update it
                            task.reset(set_started_on_success=False, force=False)
                            # clear the heaviest stuff first
                            task._clear_task(
                                system_tags=[cls._development_tag],
                                comment=make_message('Auto-generated at %(time)s by %(user)s@%(host)s'))

                except (Exception, ValueError):
                    # we failed reusing task, create a new one
                    default_task_id = None

        # create a new task
        if not default_task_id:
            task = cls(
                private=cls.__create_protection,
                project_name=default_project_name,
                task_name=default_task_name,
                task_type=default_task_type,
                log_to_backend=True,
            )
            # no need to reload yet, we clear this before the end of the function
            task._reload_skip_flag = True

        if in_dev_mode:
            # update this session, for later use
            cls.__update_last_used_task_id(default_project_name, default_task_name, default_task_type.value, task.id)
            # set default docker image from env.
            task._set_default_docker_image()

        # mark the task as started
        task.started()
        # reload, making sure we are synced
        task._reload_skip_flag = False
        task.reload()

        # force update of base logger to this current task (this is the main logger task)
        task._setup_log(replace_existing=True)
        logger = task.get_logger()
        if closed_old_task:
            logger.report_text('TRAINS Task: Closing old development task id={}'.format(default_task.get('id')))
        # print warning, reusing/creating a task
        if default_task_id:
            logger.report_text('TRAINS Task: overwriting (reusing) task id=%s' % task.id)
        else:
            logger.report_text('TRAINS Task: created new task id=%s' % task.id)

        # update current repository and put warning into logs
        if detect_repo:
            # noinspection PyBroadException
            try:
                import traceback
                stack = traceback.extract_stack(limit=10)
                # NOTICE WE ARE ALWAYS 3 down from caller in stack!
                for i in range(len(stack)-1, 0, -1):
                    # look for the Task.init call, then the one above it is the callee module
                    if stack[i].name == 'init':
                        task._calling_filename = os.path.abspath(stack[i-1].filename)
                        break
            except Exception:
                pass
            if in_dev_mode and cls.__detect_repo_async:
                task._detect_repo_async_thread = threading.Thread(target=task._update_repository)
                task._detect_repo_async_thread.daemon = True
                task._detect_repo_async_thread.start()
            else:
                task._update_repository()

        # make sure we see something in the UI
        thread = threading.Thread(target=LoggerRoot.flush)
        thread.daemon = True
        thread.start()

        return task

    def _get_logger(self, flush_period=NotSet):
        # type: (Optional[float]) -> Logger
        """
        get a logger object for reporting based on the task

        :param flush_period: The period of the logger flush.
            If None of any other False value, will not flush periodically.
            If a logger was created before, this will be the new period and
            the old one will be discarded.

        :return: Logger object
        """

        if not self._logger:
            # do not recreate logger after task was closed/quit
            if self._at_exit_called:
                raise ValueError("Cannot use Task Logger after task was closed")
            # force update of base logger to this current task (this is the main logger task)
            self._setup_log(replace_existing=self.is_main_task())
            # Get a logger object
            self._logger = Logger(private_task=self)
            # make sure we set our reported to async mode
            # we make sure we flush it in self._at_exit
            self.reporter.async_enable = True
            # if we just created the logger, set default flush period
            if not flush_period or flush_period is self.NotSet:
                flush_period = DevWorker.report_period

        if isinstance(flush_period, (int, float)):
            flush_period = int(abs(flush_period))

        if flush_period is None or isinstance(flush_period, int):
            self._logger.set_flush_period(flush_period)

        return self._logger

    def _connect_output_model(self, model):
        assert isinstance(model, OutputModel)
        model.connect(self)
        return model

    def _save_output_model(self, model):
        """
        Save a reference to the connected output model.

        :param model: The connected output model
        """
        self._connected_output_model = model

    def _reconnect_output_model(self):
        """
        If there is a saved connected output model, connect it again.

        This is needed if the input model is connected after the output model
        is connected, an then we will have to get the model design from the
        input model by reconnecting.
        """
        if self._connected_output_model:
            self.connect(self._connected_output_model)

    def _connect_input_model(self, model):
        assert isinstance(model, InputModel)
        # we only allow for an input model to be connected once
        # at least until we support multiple input models
        # notice that we do not check the task's input model because we allow task reuse and overwrite
        # add into comment that we are using this model
        comment = self.comment or ''
        if not comment.endswith('\n'):
            comment += '\n'
        comment += 'Using model id: {}'.format(model.id)
        self.set_comment(comment)
        if self._last_input_model_id and self._last_input_model_id != model.id:
            self.log.info('Task connect, second input model is not supported, adding into comment section')
            return
        self._last_input_model_id = model.id
        model.connect(self)
        return model

    def _try_set_connected_parameter_type(self, option):
        # """ Raise an error if current value is not None and not equal to the provided option value """
        # value = self._connected_parameter_type
        # if not value or value == option:
        #     self._connected_parameter_type = option
        #     return option
        #
        # def title(option):
        #     return " ".join(map(str.capitalize, option.split("_")))
        #
        # raise ValueError(
        #     "Task already connected to {}. "
        #     "Task can be connected to only one the following argument options: {}".format(
        #         title(value),
        #         ' / '.join(map(title, self._ConnectedParametersType._options())))
        # )

        # added support for multiple type connections through _Arguments
        return option

    def _connect_argparse(self, parser, args=None, namespace=None, parsed_args=None):
        # do not allow argparser to connect to jupyter notebook
        # noinspection PyBroadException
        try:
            if 'IPython' in sys.modules:
                # noinspection PyPackageRequirements
                from IPython import get_ipython
                ip = get_ipython()
                if ip is not None and 'IPKernelApp' in ip.config:
                    return parser
        except Exception:
            pass

        self._try_set_connected_parameter_type(self._ConnectedParametersType.argparse)

        if self.is_main_task():
            argparser_update_currenttask(self)

        if (parser is None or parsed_args is None) and argparser_parseargs_called():
            _parser, _parsed_args = get_argparser_last_args()
            if parser is None:
                parser = _parser
            if parsed_args is None and parser == _parser:
                parsed_args = _parsed_args

        if running_remotely() and self.is_main_task():
            self._arguments.copy_to_parser(parser, parsed_args)
        else:
            self._arguments.copy_defaults_from_argparse(
                parser, args=args, namespace=namespace, parsed_args=parsed_args)
        return parser

    def _connect_dictionary(self, dictionary):
        def _update_args_dict(task, config_dict):
            # noinspection PyProtectedMember
            task._arguments.copy_from_dict(flatten_dictionary(config_dict))

        def _refresh_args_dict(task, config_dict):
            # reread from task including newly added keys
            # noinspection PyProtectedMember
            a_flat_dict = task._arguments.copy_to_dict(flatten_dictionary(config_dict))
            # noinspection PyProtectedMember
            nested_dict = config_dict._to_dict()
            config_dict.clear()
            config_dict.update(nested_from_flat_dictionary(nested_dict, a_flat_dict))

        self._try_set_connected_parameter_type(self._ConnectedParametersType.dictionary)

        if not running_remotely() or not self.is_main_task():
            self._arguments.copy_from_dict(flatten_dictionary(dictionary))
            dictionary = ProxyDictPostWrite(self, _update_args_dict, **dictionary)
        else:
            flat_dict = flatten_dictionary(dictionary)
            flat_dict = self._arguments.copy_to_dict(flat_dict)
            dictionary = nested_from_flat_dictionary(dictionary, flat_dict)
            dictionary = ProxyDictPostWrite(self, _refresh_args_dict, **dictionary)

        return dictionary

    def _connect_task_parameters(self, attr_class):
        self._try_set_connected_parameter_type(self._ConnectedParametersType.task_parameters)

        if running_remotely() and self.is_main_task():
            attr_class.update_from_dict(self.get_parameters())
        else:
            self.set_parameters(attr_class.to_dict())
        return attr_class

    def _validate(self, check_output_dest_credentials=False):
        if running_remotely():
            super(Task, self)._validate(check_output_dest_credentials=False)

    def _output_model_updated(self):
        """ Called when a connected output model is updated """
        if running_remotely() or not self.is_main_task():
            return

        # Make sure we know we've started, just in case we didn't so far
        self._dev_mode_task_start(model_updated=True)

    def _dev_mode_task_start(self, model_updated=False):
        """ Called when we suspect the task has started running """
        self._dev_mode_setup_worker(model_updated=model_updated)

    def _dev_mode_stop_task(self, stop_reason):
        # make sure we do not get called (by a daemon thread) after at_exit
        if self._at_exit_called:
            return

        self.log.warning(
            "### TASK STOPPED - USER ABORTED - {} ###".format(
                stop_reason.upper().replace('_', ' ')
            )
        )
        self.flush(wait_for_uploads=True)
        self.stopped()

        if self._dev_worker:
            self._dev_worker.unregister()

        # NOTICE! This will end the entire execution tree!
        if self.__exit_hook:
            self.__exit_hook.remote_user_aborted = True
        self._kill_all_child_processes(send_kill=False)
        time.sleep(2.0)
        self._kill_all_child_processes(send_kill=True)
        # noinspection PyProtectedMember
        os._exit(1)

    @staticmethod
    def _kill_all_child_processes(send_kill=False):
        # get current process if pid not provided
        pid = os.getpid()
        try:
            parent = psutil.Process(pid)
        except psutil.Error:
            # could not find parent process id
            return
        for child in parent.children(recursive=True):
            if send_kill:
                child.kill()
            else:
                child.terminate()
        # kill ourselves
        if send_kill:
            parent.kill()
        else:
            parent.terminate()

    def _dev_mode_setup_worker(self, model_updated=False):
        if running_remotely() or not self.is_main_task() or self._at_exit_called:
            return

        if self._dev_worker:
            return self._dev_worker

        self._dev_worker = DevWorker()
        self._dev_worker.register(self)

        logger = self.get_logger()
        flush_period = logger.get_flush_period()
        if not flush_period or flush_period > self._dev_worker.report_period:
            logger.set_flush_period(self._dev_worker.report_period)

    def _wait_for_repo_detection(self, timeout=None):
        # wait for detection repo sync
        if not self._detect_repo_async_thread:
            return
        with self._repo_detect_lock:
            if not self._detect_repo_async_thread:
                return
            # noinspection PyBroadException
            try:
                if self._detect_repo_async_thread.is_alive():
                    # if negative timeout, just kill the thread:
                    if timeout is not None and timeout < 0:
                        from .utilities.lowlevel.threads import kill_thread
                        kill_thread(self._detect_repo_async_thread)
                    else:
                        self.log.info('Waiting for repository detection and full package requirement analysis')
                        self._detect_repo_async_thread.join(timeout=timeout)
                        # because join has no return value
                        if self._detect_repo_async_thread.is_alive():
                            self.log.info('Repository and package analysis timed out ({} sec), '
                                          'giving up'.format(timeout))
                            # done waiting, kill the thread
                            from .utilities.lowlevel.threads import kill_thread
                            kill_thread(self._detect_repo_async_thread)
                        else:
                            self.log.info('Finished repository detection and package analysis')
                self._detect_repo_async_thread = None
            except Exception:
                pass

    def _summary_artifacts(self):
        # signal artifacts upload, and stop daemon
        self._artifacts_manager.stop(wait=True)
        # print artifacts summary (if not empty)
        if self._artifacts_manager.summary:
            self.get_logger().report_text(self._artifacts_manager.summary)

    def _at_exit(self):
        # protect sub-process at_exit (should never happen)
        if self._at_exit_called:
            return
        # shutdown will clear the main, so we have to store it before.
        # is_main = self.is_main_task()
        self.__shutdown()
        # In rare cases we might need to forcefully shutdown the process, currently we should avoid it.
        # if is_main:
        #     # we have to forcefully shutdown if we have forked processes, sometimes they will get stuck
        #     os._exit(self.__exit_hook.exit_code if self.__exit_hook and self.__exit_hook.exit_code else 0)

    def __shutdown(self):
        """
        Will happen automatically once we exit code, i.e. atexit
        :return:
        """
        # protect sub-process at_exit
        if self._at_exit_called:
            return

        is_sub_process = self.__is_subprocess()

        # noinspection PyBroadException
        try:
            # from here do not get into watch dog
            self._at_exit_called = True
            wait_for_uploads = True
            # first thing mark task as stopped, so we will not end up with "running" on lost tasks
            # if we are running remotely, the daemon will take care of it
            task_status = None
            wait_for_std_log = True
            if not running_remotely() and self.is_main_task() and not is_sub_process:
                # check if we crashed, ot the signal is not interrupt (manual break)
                task_status = ('stopped', )
                if self.__exit_hook:
                    is_exception = self.__exit_hook.exception
                    # check if we are running inside a debugger
                    if not is_exception and sys.modules.get('pydevd'):
                        # noinspection PyBroadException
                        try:
                            is_exception = sys.last_type
                        except Exception:
                            pass

                    if (is_exception and not isinstance(self.__exit_hook.exception, KeyboardInterrupt)) \
                            or (not self.__exit_hook.remote_user_aborted and self.__exit_hook.signal not in (None, 2)):
                        task_status = ('failed', 'Exception')
                        wait_for_uploads = False
                    else:
                        wait_for_uploads = (self.__exit_hook.remote_user_aborted or self.__exit_hook.signal is None)
                        if not self.__exit_hook.remote_user_aborted and self.__exit_hook.signal is None and \
                                not is_exception:
                            task_status = ('completed', )
                        else:
                            task_status = ('stopped', )
                            # user aborted. do not bother flushing the stdout logs
                            wait_for_std_log = self.__exit_hook.signal is not None

            # wait for repository detection (if we didn't crash)
            if wait_for_uploads and self._logger:
                # we should print summary here
                self._summary_artifacts()
                # make sure that if we crashed the thread we are not waiting forever
                if not is_sub_process:
                    self._wait_for_repo_detection(timeout=10.)

            # kill the repo thread (negative timeout, do not wait), if it hasn't finished yet.
            self._wait_for_repo_detection(timeout=-1)

            # wait for uploads
            print_done_waiting = False
            if wait_for_uploads and (BackendModel.get_num_results() > 0 or
                                     (self._reporter and self.reporter.get_num_results() > 0)):
                self.log.info('Waiting to finish uploads')
                print_done_waiting = True
            # from here, do not send log in background thread
            if wait_for_uploads:
                self.flush(wait_for_uploads=True)
                # wait until the reporter flush everything
                if self._reporter:
                    self.reporter.stop()
                    if self.is_main_task():
                        # notice: this will close the reporting for all the Tasks in the system
                        Metrics.close_async_threads()
                        # notice: this will close the jupyter monitoring
                        ScriptInfo.close()
                if self.is_main_task():
                    # noinspection PyBroadException
                    try:
                        from .storage.helper import StorageHelper
                        StorageHelper.close_async_threads()
                    except Exception:
                        pass

                if print_done_waiting:
                    self.log.info('Finished uploading')
            elif self._logger:
                # noinspection PyProtectedMember
                self._logger._flush_stdout_handler()

            # from here, do not check worker status
            if self._dev_worker:
                self._dev_worker.unregister()
                self._dev_worker = None

            # stop resource monitoring
            if self._resource_monitor:
                self._resource_monitor.stop()
                self._resource_monitor = None

            if not is_sub_process:
                # change task status
                if not task_status:
                    pass
                elif task_status[0] == 'failed':
                    self.mark_failed(status_reason=task_status[1])
                elif task_status[0] == 'completed':
                    self.completed()
                elif task_status[0] == 'stopped':
                    self.stopped()

            if self._logger:
                self._logger.set_flush_period(None)
                # noinspection PyProtectedMember
                self._logger._close_stdout_handler(wait=wait_for_uploads or wait_for_std_log)

            # this is so in theory we can close a main task and start a new one
            if self.is_main_task():
                Task.__main_task = None
        except Exception:
            # make sure we do not interrupt the exit process
            pass
        # delete locking object (lock file)
        if self._edit_lock:
            # noinspection PyBroadException
            try:
                del self.__edit_lock
            except Exception:
                pass
            self._edit_lock = None

    @classmethod
    def __register_at_exit(cls, exit_callback, only_remove_signal_and_exception_hooks=False):
        class ExitHooks(object):
            _orig_exit = None
            _orig_exc_handler = None
            remote_user_aborted = False

            def __init__(self, callback):
                self.exit_code = None
                self.exception = None
                self.signal = None
                self._exit_callback = callback
                self._org_handlers = {}
                self._signal_recursion_protection_flag = False
                self._except_recursion_protection_flag = False

            def update_callback(self, callback):
                if self._exit_callback and not six.PY2:
                    # noinspection PyBroadException
                    try:
                        atexit.unregister(self._exit_callback)
                    except Exception:
                        pass
                self._exit_callback = callback
                if callback:
                    self.hook()
                else:
                    # un register int hook
                    if self._orig_exc_handler:
                        sys.excepthook = self._orig_exc_handler
                        self._orig_exc_handler = None
                    for h in self._org_handlers:
                        # noinspection PyBroadException
                        try:
                            signal.signal(h, self._org_handlers[h])
                        except Exception:
                            pass
                    self._org_handlers = {}

            def hook(self):
                if self._orig_exit is None:
                    self._orig_exit = sys.exit
                    sys.exit = self.exit

                if self._orig_exc_handler is None:
                    self._orig_exc_handler = sys.excepthook
                    sys.excepthook = self.exc_handler

                if self._exit_callback:
                    atexit.register(self._exit_callback)

                # TODO: check if sub-process hooks are safe enough, for the time being allow it
                if not self._org_handlers:  # ## and not Task._Task__is_subprocess():
                    if sys.platform == 'win32':
                        catch_signals = [signal.SIGINT, signal.SIGTERM, signal.SIGSEGV, signal.SIGABRT,
                                         signal.SIGILL, signal.SIGFPE]
                    else:
                        catch_signals = [signal.SIGINT, signal.SIGTERM, signal.SIGSEGV, signal.SIGABRT,
                                         signal.SIGILL, signal.SIGFPE, signal.SIGQUIT]
                    for c in catch_signals:
                        # noinspection PyBroadException
                        try:
                            self._org_handlers[c] = signal.getsignal(c)
                            signal.signal(c, self.signal_handler)
                        except Exception:
                            pass

            def exit(self, code=0):
                self.exit_code = code
                self._orig_exit(code)

            def exc_handler(self, exctype, value, traceback, *args, **kwargs):
                if self._except_recursion_protection_flag:
                    # noinspection PyArgumentList
                    return sys.__excepthook__(exctype, value, traceback, *args, **kwargs)

                self._except_recursion_protection_flag = True
                self.exception = value
                if self._orig_exc_handler:
                    # noinspection PyArgumentList
                    ret = self._orig_exc_handler(exctype, value, traceback, *args, **kwargs)
                else:
                    # noinspection PyNoneFunctionAssignment, PyArgumentList
                    ret = sys.__excepthook__(exctype, value, traceback, *args, **kwargs)
                self._except_recursion_protection_flag = False

                return ret

            def signal_handler(self, sig, frame):
                if self._signal_recursion_protection_flag:
                    # call original
                    org_handler = self._org_handlers.get(sig)
                    if isinstance(org_handler, Callable):
                        org_handler = org_handler(sig, frame)
                    return org_handler

                self._signal_recursion_protection_flag = True
                # call exit callback
                self.signal = sig
                if self._exit_callback:
                    # noinspection PyBroadException
                    try:
                        self._exit_callback()
                    except Exception:
                        pass
                # call original signal handler
                org_handler = self._org_handlers.get(sig)
                if isinstance(org_handler, Callable):
                    # noinspection PyBroadException
                    try:
                        org_handler = org_handler(sig, frame)
                    except Exception:
                        org_handler = signal.SIG_DFL
                # remove stdout logger, just in case
                # noinspection PyBroadException
                try:
                    # noinspection PyProtectedMember
                    Logger._remove_std_logger()
                except Exception:
                    pass
                self._signal_recursion_protection_flag = False
                # return handler result
                return org_handler

        # we only remove the signals since this will hang subprocesses
        if only_remove_signal_and_exception_hooks:
            if not cls.__exit_hook:
                return
            if cls.__exit_hook._orig_exc_handler:
                sys.excepthook = cls.__exit_hook._orig_exc_handler
                cls.__exit_hook._orig_exc_handler = None
            for s in cls.__exit_hook._org_handlers:
                # noinspection PyBroadException
                try:
                    signal.signal(s, cls.__exit_hook._org_handlers[s])
                except Exception:
                    pass
            cls.__exit_hook._org_handlers = {}
            return

        if cls.__exit_hook is None:
            # noinspection PyBroadException
            try:
                cls.__exit_hook = ExitHooks(exit_callback)
                cls.__exit_hook.hook()
            except Exception:
                cls.__exit_hook = None
        else:
            cls.__exit_hook.update_callback(exit_callback)

    @classmethod
    def __get_task(cls, task_id=None, project_name=None, task_name=None):
        if task_id:
            return cls(private=cls.__create_protection, task_id=task_id, log_to_backend=False)

        if project_name:
            res = cls._send(
                cls._get_default_session(),
                projects.GetAllRequest(
                    name=exact_match_regex(project_name)
                )
            )
            project = get_single_result(entity='project', query=project_name, results=res.response.projects)
        else:
            project = None

        system_tags = 'system_tags' if hasattr(tasks.Task, 'system_tags') else 'tags'
        res = cls._send(
            cls._get_default_session(),
            tasks.GetAllRequest(
                project=[project.id] if project else None,
                name=exact_match_regex(task_name) if task_name else None,
                only_fields=['id', 'name', 'last_update', system_tags]
            )
        )
        res_tasks = res.response.tasks
        # if we have more than one result, first filter 'archived' results:
        if len(res_tasks) > 1:
            filtered_tasks = [t for t in res_tasks if not getattr(t, system_tags, None) or
                              'archived' not in getattr(t, system_tags, None)]
            if filtered_tasks:
                res_tasks = filtered_tasks

        task = get_single_result(entity='task', query=task_name, results=res_tasks, raise_on_error=False)
        if not task:
            return None

        return cls(
            private=cls.__create_protection,
            task_id=task.id,
            log_to_backend=False,
        )

    @classmethod
    def __get_tasks(cls, task_ids=None, project_name=None, task_name=None, **kwargs):
        if task_ids:
            if isinstance(task_ids, six.string_types):
                task_ids = [task_ids]
            return [cls(private=cls.__create_protection, task_id=task_id, log_to_backend=False)
                    for task_id in task_ids]

        return [cls(private=cls.__create_protection, task_id=task.id, log_to_backend=False)
                for task in cls._query_tasks(project_name=project_name, task_name=task_name, **kwargs)]

    @classmethod
    def _query_tasks(cls, task_ids=None, project_name=None, task_name=None, **kwargs):
        if not task_ids:
            task_ids = None
        elif isinstance(task_ids, six.string_types):
            task_ids = [task_ids]

        if project_name:
            res = cls._send(
                cls._get_default_session(),
                projects.GetAllRequest(
                    name=exact_match_regex(project_name)
                )
            )
            project = get_single_result(entity='project', query=project_name, results=res.response.projects)
        else:
            project = None

        system_tags = 'system_tags' if hasattr(tasks.Task, 'system_tags') else 'tags'
        only_fields = ['id', 'name', 'last_update', system_tags]

        if kwargs and kwargs.get('only_fields'):
            only_fields = list(set(kwargs.pop('only_fields')) | set(only_fields))

        res = cls._send(
            cls._get_default_session(),
            tasks.GetAllRequest(
                id=task_ids,
                project=[project.id] if project else None,
                name=task_name if task_name else None,
                only_fields=only_fields,
                **kwargs
            )
        )

        return res.response.tasks

    @classmethod
    def __get_hash_key(cls, *args):
        def normalize(x):
            return "<{}>".format(x) if x is not None else ""

        return ":".join(map(normalize, args))

    @classmethod
    def __get_last_used_task_id(cls, default_project_name, default_task_name, default_task_type):
        hash_key = cls.__get_hash_key(
            cls._get_api_server(), default_project_name, default_task_name, default_task_type)

        # check if we have a cached task_id we can reuse
        # it must be from within the last 24h and with the same project/name/type
        task_sessions = SessionCache.load_dict(str(cls))

        task_data = task_sessions.get(hash_key)
        if task_data is None:
            return None

        try:
            task_data['type'] = cls.TaskTypes(task_data['type'])
        except (ValueError, KeyError):
            LoggerRoot.get_base_logger().warning(
                "Corrupted session cache entry: {}. "
                "Unsupported task type: {}"
                "Creating a new task.".format(hash_key, task_data['type']),
            )

            return None

        return task_data

    @classmethod
    def __update_last_used_task_id(cls, default_project_name, default_task_name, default_task_type, task_id):
        hash_key = cls.__get_hash_key(
            cls._get_api_server(), default_project_name, default_task_name, default_task_type)

        task_id = str(task_id)
        # update task session cache
        task_sessions = SessionCache.load_dict(str(cls))
        last_task_session = {'time': time.time(), 'project': default_project_name, 'name': default_task_name,
                             'type': default_task_type, 'id': task_id}

        # remove stale sessions
        for k in list(task_sessions.keys()):
            if ((time.time() - task_sessions[k].get('time', 0)) >
                    60 * 60 * cls.__task_id_reuse_time_window_in_hours):
                task_sessions.pop(k)
        # update current session
        task_sessions[hash_key] = last_task_session
        # store
        SessionCache.store_dict(str(cls), task_sessions)

    @classmethod
    def __task_timed_out(cls, task_data):
        return \
            task_data and \
            task_data.get('id') and \
            task_data.get('time') and \
            (time.time() - task_data.get('time')) > (60 * 60 * cls.__task_id_reuse_time_window_in_hours)

    @classmethod
    def __get_task_api_obj(cls, task_id, only_fields=None):
        if not task_id:
            return None

        all_tasks = cls._send(
            cls._get_default_session(),
            tasks.GetAllRequest(id=[task_id], only_fields=only_fields),
        ).response.tasks

        # The task may not exist in environment changes
        if not all_tasks:
            return None

        return all_tasks[0]

    @classmethod
    def __task_is_relevant(cls, task_data):
        """
        Check that a cached task is relevant for reuse.

        A task is relevant for reuse if:
            1. It is not timed out i.e it was last use in the previous 24 hours.
            2. It's name, project and type match the data in the server, so not
               to override user changes made by using the UI.

        :param task_data: A mapping from 'id', 'name', 'project', 'type' keys
            to the task's values, as saved in the cache.

        :return: True if the task is relevant for reuse, False if not.
        """
        if not task_data:
            return False

        if cls.__task_timed_out(task_data):
            return False

        task_id = task_data.get('id')

        if not task_id:
            return False

        task = cls.__get_task_api_obj(task_id, ('id', 'name', 'project', 'type'))

        if task is None:
            return False

        project_name = None
        if task.project:
            project = cls._send(
                cls._get_default_session(),
                projects.GetByIdRequest(project=task.project)
            ).response.project

            if project:
                project_name = project.name

        if task_data.get('type') and \
                task_data.get('type') not in (cls.TaskTypes.training, cls.TaskTypes.testing) and \
                not Session.check_min_api_version(2.8):
            print('WARNING: Changing task type to "{}" : '
                  'trains-server does not support task type "{}", '
                  'please upgrade trains-server.'.format(cls.TaskTypes.training, task_data['type'].value))
            task_data['type'] = cls.TaskTypes.training

        compares = (
            (task.name, 'name'),
            (project_name, 'project'),
            (task.type, 'type'),
        )

        # compare after casting to string to avoid enum instance issues
        # remember we might have replaced the api version by now, so enums are different
        return all(six.text_type(server_data) == six.text_type(task_data.get(task_data_key))
                   for server_data, task_data_key in compares)

    @classmethod
    def __close_timed_out_task(cls, task_data):
        if not task_data:
            return False

        task = cls.__get_task_api_obj(task_data.get('id'), ('id', 'status'))

        if task is None:
            return False

        stopped_statuses = (
            str(tasks.TaskStatusEnum.stopped),
            str(tasks.TaskStatusEnum.published),
            str(tasks.TaskStatusEnum.publishing),
            str(tasks.TaskStatusEnum.closed),
            str(tasks.TaskStatusEnum.failed),
            str(tasks.TaskStatusEnum.completed),
        )

        if str(task.status) not in stopped_statuses:
            cls._send(
                cls._get_default_session(),
                tasks.StoppedRequest(
                    task=task.id,
                    force=True,
                    status_message="Stopped timed out development task"
                ),
            )

            return True
        return False

    def execute_remotely(self, queue_name=None, clone=False, exit_process=True):
        # type: (Optional[str], bool, bool) -> ()
        """
        If task is running locally (i.e. not by 'trains-agent'),
        this call will clone the Task and enqueue it for remote execution.
        Or it will stop the execution of the current task, reset its state, and enqueue it.
        Finally if exit==True it will *exit* this process!

        If task is executed by trains-agent (i.e. running remotely),
        this call is a no-op (i.e. does nothing).

        :param queue_name: Queue name used for enqueueing the task.
            If None, this call will exit the process without enqueuing the task.
        :param clone: If True a cloned copy of the Task will be created (and enqueued) instead of this Task.
        :param exit_process: If True, the function call will leave the calling process at the end
            i.e. If True, exit(0) will be called. If clone==False exit_process must be True.
        """
        # do nothing, we are running remotely
        if running_remotely():
            return

        if not clone and not exit_process:
            raise ValueError(
                "clone==False and exit_process==False is not supported. "
                "Task enqueuing itself must exit the process afterwards.")

        # make sure we analyze the process
        if self.status in (Task.TaskStatusEnum.in_progress, ):
            if clone:
                # wait for repository detection (5 minutes should be reasonable time to detect all packages)
                self.flush(wait_for_uploads=True)
                if self._logger and not self.__is_subprocess():
                    self._wait_for_repo_detection(timeout=300.)
            else:
                # close ourselves (it will make sure the repo is updated)
                self.close()

        # clone / reset Task
        if clone:
            task = Task.clone(self)
        else:
            task = self
            self.reset()

        # enqueue ourselves
        if queue_name:
            Task.enqueue(task, queue_name=queue_name)
            LoggerRoot.get_base_logger().warning(
                'Switching to remote execution, output log page {}'.format(task.get_output_log_web_page()))

        # leave this process.
        if exit_process:
            LoggerRoot.get_base_logger().warning('Terminating local execution process')
            exit(0)

        return
