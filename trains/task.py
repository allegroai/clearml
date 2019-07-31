import atexit
import os
import re
import signal
import sys
import threading
import time
from argparse import ArgumentParser
from collections import OrderedDict, Callable

import psutil
import six
from pathlib2 import Path

from .binding.joblib_bind import PatchedJoblib
from .backend_api.services import tasks, projects
from .backend_api.session.session import Session
from .backend_interface.model import Model as BackendModel
from .backend_interface.task import Task as _Task
from .backend_interface.task.args import _Arguments
from .backend_interface.task.development.worker import DevWorker
from .backend_interface.task.repo import ScriptInfo
from .backend_interface.util import get_single_result, exact_match_regex, make_message
from .config import config, PROC_MASTER_ID_ENV_VAR, DEV_TASK_NO_REUSE
from .config import running_remotely, get_remote_task_id
from .config.cache import SessionCache
from .debugging.log import LoggerRoot
from .errors import UsageError
from .logger import Logger
from .model import InputModel, OutputModel, ARCHIVED_TAG
from .task_parameters import TaskParameters
from .binding.artifacts import Artifacts
from .binding.environ_bind import EnvironmentBind, PatchOsFork
from .binding.absl_bind import PatchAbsl
from .utilities.args import argparser_parseargs_called, get_argparser_last_args, \
    argparser_update_currenttask
from .binding.frameworks.pytorch_bind import PatchPyTorchModelIO
from .binding.frameworks.tensorflow_bind import PatchSummaryToEventTransformer, PatchTensorFlowEager, \
    PatchKerasModelIO, PatchTensorflowModelIO
from .binding.frameworks.xgboost_bind import PatchXGBoostModelIO
from .binding.matplotlib_bind import PatchedMatplotlib
from .utilities.resource_monitor import ResourceMonitor
from .utilities.seed import make_deterministic

NotSet = object()


class Task(_Task):
    """
    Task (experiment) object represents the current running experiments and connects all the different parts into \
    a fully reproducible experiment

    Common usage is calling Task.init() to initialize the main task.
    The main task is development / remote execution mode-aware, and supports connecting various SDK objects
    such as Models etc. In development mode, the main task supports task reuse (see Task.init() for more
    information in development mode features).
    Any subsequent call to Task.init() will return the already-initialized main task
    and will not create a new main task.

    Sub-tasks, meaning tasks which are not the main task and are not development / remote execution mode aware, can be
    created using Task.create(). These tasks do no support task reuse and any call
    to Task.create() will always create a new task.

    You can also query existing tasks in the system by calling Task.get_task().

    **Usage: Task.init(...), Task.create() or Task.get_task(...)**
    """

    TaskTypes = _Task.TaskTypes

    __create_protection = object()
    __main_task = None
    __exit_hook = None
    __forked_proc_main_pid = None
    __task_id_reuse_time_window_in_hours = float(config.get('development.task_reuse_time_window_in_hours', 24.0))
    __store_diff_on_train = config.get('development.store_uncommitted_code_diff_on_train', False)
    __detect_repo_async = config.get('development.vcs_repo_detect_async', False)

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
        **Do not construct Task manually!**

        please use Task.current_task() or Task.get_task(id=, project=, name=)
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
        # register atexit, so that we mark the task as stopped
        self._at_exit_called = False

    @classmethod
    def current_task(cls):
        """
        Return the Current Task object for the main execution task (task context).
        :return: Task() object or None
        """
        return cls.__main_task

    @classmethod
    def init(
            cls,
            project_name=None,
            task_name=None,
            task_type=TaskTypes.training,
            reuse_last_task_id=True,
            output_uri=None,
            auto_connect_arg_parser=True,
            auto_connect_frameworks=True,
            auto_resource_monitoring=True,
    ):
        """
        Return the Task object for the main execution task (task context).

        :param project_name: project to create the task in (if project doesn't exist, it will be created)
        :param task_name: task name to be created (in development mode, not when running remotely)
        :param task_type: task type to be created (in development mode, not when running remotely)
        :param reuse_last_task_id: start with the previously used task id (stored in the data cache folder).
            if False every time we call the function we create a new task with the same name
            Notice! The reused task will be reset. (when running remotely, the usual behaviour applies)
            If reuse_last_task_id is of type string, it will assume this is the task_id to reuse!
            Note: A closed or published task will not be reused, and a new task will be created.
        :param output_uri: Default location for output models (currently support folder/S3/GS/ ).
            notice: sub-folders (task_id) is created in the destination folder for all outputs.
        :param auto_connect_arg_parser: Automatically grab the ArgParser and connect it with the task.
            if set to false, you can manually connect the ArgParser with task.connect(parser)
        :param auto_connect_frameworks: If true automatically patch MatplotLib, Keras callbacks, and TensorBoard/X to
            serialize plots, graphs and model location to trains backend (in addition to original output destination)
        :param auto_resource_monitoring: If true, machine vitals will be sent along side the task scalars,
            Resources graphs will appear under the title ':resource monitor:' in the scalars tab.
        :return: Task() object
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
                        "and requested {field} '{default}' does not match current {field} '{current}'".format(
                            field=field,
                            default=default,
                            current=current,
                        )
                    )

        if cls.__main_task is not None:
            # if this is a subprocess, regardless of what the init was called for,
            # we have to fix the main task hooks and stdout bindings
            if cls.__forked_proc_main_pid != os.getpid() and PROC_MASTER_ID_ENV_VAR.get() != os.getpid():
                # make sure we only do it once  per process
                cls.__forked_proc_main_pid = os.getpid()
                # make sure we do not wait for the repo detect thread
                cls.__main_task._detect_repo_async_thread = None
                # remove the logger from the previous process
                logger = cls.__main_task.get_logger()
                logger.set_flush_period(None)
                # create a new logger (to catch stdout/err)
                cls.__main_task._logger = None
                cls.__main_task._reporter = None
                cls.__main_task.get_logger()
                # unregister signal hooks, they cause subprocess to hang
                cls.__main_task.__register_at_exit(cls.__main_task._at_exit)
                cls.__main_task.__register_at_exit(None, only_remove_signal_and_exception_hooks=True)

            if not running_remotely():
                verify_defaults_match()

            return cls.__main_task

        # check that we are not a child process, in that case do nothing.
        # we should not get here unless this is Windows platform, all others support fork
        if PROC_MASTER_ID_ENV_VAR.get() and PROC_MASTER_ID_ENV_VAR.get() != os.getpid():
            class _TaskStub(object):
                def __call__(self, *args, **kwargs):
                    return self

                def __getattr__(self, attr):
                    return self

                def __setattr__(self, attr, val):
                    pass

            return _TaskStub()
        # set us as master process
        PROC_MASTER_ID_ENV_VAR.set(os.getpid())

        if task_type is None:
            # Backwards compatibility: if called from Task.current_task and task_type
            # was not specified, keep legacy default value of TaskTypes.training
            task_type = cls.TaskTypes.training

        try:
            if not running_remotely():
                task = cls._create_dev_task(
                    project_name,
                    task_name,
                    task_type,
                    reuse_last_task_id,
                )
                if output_uri:
                    task.output_uri = output_uri
            else:
                task = cls(
                    private=cls.__create_protection,
                    task_id=get_remote_task_id(),
                    log_to_backend=False,
                )
        except Exception:
            raise
        else:
            Task.__main_task = task
            # register the main task for at exit hooks (there should only be one)
            task.__register_at_exit(task._at_exit)
            # patch OS forking
            PatchOsFork.patch_fork()
            if auto_connect_frameworks:
                PatchedJoblib.update_current_task(task)
                PatchedMatplotlib.update_current_task(Task.__main_task)
                PatchAbsl.update_current_task(Task.__main_task)
                PatchSummaryToEventTransformer.update_current_task(task)
                PatchTensorFlowEager.update_current_task(task)
                PatchKerasModelIO.update_current_task(task)
                PatchTensorflowModelIO.update_current_task(task)
                PatchPyTorchModelIO.update_current_task(task)
                PatchXGBoostModelIO.update_current_task(task)
            if auto_resource_monitoring:
                task._resource_monitor = ResourceMonitor(task)
                task._resource_monitor.start()

            # make sure all random generators are initialized with new seed
            make_deterministic(task.get_random_seed())

            if auto_connect_arg_parser:
                EnvironmentBind.update_current_task(Task.__main_task)

                # Patch ArgParser to be aware of the current task
                argparser_update_currenttask(Task.__main_task)
                # Check if parse args already called. If so, sync task parameters with parser
                if argparser_parseargs_called():
                    parser, parsed_args = get_argparser_last_args()
                    task._connect_argparse(parser=parser, parsed_args=parsed_args)

        # Make sure we start the logger, it will patch the main logging object and pipe all output
        # if we are running locally and using development mode worker, we will pipe all stdout to logger.
        # The logger will automatically take care of all patching (we just need to make sure to initialize it)
        task.get_logger()

        # Make sure we start the dev worker if required, otherwise it will only be started when we write
        # something to the log.
        task._dev_mode_task_start()

        return task

    @classmethod
    def create(
            cls,
            task_name=None,
            project_name=None,
            task_type=TaskTypes.training,
    ):
        """
        Create a new Task object, regardless of the main execution task (Task.init).

        Notice: This function will always create a new task, whether running in development or remote execution mode.

        :param task_name: task name to be created
        :param project_name: Project to create the task in.
            If project is None, and the main execution task is initialized (Task.init), its project will be used.
            If project is provided but doesn't exist, it will be created.
        :param task_type: Task type to be created. (default: "training")
        Optional Task types are: "training" / "testing" / "dataset_import" / "annotation" / "annotation_manual"
        :return: Task() object
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
    def _reset_current_task_obj(cls):
        if not cls.__main_task:
            return
        task = cls.__main_task
        cls.__main_task = None
        if task._dev_worker:
            task._dev_worker.unregister()
            task._dev_worker = None

    @classmethod
    def _create_dev_task(cls, default_project_name, default_task_name, default_task_type, reuse_last_task_id):
        if not default_project_name or not default_task_name:
            # get project name and task name from repository name and entry_point
            result = ScriptInfo.get(create_requirements=False, check_uncommitted=False)
            if result:
                if not default_project_name:
                    try:
                        default_project_name = re.sub(r"\.git$", "", result.script.get('repository')) or "Untitled"
                    except TypeError:
                        default_project_name = 'Untitled'
                if not default_task_name:
                    try:
                        default_task_name = Path(result.script.get("entry_point")).stem
                    except TypeError:
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
                    if ((task.status in (tasks.TaskStatusEnum.published, tasks.TaskStatusEnum.closed))
                            or (ARCHIVED_TAG in task.data.tags) or task.output_model_id):
                        # If the task is published or closed, we shouldn't reset it so we can't use it in dev mode
                        # If the task is archived, or already has an output model,
                        #  we shouldn't use it in development mode either
                        default_task_id = None
                        task = None
                    else:
                        # reset the task, so we can update it
                        task.reset(set_started_on_success=False, force=False)
                        # set development tags
                        task.set_tags([cls._development_tag])
                        # clear task parameters, they are not cleared by the Task reset
                        task.set_parameters({}, __update=False)
                        # clear the comment, it is not cleared on reset
                        task.set_comment(make_message('Auto-generated at %(time)s by %(user)s@%(host)s'))
                        # clear the input model (and task model design/labels)
                        task.set_input_model(model_id='', update_task_design=False, update_task_labels=False)
                        task.set_model_config(config_text='')
                        task.set_model_label_enumeration({})

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

        if in_dev_mode:
            # update this session, for later use
            cls.__update_last_used_task_id(default_project_name, default_task_name, default_task_type.value, task.id)

        # force update of base logger to this current task (this is the main logger task)
        task._setup_log(replace_existing=True)
        logger = task.get_logger()
        if closed_old_task:
            logger.console('TRAINS Task: Closing old development task id={}'.format(default_task.get('id')))
        # print warning, reusing/creating a task
        if default_task_id:
            logger.console('TRAINS Task: overwriting (reusing) task id=%s' % task.id)
        else:
            logger.console('TRAINS Task: created new task id=%s' % task.id)

        # update current repository and put warning into logs
        if in_dev_mode and cls.__detect_repo_async:
            task._detect_repo_async_thread = threading.Thread(target=task._update_repository)
            task._detect_repo_async_thread.daemon = True
            task._detect_repo_async_thread.start()
        else:
            task._update_repository()

        # show the debug metrics page in the log, it is very convenient
        logger.console(
            'TRAINS results page: {}/projects/{}/experiments/{}/output/log'.format(
                task._get_app_server(),
                task.project if task.project is not None else '*',
                task.id,
            ),
        )
        # make sure everything is in sync
        task.reload()
        # make sure we see something in the UI
        thread = threading.Thread(target=LoggerRoot.flush)
        thread.daemon = True
        thread.start()
        return task

    @staticmethod
    def get_task(task_id=None, project_name=None, task_name=None):
        """
        Returns Task object based on either, task_id (system uuid) or task name

        :param task_id: unique task id string (if exists other parameters are ignored)
        :param project_name: project name (str) the task belogs to
        :param task_name: task name (str) in within the selected project
        :return: Task object
        """
        return Task.__get_task(task_id=task_id, project_name=project_name, task_name=task_name)

    @property
    def output_uri(self):
        return self.storage_uri

    @output_uri.setter
    def output_uri(self, value):
        self.storage_uri = value

    @property
    def artifacts(self):
        """
        dictionary of Task artifacts (name, artifact)
        :return: dict
        """
        return self._artifacts_manager.artifacts

    def set_comment(self, comment):
        """
        Set a comment text to the task.

        In remote, this is a no-op.

        :param comment: The comment of the task
        :type comment: str
        """
        if not running_remotely() or not self.is_main_task():
            self._edit(comment=comment)
            self.reload()

    def add_tags(self, tags):
        """
        Add tags to this task. Old tags are not deleted

        In remote, this is a no-op.

        :param tags: An iterable or space separated string of new tags (string) to add.
        :type tags: str or iterable of str
        """

        if not running_remotely() or not self.is_main_task():
            if isinstance(tags, six.string_types):
                tags = tags.split(" ")

            self.data.tags.extend(tags)
            self._edit(tags=list(set(self.data.tags)))

    def connect(self, mutable):
        """
        Connect an object to a task (see introduction to Task connect design)

        :param mutable: can be any object Task supports integrating with:
            - argparse : for argument passing
            - dict : for argument passing
            - TaskParameters : for argument passing
            - model : for initial model warmup or model update/snapshot uploads
        :return: connect_task() return value if supported
        :raise: raise exception on unsupported objects
        """

        dispatch = OrderedDict((
            (OutputModel, self._connect_output_model),
            (InputModel, self._connect_input_model),
            (ArgumentParser, self._connect_argparse),
            (dict, self._connect_dictionary),
            (TaskParameters, self._connect_task_parameters),
        ))

        for mutable_type, method in dispatch.items():
            if isinstance(mutable, mutable_type):
                return method(mutable)

        raise Exception('Unsupported mutable type %s: no connect function found' % type(mutable).__name__)

    def get_logger(self, flush_period=NotSet):
        """
        get a logger object for reporting based on the task

        :param flush_period: The period of the logger flush.
            If None of any other False value, will not flush periodically.
            If a logger was created before, this will be the new period and
            the old one will be discarded.

        :return: .Logger object
        """
        if not self._logger:
            # force update of base logger to this current task (this is the main logger task)
            self._setup_log(replace_existing=self.is_main_task())
            # Get a logger object
            self._logger = Logger(private_task=self)
            # make sure we set our reported to async mode
            # we make sure we flush it in self._at_exit
            self.reporter.async_enable = True
            # if we just created the logger, set default flush period
            if not flush_period or flush_period is NotSet:
                flush_period = DevWorker.report_period

        if isinstance(flush_period, (int, float)):
            flush_period = int(abs(flush_period))

        if flush_period is None or isinstance(flush_period, int):
            self._logger.set_flush_period(flush_period)

        return self._logger

    def mark_started(self):
        """
        Manually Mark the task as started (will happen automatically)
        """
        # UI won't let us see metrics if we're not started
        self.started()
        self.reload()

    def mark_stopped(self):
        """
        Manually Mark the task as stopped (also used in self._at_exit)
        """
        # flush any outstanding logs
        self.flush(wait_for_uploads=True)
        # mark task as stopped
        self.stopped()

    def flush(self, wait_for_uploads=False):
        """
        flush any outstanding reports or console logs

        :param wait_for_uploads: if True the flush will exit only after all outstanding uploads are completed
        :return: True
        """

        # make sure model upload is done
        if BackendModel.get_num_results() > 0 and wait_for_uploads:
            BackendModel.wait_for_results()

        # flush any outstanding logs
        if self._logger:
            # noinspection PyProtectedMember
            self._logger._flush_stdout_handler()
        self.reporter.flush()
        LoggerRoot.flush()

        return True

    def reset(self, set_started_on_success=False, force=False):
        """
        Reset the task. Task will be reloaded following a successful reset.

        Notice: when running remotely the task will not be reset (as it will clear all logs and metrics)

        :param set_started_on_success: automatically set started if reset was successful
        :param force: force task reset even if running remotely
        """
        if not running_remotely() or not self.is_main_task() or force:
            super(Task, self).reset(set_started_on_success=set_started_on_success)

    def close(self):
        """
        Close the current Task. Enables to manually shutdown the task.
        Should only be called if you are absolutely sure there is no need for the Task.
        """
        self._at_exit()
        self._at_exit_called = False
        # unregister atexit callbacks and signal hooks, if we are the main task
        if self.is_main_task():
            self.__register_at_exit(None)

    def add_artifact(self, name, artifact):
        """
        Add artifact for the current Task, used mostly for Data Audition.
        Currently supported artifacts object types: pandas.DataFrame
        :param name: name of the artifacts. can override previous artifacts if name already exists
        :type name: str
        :param artifact: artifact object, supported artifacts object types: pandas.DataFrame
        :type artifact: pandas.DataFrame
        """
        self._artifacts_manager.add_artifact(name=name, artifact=artifact)

    def is_current_task(self):
        """
        Check if this task is the main task (returned by Task.init())

        NOTE: This call is deprecated. Please use Task.is_main_task()

        If Task.init() was never called, this method will *not* create
        it, making this test cheaper than Task.init() == task
        
        :return: True if this task is the current task
        """
        return self.is_main_task()

    def is_main_task(self):
        """
        Check if this task is the main task (returned by Task.init())

        If Task.init() was never called, this method will *not* create
        it, making this test cheaper than Task.init() == task

        :return: True if this task is the current task
        """
        return self is self.__main_task

    def set_model_config(self, config_text=None, config_dict=None):
        """
        Set Task model configuration  text/dict  (before creating an output model)
        When an output model is created it will inherit these properties

        :param config_text: model configuration (unconstrained text string). usually the content of a configuration file.
            If `config_text` is not None, `config_dict` must not be provided.
        :param config_dict: model configuration parameters dictionary.
            If `config_dict` is not None, `config_text` must not be provided.
        """
        design = OutputModel._resolve_config(config_text=config_text, config_dict=config_dict)
        super(Task, self)._set_model_design(design=design)

    def get_model_config_text(self):
        """
        Get Task model configuration text (before creating an output model)
        When an output model is created it will inherit these properties

        :return model config_text (unconstrained text string). usually the content of a configuration file.
            If `config_text` is not None, `config_dict` must not be provided.
        """
        return super(Task, self).get_model_design()

    def get_model_config_dict(self):
        """
        Get Task model configuration dictionary (before creating an output model)
        When an output model is created it will inherit these properties

        :return model config_text (unconstrained text string). usually the content of a configuration file.
            If `config_text` is not None, `config_dict` must not be provided.
        """
        config_text = self.get_model_config_text()
        return OutputModel._text_to_config_dict(config_text)

    def set_model_label_enumeration(self, enumeration=None):
        """
        Set Task output label enumeration (before creating an output model)
        When an output model is created it will inherit these properties

        :param enumeration: dictionary of string to integer, enumerating the model output to labels
            example: {'background': 0 , 'person': 1}
        """
        super(Task, self).set_model_label_enumeration(enumeration=enumeration)

    def get_last_iteration(self):
        """
        Return the last reported iteration (i.e. the maximum iteration the task reported a metric for)
        Notice, this is not a cached call, it will ask the backend for the answer (no local caching)

        :return integer, last reported iteration number
        """
        self.reload()
        return self.data.last_iteration

    def set_last_iteration(self, last_iteration):
        """
        Forcefully set the last reported iteration
        (i.e. the maximum iteration the task reported a metric for)

        :param last_iteration: last reported iteration number
        :type last_iteration: integer
        """
        self.data.last_iteration = int(last_iteration)
        self._edit(last_iteration=self.data.last_iteration)

    @classmethod
    def set_credentials(cls, host=None, key=None, secret=None):
        """
        Set new default TRAINS-server host and credentials
        These configurations will be overridden by wither OS environment variables or trains.conf configuration file
        Notice: credentials needs to be set prior to Task initialization
        :param host: host url, example: host='http://localhost:8008'
        :type  host: str
        :param key: user key/secret pair, example: key='thisisakey123'
        :type  key: str
        :param secret: user key/secret pair, example: secret='thisisseceret123'
        :type  secret: str
        """
        if host:
            Session.default_host = host
        if key:
            Session.default_key = key
        if secret:
            Session.default_secret = secret

    def _connect_output_model(self, model):
        assert isinstance(model, OutputModel)
        model.connect(self)

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
            self.log.warning('Task connect, second input model is not supported, adding into comment section')
            return
        self._last_input_model_id = model.id
        model.connect(self)

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
                from IPython import get_ipython
                ip = get_ipython()
                if ip is not None and 'IPKernelApp' in ip.config:
                    return
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
            # This hack prevents Argparse from crashing when running remotely with different set of parameters
            sys.argv = sys.argv[:1]
            self._arguments.copy_to_parser(parser, parsed_args)
        else:
            self._arguments.copy_defaults_from_argparse(parser, args=args, namespace=namespace, parsed_args=parsed_args)

    def _connect_dictionary(self, dictionary):
        self._try_set_connected_parameter_type(self._ConnectedParametersType.dictionary)

        if running_remotely() and self.is_main_task():
            dictionary = self._arguments.copy_to_dict(dictionary)
        else:
            dictionary = self._arguments.copy_from_dict(dictionary)

        return dictionary

    def _connect_task_parameters(self, attr_class):
        self._try_set_connected_parameter_type(self._ConnectedParametersType.task_parameters)

        if running_remotely() and self.is_main_task():
            attr_class.update_from_dict(self.get_parameters())
        else:
            self.set_parameters(attr_class.to_dict())

    def _validate(self, check_output_dest_credentials=False):
        if running_remotely():
            super(Task, self)._validate(check_output_dest_credentials=False)

    def _output_model_updated(self):
        """ Called when a connected output model is updated """
        if running_remotely() or not self.is_main_task():
            return

        # Make sure we know we've started, just in case we didn't so far
        self._dev_mode_task_start(model_updated=True)

        # Store uncommitted code changes
        self._store_uncommitted_code_changes()

    def _store_uncommitted_code_changes(self):
        if running_remotely() or not self.is_main_task():
            return

        if not self.__store_diff_on_train:
            # Feature turned off
            return

        return

    def _dev_mode_task_start(self, model_updated=False):
        """ Called when we suspect the task has started running """
        self._dev_mode_setup_worker(model_updated=model_updated)

    def _dev_mode_stop_task(self, stop_reason):
        # make sure we do not get called (by a daemon thread) after at_exit
        if self._at_exit_called:
            return

        self.get_logger().warn(
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
        include_parent = True
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
        if running_remotely() or not self.is_main_task():
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
        if self._detect_repo_async_thread:
            with self._repo_detect_lock:
                if self._detect_repo_async_thread:
                    try:
                        if self._detect_repo_async_thread.is_alive():
                            self._detect_repo_async_thread.join(timeout=timeout)
                        self._detect_repo_async_thread = None
                    except Exception:
                        pass

    def _summary_artifacts(self):
        # signal artifacts upload, and stop daemon
        self._artifacts_manager.stop(wait=True)
        # print artifacts summary
        self.get_logger().console(self._artifacts_manager.summary)

    def _at_exit(self):
        """
        Will happen automatically once we exit code, i.e. atexit
        :return:
        """
        # protect sub-process at_exit
        if self._at_exit_called:
            return

        is_sub_process = PROC_MASTER_ID_ENV_VAR.get() and PROC_MASTER_ID_ENV_VAR.get() != os.getpid()

        # noinspection PyBroadException
        try:
            # from here do not get into watch dog
            self._at_exit_called = True
            wait_for_uploads = True
            # first thing mark task as stopped, so we will not end up with "running" on lost tasks
            # if we are running remotely, the daemon will take care of it
            task_status = None
            if not running_remotely() and self.is_main_task():
                # check if we crashed, ot the signal is not interrupt (manual break)
                task_status = ('stopped', )
                if self.__exit_hook:
                    if self.__exit_hook.exception is not None or \
                            (not self.__exit_hook.remote_user_aborted and self.__exit_hook.signal not in (None, 2)):
                        task_status = ('failed', 'Exception')
                        wait_for_uploads = False
                    else:
                        wait_for_uploads = (self.__exit_hook.remote_user_aborted or self.__exit_hook.signal is None)
                        if not self.__exit_hook.remote_user_aborted and self.__exit_hook.signal is None:
                            task_status = ('completed', )
                        else:
                            task_status = ('stopped', )

            # wait for repository detection (if we didn't crash)
            if not is_sub_process and wait_for_uploads:
                # we should print summary here
                self._summary_artifacts()
                # make sure that if we crashed the thread we are not waiting forever
                self._wait_for_repo_detection(timeout=10.)

            # wait for uploads
            print_done_waiting = False
            if wait_for_uploads and (BackendModel.get_num_results() > 0 or self.reporter.get_num_results() > 0):
                self.log.info('Waiting to finish uploads')
                print_done_waiting = True
            # from here, do not send log in background thread
            if wait_for_uploads:
                self.flush(wait_for_uploads=True)
                # wait until the reporter flush everything
                self.reporter.stop()
                if print_done_waiting:
                    self.log.info('Finished uploading')
            else:
                self._logger._flush_stdout_handler()

            if not is_sub_process:
                # from here, do not check worker status
                if self._dev_worker:
                    self._dev_worker.unregister()

                # change task status
                if not task_status:
                    pass
                elif task_status[0] == 'failed':
                    self.mark_failed(status_reason=task_status[1])
                elif task_status[0] == 'completed':
                    self.completed()
                elif task_status[0] == 'stopped':
                    self.stopped()

            # stop resource monitoring
            if self._resource_monitor:
                self._resource_monitor.stop()

            self._logger.set_flush_period(None)
            # this is so in theory we can close a main task and start a new one
            Task.__main_task = None
        except Exception:
            # make sure we do not interrupt the exit process
            pass

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
                    for s in self._org_handlers:
                        # noinspection PyBroadException
                        try:
                            signal.signal(s, self._org_handlers[s])
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

                if self._org_handlers:
                    if sys.platform == 'win32':
                        catch_signals = [signal.SIGINT, signal.SIGTERM, signal.SIGSEGV, signal.SIGABRT,
                                         signal.SIGILL, signal.SIGFPE]
                    else:
                        catch_signals = [signal.SIGINT, signal.SIGTERM, signal.SIGSEGV, signal.SIGABRT,
                                         signal.SIGILL, signal.SIGFPE, signal.SIGQUIT]
                    for s in catch_signals:
                        # noinspection PyBroadException
                        try:
                            self._org_handlers[s] = signal.getsignal(s)
                            signal.signal(s, self.signal_handler)
                        except Exception:
                            pass

            def exit(self, code=0):
                self.exit_code = code
                self._orig_exit(code)

            def exc_handler(self, exctype, value, traceback, *args, **kwargs):
                if self._except_recursion_protection_flag:
                    return sys.__excepthook__(exctype, value, traceback, *args, **kwargs)

                self._except_recursion_protection_flag = True
                self.exception = value
                if self._orig_exc_handler:
                    ret = self._orig_exc_handler(exctype, value, traceback, *args, **kwargs)
                else:
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

        res = cls._send(
            cls._get_default_session(),
            projects.GetAllRequest(
                name=exact_match_regex(project_name)
            )
        )
        project = get_single_result(entity='project', query=project_name, results=res.response.projects)

        res = cls._send(
            cls._get_default_session(),
            tasks.GetAllRequest(
                project=[project.id],
                name=exact_match_regex(task_name),
                only_fields=['id', 'name']
            )
        )
        task = get_single_result(entity='task', query=task_name, results=res.response.tasks)

        return cls(
            private=cls.__create_protection,
            task_id=task.id,
            log_to_backend=False,
        )

    @classmethod
    def __get_hash_key(cls, *args):
        def normalize(x):
            return "<{}>".format(x) if x is not None else ""

        return ":".join(map(normalize, args))

    @classmethod
    def __get_last_used_task_id(cls, default_project_name, default_task_name, default_task_type):
        hash_key = cls.__get_hash_key(cls._get_api_server(), default_project_name, default_task_name, default_task_type)

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
        hash_key = cls.__get_hash_key(cls._get_api_server(), default_project_name, default_task_name, default_task_type)

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

        compares = (
            (task.name, 'name'),
            (project_name, 'project'),
            (task.type, 'type'),
        )

        # compare after casting to string to avoid enum instance issues
        # remember we might have replaced the api version by now, so enums are different
        return all(str(server_data) == str(task_data.get(task_data_key)) for server_data, task_data_key in compares)

    @classmethod
    def __close_timed_out_task(cls, task_data):
        if not task_data:
            return False

        task = cls.__get_task_api_obj(task_data.get('id'), ('id', 'status'))

        if task is None:
            return False

        stopped_statuses = (
            tasks.TaskStatusEnum.stopped,
            tasks.TaskStatusEnum.published,
            tasks.TaskStatusEnum.publishing,
            tasks.TaskStatusEnum.closed,
            tasks.TaskStatusEnum.failed,
            tasks.TaskStatusEnum.completed,
        )

        if task.status not in stopped_statuses:
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
