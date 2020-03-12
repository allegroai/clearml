""" Backend task management support """
import itertools
import logging
import os
from enum import Enum
from tempfile import gettempdir
from multiprocessing import RLock
from threading import Thread

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import six
from six.moves.urllib.parse import quote

from ...utilities.locks import RLock as FileRLock
from ...backend_interface.task.development.worker import DevWorker
from ...backend_api import Session
from ...backend_api.services import tasks, models, events, projects
from pathlib2 import Path
from ...utilities.pyhocon import ConfigTree, ConfigFactory

from ..base import IdObjectBase
from ..metrics import Metrics, Reporter
from ..model import Model
from ..setupuploadmixin import SetupUploadMixin
from ..util import make_message, get_or_create_project, get_single_result, \
    exact_match_regex
from ...config import get_config_for_bucket, get_remote_task_id, TASK_ID_ENV_VAR, get_log_to_backend, \
    running_remotely, get_cache_dir, DOCKER_IMAGE_ENV_VAR
from ...debugging import get_logger
from ...debugging.log import LoggerRoot
from ...storage import StorageHelper
from ...storage.helper import StorageError
from .access import AccessMixin
from .log import TaskHandler
from .repo import ScriptInfo
from ...config import config, PROC_MASTER_ID_ENV_VAR


class Task(IdObjectBase, AccessMixin, SetupUploadMixin):
    """ Task manager providing task object access and management. Includes read/write access to task-associated
        frames and models.
    """

    _anonymous_dataview_id = '__anonymous__'
    _development_tag = 'development'

    _store_diff = config.get('development.store_uncommitted_code_diff', False)

    class TaskTypes(Enum):
        def __str__(self):
            return str(self.value)
        training = 'training'
        testing = 'testing'

    def __init__(self, session=None, task_id=None, log=None, project_name=None,
                 task_name=None, task_type=TaskTypes.training, log_to_backend=True,
                 raise_on_validation_errors=True, force_create=False):
        """
        Create a new task instance.
        :param session: Optional API Session instance. If not provided, a default session based on the system's
            configuration will be used.
        :type session: Session
        :param task_id: Optional task ID. If not provided, a new task will be created using the API
            and its information reflected in the resulting instance.
        :type task_id: string
        :param log: Optional log to be used. If not provided, and internal log shared with all backend objects will be
            used instead.
        :type log: logging.Logger
        :param project_name: Optional project name, used only if a new task is created. The new task will be associated
            with a project by this name. If no such project exists, a new project will be created using the API.
        :type project_name: str
        :param task_name: Optional task name, used only if a new task is created.
        :type project_name: str
        :param task_type: Optional task type, used only if a new task is created. Default is training task.
        :type project_name: str (see tasks.TaskTypeEnum)
        :param log_to_backend: If True, all calls to the task's log will be logged to the backend using the API.
            This value can be overridden using the environment variable TRAINS_LOG_TASK_TO_BACKEND.
        :type log_to_backend: bool
        :param force_create: If True a new task will always be created (task_id, if provided, will be ignored)
        :type force_create: bool
        """
        task_id = self._resolve_task_id(task_id, log=log) if not force_create else None
        self.__edit_lock = None
        super(Task, self).__init__(id=task_id, session=session, log=log)
        self._project_name = None
        self._storage_uri = None
        self._input_model = None
        self._output_model = None
        self._metrics_manager = None
        self._reporter = None
        self._curr_label_stats = {}
        self._raise_on_validation_errors = raise_on_validation_errors
        self._parameters_allowed_types = (
                six.string_types + six.integer_types + (six.text_type, float, list, tuple, dict, type(None))
        )
        self._app_server = None
        self._files_server = None
        self._initial_iteration_offset = 0

        if not task_id:
            # generate a new task
            self.id = self._auto_generate(project_name=project_name, task_name=task_name, task_type=task_type)
        else:
            # this is an existing task, let's try to verify stuff
            self._validate()

        self._project_name = (self.project, project_name)

        if running_remotely() or DevWorker.report_stdout:
            log_to_backend = False
        self._log_to_backend = log_to_backend
        self._setup_log(default_log_to_backend=log_to_backend)

    def _setup_log(self, default_log_to_backend=None, replace_existing=False):
        """
        Setup logging facilities for this task.
        :param default_log_to_backend: Should this task log to the backend. If not specified, value for this option
        will be obtained from the environment, with this value acting as a default in case configuration for this is
        missing.
        If the value for this option is false, we won't touch the current logger configuration regarding TaskHandler(s)
        :param replace_existing: If True and another task is already logging to the backend, replace the handler with
        a handler for this task.
        """
        # Make sure urllib is never in debug/info,
        disable_urllib3_info = config.get('log.disable_urllib3_info', True)
        if disable_urllib3_info and logging.getLogger('urllib3').isEnabledFor(logging.INFO):
            logging.getLogger('urllib3').setLevel(logging.WARNING)

        log_to_backend = get_log_to_backend(default=default_log_to_backend) or self._log_to_backend
        if not log_to_backend:
            return

        # Handle the root logger and our own logger. We use set() to make sure we create no duplicates
        # in case these are the same logger...
        loggers = {logging.getLogger(), LoggerRoot.get_base_logger()}

        # Find all TaskHandler handlers for these loggers
        handlers = {logger: h for logger in loggers for h in logger.handlers if isinstance(h, TaskHandler)}

        if handlers and not replace_existing:
            # Handlers exist and we shouldn't replace them
            return

        # Remove all handlers, we'll add new ones
        for logger, handler in handlers.items():
            logger.removeHandler(handler)

        # Create a handler that will be used in all loggers. Since our handler is a buffering handler, using more
        # than one instance to report to the same task will result in out-of-order log reports (grouped by whichever
        # handler instance handled them)
        backend_handler = TaskHandler(self.session, self.task_id)

        # Add backend handler to both loggers:
        # 1. to root logger root logger
        # 2. to our own logger as well, since our logger is not propagated to the root logger
        #    (if we propagate our logger will be caught be the root handlers as well, and
        #    we do not want that)
        for logger in loggers:
            logger.addHandler(backend_handler)

    def _validate(self, check_output_dest_credentials=True):
        raise_errors = self._raise_on_validation_errors
        output_dest = self.get_output_destination(raise_on_error=False, log_on_error=False)
        if output_dest and check_output_dest_credentials:
            try:
                self.log.info('Validating output destination')
                conf = get_config_for_bucket(base_url=output_dest)
                if not conf:
                    msg = 'Failed resolving output destination (no credentials found for %s)' % output_dest
                    self.log.warning(msg)
                    if raise_errors:
                        raise Exception(msg)
                else:
                    StorageHelper._test_bucket_config(conf=conf, log=self.log, raise_on_error=raise_errors)
            except StorageError:
                raise
            except Exception as ex:
                self.log.error('Failed trying to verify output destination: %s' % ex)

    @classmethod
    def _resolve_task_id(cls, task_id, log=None):
        if not task_id:
            task_id = cls.normalize_id(get_remote_task_id())
            if task_id:
                log = log or get_logger('task')
                log.info('Using task ID from env %s=%s' % (TASK_ID_ENV_VAR[0], task_id))
        return task_id

    def _update_repository(self):
        def check_package_update():
            try:
                # check latest version
                from ...utilities.check_updates import CheckPackageUpdates
                latest_version = CheckPackageUpdates.check_new_package_available(only_once=True)
                if latest_version:
                    if not latest_version[1]:
                        sep = os.linesep
                        self.get_logger().report_text(
                            'TRAINS new package available: UPGRADE to v{} is recommended!\nRelease Notes:\n{}'.format(
                                latest_version[0], sep.join(latest_version[2])),
                        )
                    else:
                        self.get_logger().report_text(
                            'TRAINS new version available: upgrade to v{} is recommended!'.format(
                                latest_version[0]),
                        )
            except Exception:
                pass

        # get repository and create requirements.txt from code base
        try:
            check_package_update_thread = Thread(target=check_package_update)
            check_package_update_thread.daemon = True
            check_package_update_thread.start()
            # do not request requirements, because it might be a long process, and we first want to update the git repo
            result, script_requirements = ScriptInfo.get(
                log=self.log, create_requirements=False, check_uncommitted=self._store_diff
            )
            for msg in result.warning_messages:
                self.get_logger().report_text(msg)

            self.data.script = result.script
            # Since we might run asynchronously, don't use self.data (lest someone else
            # overwrite it before we have a chance to call edit)
            self._edit(script=result.script)
            self.reload()
            # if jupyter is present, requirements will be created in the background, when saving a snapshot
            if result.script and script_requirements:
                requirements, conda_requirements = script_requirements.get_requirements()
                if requirements:
                    if not result.script['requirements']:
                        result.script['requirements'] = {}
                    result.script['requirements']['pip'] = requirements
                    result.script['requirements']['conda'] = conda_requirements

                self._update_requirements(result.script.get('requirements') or '')
                self.reload()

            # we do not want to wait for the check version thread,
            # because someone might wait for us to finish the repo detection update
        except Exception as e:
            get_logger('task').debug(str(e))

    def _auto_generate(self, project_name=None, task_name=None, task_type=TaskTypes.training):
        created_msg = make_message('Auto-generated at %(time)s by %(user)s@%(host)s')

        project_id = None
        if project_name:
            project_id = get_or_create_project(self, project_name, created_msg)

        tags = [self._development_tag] if not running_remotely() else []
        extra_properties = {'system_tags': tags} if Session.check_min_api_version('2.3') else {'tags': tags}
        req = tasks.CreateRequest(
            name=task_name or make_message('Anonymous task (%(user)s@%(host)s %(time)s)'),
            type=tasks.TaskTypeEnum(task_type.value),
            comment=created_msg,
            project=project_id,
            input={'view': {}},
            **extra_properties
        )
        res = self.send(req)

        return res.response.id

    def _set_storage_uri(self, value):
        value = value.rstrip('/') if value else None
        self._storage_uri = StorageHelper.conform_url(value)
        self.data.output.destination = self._storage_uri
        self._edit(output_dest=self._storage_uri or ('' if Session.check_min_api_version('2.3') else None))
        if self._storage_uri or self._output_model:
            self.output_model.upload_storage_uri = self._storage_uri

    @property
    def storage_uri(self):
        if self._storage_uri:
            return self._storage_uri
        if running_remotely():
            return self.data.output.destination
        else:
            return None

    @storage_uri.setter
    def storage_uri(self, value):
        self._set_storage_uri(value)

    @property
    def task_id(self):
        return self.id

    @property
    def name(self):
        return self.data.name or ''

    @name.setter
    def name(self, value):
        self.set_name(value)

    @property
    def task_type(self):
        return self.data.type

    @property
    def project(self):
        return self.data.project

    @property
    def parent(self):
        return self.data.parent

    @property
    def input_model_id(self):
        return self.data.execution.model

    @property
    def output_model_id(self):
        return self.data.output.model

    @property
    def comment(self):
        return self.data.comment or ''

    @comment.setter
    def comment(self, value):
        self.set_comment(value)

    @property
    def cache_dir(self):
        """ Cache dir used to store task related files """
        return Path(get_cache_dir()) / self.id

    @property
    def status(self):
        """ The task's status. In order to stay updated, we always reload the task info when this value is accessed. """
        self.reload()
        return self._status

    @property
    def _status(self):
        """ Return the task's cached status (don't reload if we don't have to) """
        return str(self.data.status)

    @property
    def input_model(self):
        """ A model manager used to handle the input model object """
        model_id = self._get_task_property('execution.model', raise_on_error=False)
        if not model_id:
            return None
        if self._input_model is None:
            self._input_model = Model(
                session=self.session,
                model_id=model_id,
                cache_dir=self.cache_dir,
                log=self.log,
                upload_storage_uri=None)
        return self._input_model

    @property
    def output_model(self):
        """ A model manager used to manage the output model object """
        if self._output_model is None:
            self._output_model = self._get_output_model(upload_required=True)
        return self._output_model

    def create_output_model(self):
        return self._get_output_model(upload_required=False, force=True)

    def _get_output_model(self, upload_required=True, force=False):
        return Model(
            session=self.session,
            model_id=None if force else self._get_task_property(
                'output.model', raise_on_error=False, log_on_error=False),
            cache_dir=self.cache_dir,
            upload_storage_uri=self.storage_uri or self.get_output_destination(
                raise_on_error=upload_required, log_on_error=upload_required),
            upload_storage_suffix=self._get_output_destination_suffix('models'),
            log=self.log)

    @property
    def metrics_manager(self):
        """ A metrics manager used to manage the metrics related to this task """
        return self._get_metrics_manager(self.get_output_destination())

    @property
    def reporter(self):
        """
        Returns a simple metrics reporter instance
        """
        if self._reporter is None:
            self._setup_reporter()
        return self._reporter

    def _get_metrics_manager(self, storage_uri):
        if self._metrics_manager is None:
            self._metrics_manager = Metrics(
                session=self.session,
                task_id=self.id,
                storage_uri=storage_uri,
                storage_uri_suffix=self._get_output_destination_suffix('metrics'),
                iteration_offset=self.get_initial_iteration()
            )
        return self._metrics_manager

    def _setup_reporter(self):
        try:
            storage_uri = self.get_output_destination(log_on_error=False)
        except ValueError:
            storage_uri = None
        self._reporter = Reporter(self._get_metrics_manager(storage_uri=storage_uri))
        return self._reporter

    def _get_output_destination_suffix(self, extra_path=None):
        return '/'.join(quote(x, safe="'[]{}()$^,.; -_+-=") for x in
                        (self.get_project_name(), '%s.%s' % (self.name, self.data.id), extra_path) if x)

    def _reload(self):
        """ Reload the task object from the backend """
        with self._edit_lock:
            res = self.send(tasks.GetByIdRequest(task=self.id))
            return res.response.task

    def reset(self, set_started_on_success=True):
        """ Reset the task. Task will be reloaded following a successful reset. """
        self.send(tasks.ResetRequest(task=self.id))
        if set_started_on_success:
            self.started()
        self.reload()

    def started(self, ignore_errors=True):
        """ Signal that this task has started """
        return self.send(tasks.StartedRequest(self.id), ignore_errors=ignore_errors)

    def stopped(self, ignore_errors=True):
        """ Signal that this task has stopped """
        return self.send(tasks.StoppedRequest(self.id), ignore_errors=ignore_errors)

    def completed(self, ignore_errors=True):
        """ Signal that this task has been completed """
        if hasattr(tasks, 'CompletedRequest'):
            return self.send(tasks.CompletedRequest(self.id, status_reason='completed'), ignore_errors=ignore_errors)
        return self.send(tasks.StoppedRequest(self.id, status_reason='completed'), ignore_errors=ignore_errors)

    def mark_failed(self, ignore_errors=True, status_reason=None, status_message=None):
        """ Signal that this task has stopped """
        return self.send(tasks.FailedRequest(self.id, status_reason=status_reason, status_message=status_message),
                         ignore_errors=ignore_errors)

    def publish(self, ignore_errors=True):
        """ Signal that this task will be published """
        if str(self.status) != str(tasks.TaskStatusEnum.stopped):
            raise ValueError("Can't publish, Task is not stopped")
        resp = self.send(tasks.PublishRequest(self.id), ignore_errors=ignore_errors)
        assert isinstance(resp.response, tasks.PublishResponse)
        return resp

    def update_model_desc(self, new_model_desc_file=None):
        """ Change the task's model_desc """
        with self._edit_lock:
            self.reload()
            execution = self._get_task_property('execution')
            p = Path(new_model_desc_file)
            if not p.is_file():
                raise IOError('mode_desc file %s cannot be found' % new_model_desc_file)
            new_model_desc = p.read_text()
            model_desc_key = list(execution.model_desc.keys())[0] if execution.model_desc else 'design'
            execution.model_desc[model_desc_key] = new_model_desc

            res = self._edit(execution=execution)
            return res.response

    def update_output_model(self, model_uri, name=None, comment=None, tags=None):
        """
        Update the task's output model.
        Note that this method only updates the model's metadata using the API and does not upload any data. Use this
        method to update the output model when you have a local model URI (e.g. storing the weights file locally and
        providing a file://path/to/file URI)

        :param model_uri: URI for the updated model weights file
        :type model_uri: str
        :param name: Optional updated model name
        :type name: str
        :param comment: Optional updated model description
        :type comment: str
        :param tags: Optional updated model tags
        :type tags: [str]
        """
        self._conditionally_start_task()
        self._get_output_model(upload_required=False).update_for_task(model_uri, self.id, name, comment, tags)

    def update_output_model_and_upload(
            self, model_file, name=None, comment=None, tags=None, async_enable=False, cb=None, iteration=None):
        """
        Update the task's output model weights file. File is first uploaded to the preconfigured output destination (see
        task's output.destination property or call setup_upload()), than the model object associated with the task is
        updated using an API call with the URI of the uploaded file (and other values provided by additional arguments)

        :param model_file: Path to the updated model weights file
        :type model_file: str
        :param name: Optional updated model name
        :type name: str
        :param comment: Optional updated model description
        :type comment: str
        :param tags: Optional updated model tags
        :type tags: [str]
        :param async_enable: Request asynchronous upload. If False, the call blocks until upload is completed and the
            API call updating the model returns. If True, the call returns immediately, while upload and update are
            scheduled in another thread. Default is False.
        :type async_enable: bool
        :param cb: Asynchronous callback. If async=True, this callback will be invoked once the asynchronous upload and
            update have completed.
        :return: The URI of the uploaded weights file. If async=True, this is the expected URI as the upload is
            probably still in progress.
        """
        self._conditionally_start_task()
        uri = self.output_model.update_for_task_and_upload(
            model_file, self.id, name=name, comment=comment, tags=tags, async_enable=async_enable, cb=cb,
            iteration=iteration
        )
        return uri

    def _conditionally_start_task(self):
        if str(self.status) == str(tasks.TaskStatusEnum.created):
            self.started()

    @property
    def labels_stats(self):
        """ Get accumulated label stats for the current/last frames iteration """
        return self._curr_label_stats

    def _accumulate_label_stats(self, roi_stats, reset=False):
        if reset:
            self._curr_label_stats = {}
        for label in roi_stats:
            if label in self._curr_label_stats:
                self._curr_label_stats[label] += roi_stats[label]
            else:
                self._curr_label_stats[label] = roi_stats[label]

    def set_input_model(self, model_id=None, model_name=None, update_task_design=True, update_task_labels=True):
        """
        Set a new input model for this task. Model must be 'ready' in order to be used as the Task's input model.

        :param model_id: ID for a model that exists in the backend. Required if model_name is not provided.
        :param model_name: Model name. Required if model_id is not provided. If provided, this name will be used to
            locate an existing model in the backend.
        :param update_task_design: if True, the task's model design will be copied from the input model
        :param update_task_labels: if True, the task's label enumeration will be copied from the input model
        """
        if model_id is None and not model_name:
            raise ValueError('Expected one of [model_id, model_name]')

        if model_name:
            # Try getting the model by name. Limit to 10 results.
            res = self.send(
                models.GetAllRequest(
                    name=exact_match_regex(model_name),
                    ready=True,
                    page=0,
                    page_size=10,
                    order_by='-created',
                    only_fields=['id', 'created']
                )
            )
            model = get_single_result(entity='model', query=model_name, results=res.response.models, log=self.log)
            model_id = model.id

        if model_id:
            res = self.send(models.GetByIdRequest(model=model_id))
            model = res.response.model
            if not model.ready:
                # raise ValueError('Model %s is not published (not ready)' % model_id)
                self.log.debug('Model %s [%s] is not published yet (not ready)' % (model_id, model.uri))
        else:
            # clear the input model
            model = None
            model_id = ''

        with self._edit_lock:
            self.reload()
            # store model id
            self.data.execution.model = model_id

            # Auto populate input field from model, if they are empty
            if update_task_design and not self.data.execution.model_desc:
                self.data.execution.model_desc = model.design if model else ''
            if update_task_labels and not self.data.execution.model_labels:
                self.data.execution.model_labels = model.labels if model else {}

            self._edit(execution=self.data.execution)

    def set_parameters(self, *args, **kwargs):
        """
        Set parameters for this task. This allows setting a complete set of key/value parameters, but does not support
        parameter descriptions (as the input is a dictionary or key/value pairs.

        :param args: Positional arguments (one or more dictionary or (key, value) iterable). These will be merged into
            a single key/value dictionary.
        :param kwargs: Key/value pairs, merged into the parameters dictionary created from `args`.
        """
        if not all(isinstance(x, (dict, Iterable)) for x in args):
            raise ValueError('only dict or iterable are supported as positional arguments')

        update = kwargs.pop('__update', False)

        with self._edit_lock:
            self.reload()
            if update:
                parameters = self.get_parameters()
            else:
                parameters = dict()
            parameters.update(itertools.chain.from_iterable(x.items() if isinstance(x, dict) else x for x in args))
            parameters.update(kwargs)

            not_allowed = {
                k: type(v).__name__
                for k, v in parameters.items()
                if not isinstance(v, self._parameters_allowed_types)
            }
            if not_allowed:
                raise ValueError(
                    "Only builtin types ({}) are allowed for values (got {})".format(
                        ', '.join(t.__name__ for t in self._parameters_allowed_types),
                        ', '.join('%s=>%s' % p for p in not_allowed.items())),
                    )

            # force cast all variables to strings (so that we can later edit them in UI)
            parameters = {k: str(v) if v is not None else "" for k, v in parameters.items()}

            execution = self.data.execution
            if execution is None:
                execution = tasks.Execution(parameters=parameters)
            else:
                execution.parameters = parameters
            self._edit(execution=execution)

    def set_parameter(self, name, value, description=None):
        """
        Set a single task parameter. This overrides any previous value for this parameter.

        :param name: Parameter name
        :param value: Parameter value
        :param description: Parameter description (unused for now)
        """
        self.set_parameters({name: value}, __update=True)

    def get_parameter(self, name, default=None):
        """
        Get a value for a parameter.

        :param name: Parameter name
        :param default: Default value
        :return: Parameter value (or default value if parameter is not defined)
        """
        params = self.get_parameters()
        return params.get(name, default)

    def update_parameters(self, *args, **kwargs):
        """
        Update parameters for this task.

        This allows updating a complete set of key/value parameters,but does not support
        parameter descriptions (as the input is a dictionary or key/value pairs.

        :param args: Positional arguments (one or more dictionary or (key, value) iterable). These will be merged into
            a single key/value dictionary.
        :param kwargs: Key/value pairs, merged into the parameters dictionary created from `args`.
        """
        self.set_parameters(__update=True, *args, **kwargs)

    def set_model_label_enumeration(self, enumeration=None):
        """
        Set a dictionary of labels (text) to ids (integers) {str(label): integer(id)}

        :param dict enumeration: For example: {str(label): integer(id)}
        """
        enumeration = enumeration or {}
        with self._edit_lock:
            self.reload()
            execution = self.data.execution
            if enumeration is None:
                return
            if not (isinstance(enumeration, dict)
                    and all(isinstance(k, six.string_types) and isinstance(v, int) for k, v in enumeration.items())):
                raise ValueError('Expected label to be a dict[str => int]')
            execution.model_labels = enumeration
            self._edit(execution=execution)

    def _set_default_docker_image(self):
        if not DOCKER_IMAGE_ENV_VAR.exists():
            return
        self.set_base_docker(DOCKER_IMAGE_ENV_VAR.get(default=""))

    def set_base_docker(self, docker_cmd):
        """
        Set the base docker image for this experiment
        If provided, this value will be used by trains-agent to execute this experiment
         inside the provided docker image.
        """
        with self._edit_lock:
            self.reload()
            execution = self.data.execution
            execution.docker_cmd = docker_cmd
            self._edit(execution=execution)

    def get_base_docker(self):
        """Get the base docker command (image) set for this experiment"""
        return self._get_task_property('execution.docker_cmd', raise_on_error=False, log_on_error=False)

    def set_artifacts(self, artifacts_list=None):
        """
        List of artifacts (tasks.Artifact) to update the task

        :param list artifacts_list: list of artifacts (type tasks.Artifact)
        """
        if not Session.check_min_api_version('2.3'):
            return False
        if not (isinstance(artifacts_list, (list, tuple))
                and all(isinstance(a, tasks.Artifact) for a in artifacts_list)):
            raise ValueError('Expected artifacts to [tasks.Artifacts]')
        with self._edit_lock:
            self.reload()
            execution = self.data.execution
            keys = [a.key for a in artifacts_list]
            execution.artifacts = [a for a in execution.artifacts or [] if a.key not in keys] + artifacts_list
            self._edit(execution=execution)

    def _set_model_design(self, design=None):
        with self._edit_lock:
            self.reload()
            execution = self.data.execution
            if design is not None:
                execution.model_desc = Model._wrap_design(design)

            self._edit(execution=execution)

    def get_labels_enumeration(self):
        """
        Return a dictionary of labels (text) to ids (integers) {str(label): integer(id)}
        :return: dict
        """
        if not self.data or not self.data.execution:
            return {}
        return self.data.execution.model_labels

    def get_model_design(self):
        """
        Returns the model configuration as blob of text
        :return:
        """
        design = self._get_task_property("execution.model_desc", default={}, raise_on_error=False, log_on_error=False)
        return Model._unwrap_design(design)

    def set_output_model_id(self, model_id):
        self.data.output.model = str(model_id)
        self._edit(output=self.data.output)

    def get_random_seed(self):
        # fixed seed for the time being
        return 1337

    def set_random_seed(self, random_seed):
        # fixed seed for the time being
        pass

    def set_project(self, project_id):
        assert isinstance(project_id, six.string_types)
        self._set_task_property("project", project_id)
        self._edit(project=project_id)

    def get_project_name(self):
        if self.project is None:
            return None

        if self._project_name and self._project_name[1] is not None and self._project_name[0] == self.project:
            return self._project_name[1]

        res = self.send(projects.GetByIdRequest(project=self.project), raise_on_errors=False)
        if not res or not res.response or not res.response.project:
            return None
        self._project_name = (self.project, res.response.project.name)
        return self._project_name[1]

    def get_tags(self):
        return self._get_task_property("tags")

    def set_system_tags(self, tags):
        assert isinstance(tags, (list, tuple))
        if Session.check_min_api_version('2.3'):
            self._set_task_property("system_tags", tags)
            self._edit(system_tags=self.data.system_tags)
        else:
            self._set_task_property("tags", tags)
            self._edit(tags=self.data.tags)

    def set_tags(self, tags):
        assert isinstance(tags, (list, tuple))
        if not Session.check_min_api_version('2.3'):
            # not supported
            return
        self._set_task_property("tags", tags)
        self._edit(tags=self.data.tags)

    def set_name(self, name):
        """
        Set a comment text to the task.

        :param name: The name of the task
        :type name: str
        """
        self._set_task_property("name", str(name))
        self._edit(name=self.data.name)

    def set_comment(self, comment):
        """
        Set a comment text to the task.

        :param comment: The comment of the task
        :type comment: str
        """
        self._set_task_property("comment", str(comment))
        self._edit(comment=comment)

    def set_initial_iteration(self, offset=0):
        """
        Set initial iteration, instead of zero. Useful when continuing training from previous checkpoints

        :param int offset: Initial iteration (at starting point)
        :return: newly set initial offset
        """
        if not isinstance(offset, int):
            raise ValueError("Initial iteration offset must be an integer")

        self._initial_iteration_offset = offset
        if self._metrics_manager:
            self._metrics_manager.set_iteration_offset(self._initial_iteration_offset)
        return self._initial_iteration_offset

    def get_initial_iteration(self):
        """
        Return the initial iteration offset, default is 0.
        Useful when continuing training from previous checkpoints.

        :return int: initial iteration offset
        """
        return self._initial_iteration_offset

    def _get_default_report_storage_uri(self):
        if not self._files_server:
            self._files_server = Session.get_files_server_host()
        return self._files_server

    def _get_status(self):
        try:
            all_tasks = self.send(
                tasks.GetAllRequest(id=[self.id], only_fields=['status', 'status_message']),
            ).response.tasks
            return all_tasks[0].status, all_tasks[0].status_message
        except Exception:
            return None, None

    def _reload_last_iteration(self):
        try:
            all_tasks = self.send(
                tasks.GetAllRequest(id=[self.id], only_fields=['last_iteration']),
            ).response.tasks
            self.data.last_iteration = all_tasks[0].last_iteration
        except Exception:
            return None

    @classmethod
    def _get_api_server(cls):
        return Session.get_api_server_host()

    def _get_app_server(self):
        if not self._app_server:
            self._app_server = Session.get_app_server_host()
        return self._app_server

    def _edit(self, **kwargs):
        with self._edit_lock:
            # Since we ae using forced update, make sure he task status is valid
            if not self._data or (str(self.data.status) not in (str(tasks.TaskStatusEnum.created),
                                                                str(tasks.TaskStatusEnum.in_progress))):
                # the exception being name/comment that we can always change.
                if kwargs and all(k in ('name', 'comment') for k in kwargs.keys()):
                    pass
                else:
                    raise ValueError('Task object can only be updated if created or in_progress')

            res = self.send(tasks.EditRequest(task=self.id, force=True, **kwargs), raise_on_errors=False)
            return res

    def _update_requirements(self, requirements):
        if not isinstance(requirements, dict):
            requirements = {'pip': requirements}
        # protection, Old API might not support it
        try:
            self.data.script.requirements = requirements
            self.send(tasks.SetRequirementsRequest(task=self.id, requirements=requirements))
        except Exception:
            pass

    def _update_script(self, script):
        self.data.script = script
        self._edit(script=script)

    @classmethod
    def _clone_task(cls, cloned_task_id, name=None, comment=None, execution_overrides=None,
                    tags=None, parent=None, project=None, log=None, session=None):
        """
        Clone a task

        :param cloned_task_id: Task ID for the task to be cloned
        :type cloned_task_id: str
        :param name: New for the new task
        :type name: str
        :param comment: Optional comment for the new task
        :type comment: str
        :param execution_overrides: Task execution overrides. Applied over the cloned task's execution
            section, useful for overriding values in the cloned task.
        :type execution_overrides: dict
        :param tags: Optional updated model tags
        :type tags: [str]
        :param parent: Optional parent Task ID of the new task.
        :type parent: str
        :param project: Optional project ID of the new task.
            If None, the new task will inherit the cloned task's project.
        :type project: str
        :param log: Log object used by the infrastructure.
        :type log: logging.Logger
        :param session: Session object used for sending requests to the API
        :type session: Session
        :return: The new tasks's ID
        """

        session = session if session else cls._get_default_session()

        res = cls._send(session=session, log=log, req=tasks.GetByIdRequest(task=cloned_task_id))
        task = res.response.task
        output_dest = None
        if task.output:
            output_dest = task.output.destination
        execution = task.execution.to_dict() if task.execution else {}
        execution = ConfigTree.merge_configs(ConfigFactory.from_dict(execution),
                                             ConfigFactory.from_dict(execution_overrides or {}))
        # clear all artifacts
        execution['artifacts'] = [e for e in execution['artifacts'] if e.get('mode') == 'input']

        if not tags and task.tags:
            tags = [t for t in task.tags if t != cls._development_tag]

        req = tasks.CreateRequest(
            name=name or task.name,
            type=task.type,
            input=task.input if hasattr(task, 'input') else {'view': {}},
            tags=tags,
            comment=comment or task.comment,
            parent=parent,
            project=project if project else task.project,
            output_dest=output_dest,
            execution=execution.as_plain_ordered_dict(),
            script=task.script
        )
        res = cls._send(session=session, log=log, req=req)
        cloned_task_id = res.response.id

        if task.script and task.script.requirements:
            cls._send(session=session, log=log, req=tasks.SetRequirementsRequest(
                task=cloned_task_id, requirements=task.script.requirements))
        return cloned_task_id

    @classmethod
    def get_all(cls, session=None, log=None, **kwargs):
        """
        List all tasks based on specific projection

        :param session: Session object used for sending requests to the API
        :type session: Session
        :param log: Log object
        :type log: logging.Logger
        :param kwargs: Keyword args passed to the GetAllRequest (see .backend_api.services.tasks.GetAllRequest)
            Example: status='completed', 'search_text'='specific_word', 'user'='user_id', 'project'='project_id'
        :type kwargs: dict
        :return: API response
        """
        session = session if session else cls._get_default_session()
        req = tasks.GetAllRequest(**kwargs)
        res = cls._send(session=session, req=req, log=log)
        return res

    @classmethod
    def get_by_name(cls, task_name):
        res = cls._send(cls._get_default_session(), tasks.GetAllRequest(name=exact_match_regex(task_name)))

        task = get_single_result(entity='task', query=task_name, results=res.response.tasks)
        return cls(task_id=task.id)

    def _get_all_events(self, max_events=100):
        """
        Get a list of all reported events.

        Warning: Debug only. Do not use outside of testing.

        :param max_events: The maximum events the function will return. Pass None
            to return all the reported events.
        :return: A list of events from the task.
        """

        log_events = self.send(events.GetTaskEventsRequest(
            task=self.id,
            order='asc',
            batch_size=max_events,
        ))

        events_list = log_events.response.events
        total_events = log_events.response.total
        scroll = log_events.response.scroll_id

        while len(events_list) < total_events and (max_events is None or len(events_list) < max_events):
            log_events = self.send(events.GetTaskEventsRequest(
                task=self.id,
                order='asc',
                batch_size=max_events,
                scroll_id=scroll,
            ))
            events_list.extend(log_events.response.events)
            scroll = log_events.response.scroll_id

        return events_list

    @property
    def _edit_lock(self):
        if self.__edit_lock:
            return self.__edit_lock
        if not PROC_MASTER_ID_ENV_VAR.get() or len(PROC_MASTER_ID_ENV_VAR.get().split(':')) < 2:
            self.__edit_lock = RLock()
        elif PROC_MASTER_ID_ENV_VAR.get().split(':')[1] == str(self.id):
            # remove previous file lock instance, just in case.
            filename = os.path.join(gettempdir(), 'trains_{}.lock'.format(self.id))
            try:
                os.unlink(filename)
            except Exception:
                pass
            # create a new file based lock
            self.__edit_lock = FileRLock(filename=filename)
        else:
            self.__edit_lock = RLock()
        return self.__edit_lock

    @_edit_lock.setter
    def _edit_lock(self, value):
        self.__edit_lock = value

    @classmethod
    def __update_master_pid_task(cls, pid=None, task=None):
        pid = pid or os.getpid()
        if not task:
            PROC_MASTER_ID_ENV_VAR.set(str(pid) + ':')
        elif isinstance(task, str):
            PROC_MASTER_ID_ENV_VAR.set(str(pid) + ':' + task)
        else:
            PROC_MASTER_ID_ENV_VAR.set(str(pid) + ':' + str(task.id))
            # make sure we refresh the edit lock next time we need it,
            task._edit_lock = None

    @classmethod
    def __get_master_id_task_id(cls):
        master_task_id = PROC_MASTER_ID_ENV_VAR.get().split(':')
        # we could not find a task ID, revert to old stub behaviour
        if len(master_task_id) < 2 or not master_task_id[1]:
            return None
        return master_task_id[1]

    @classmethod
    def __is_subprocess(cls):
        # notice this class function is called from Task.ExitHooks, do not rename/move it.
        is_subprocess = PROC_MASTER_ID_ENV_VAR.get() and \
                        PROC_MASTER_ID_ENV_VAR.get().split(':')[0] != str(os.getpid())
        return is_subprocess
