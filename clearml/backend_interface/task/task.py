""" Backend task management support """
import itertools
import json
import logging
import os
import re
import sys
from copy import copy
from datetime import datetime
from enum import Enum
from multiprocessing import RLock
from operator import itemgetter
from tempfile import gettempdir
from threading import Thread
from typing import Optional, Any, Sequence, Callable, Mapping, Union, List, Set
from uuid import uuid4

from pathlib2 import Path

try:
    # noinspection PyCompatibility
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import six
from six.moves.urllib.parse import quote

from ...utilities.locks import RLock as FileRLock
from ...utilities.proxy_object import verify_basic_type
from ...binding.artifacts import Artifacts
from ...backend_interface.task.development.worker import DevWorker
from ...backend_interface.session import SendError
from ...backend_api import Session
from ...backend_api.services import tasks, models, events, projects
from ...backend_api.session.defs import ENV_OFFLINE_MODE
from ...utilities.pyhocon import ConfigTree, ConfigFactory
from ...utilities.config import config_dict_to_text, text_to_config_dict

from ..base import IdObjectBase, InterfaceBase
from ..metrics import Metrics, Reporter
from ..model import Model
from ..setupuploadmixin import SetupUploadMixin
from ..util import (
    make_message, get_or_create_project, get_single_result,
    exact_match_regex, mutually_exclusive, )
from ...config import (
    get_config_for_bucket, get_remote_task_id, TASK_ID_ENV_VAR,
    running_remotely, get_cache_dir, DOCKER_IMAGE_ENV_VAR, get_offline_dir, get_log_to_backend, )
from ...debugging import get_logger
from ...storage.helper import StorageHelper, StorageError
from .access import AccessMixin
from .repo import ScriptInfo, pip_freeze
from .hyperparams import HyperParams
from ...config import config, PROC_MASTER_ID_ENV_VAR, SUPPRESS_UPDATE_MESSAGE_ENV_VAR, DOCKER_BASH_SETUP_ENV_VAR
from ...utilities.process.mp import SingletonLock


class Task(IdObjectBase, AccessMixin, SetupUploadMixin):
    """ Task manager providing task object access and management. Includes read/write access to task-associated
        frames and models.
    """

    _anonymous_dataview_id = '__anonymous__'
    _development_tag = 'development'
    archived_tag = 'archived'
    _default_configuration_section_name = 'General'
    _legacy_parameters_section_name = 'Args'
    _force_requirements = {}
    _ignore_requirements = set()

    _store_diff = config.get('development.store_uncommitted_code_diff', False)
    _store_remote_diff = config.get('development.store_code_diff_from_remote', False)
    _report_subprocess_enabled = config.get('development.report_use_subprocess', sys.platform == 'linux')
    _force_use_pip_freeze = \
        config.get('development.detect_with_pip_freeze', False) or \
        config.get('development.detect_with_conda_freeze', False)
    _offline_filename = 'task.json'

    class TaskTypes(Enum):
        def __str__(self):
            return str(self.value)

        def __eq__(self, other):
            return str(self) == str(other)

        training = 'training'
        testing = 'testing'
        inference = "inference"
        data_processing = "data_processing"
        application = "application"
        monitor = "monitor"
        controller = "controller"
        optimizer = "optimizer"
        service = "service"
        qc = "qc"
        custom = "custom"

    class TaskStatusEnum(Enum):
        def __str__(self):
            return str(self.value)

        def __eq__(self, other):
            return str(self) == str(other)

        created = "created"
        queued = "queued"
        in_progress = "in_progress"
        stopped = "stopped"
        published = "published"
        publishing = "publishing"
        closed = "closed"
        failed = "failed"
        completed = "completed"
        unknown = "unknown"

    class DeleteError(Exception):
        pass

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
        :type task_type: str (see tasks.TaskTypeEnum)
        :param log_to_backend: If True, all calls to the task's log will be logged to the backend using the API.
            This value can be overridden using the environment variable TRAINS_LOG_TASK_TO_BACKEND.
        :type log_to_backend: bool
        :param force_create: If True a new task will always be created (task_id, if provided, will be ignored)
        :type force_create: bool
        """
        SingletonLock.instantiate()
        task_id = self._resolve_task_id(task_id, log=log) if not force_create else None
        self.__edit_lock = None
        super(Task, self).__init__(id=task_id, session=session, log=log)
        self._project_name = None
        self._storage_uri = None
        self._metrics_manager = None
        self.__reporter = None
        self._curr_label_stats = {}
        self._raise_on_validation_errors = raise_on_validation_errors
        self._parameters_allowed_types = tuple(set(
            six.string_types + six.integer_types + (six.text_type, float, list, tuple, dict, type(None))
        ))
        self._app_server = None
        self._files_server = None
        self._initial_iteration_offset = 0
        self._reload_skip_flag = False

        if not task_id:
            # generate a new task
            self.id = self._auto_generate(project_name=project_name, task_name=task_name, task_type=task_type)
            if self._offline_mode:
                self.data.id = self.id
                self.name = task_name
        else:
            # this is an existing task, let's try to verify stuff
            self._validate()

        if self.data is None:
            raise ValueError("Task ID \"{}\" could not be found".format(self.id))

        self._project_name = (self.project, project_name)

        if running_remotely() or DevWorker.report_stdout:
            log_to_backend = False
        self._log_to_backend = get_log_to_backend(default=log_to_backend)
        self._artifacts_manager = Artifacts(self)
        self._hyper_params_manager = HyperParams(self)

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
            # noinspection PyBroadException
            try:
                # check latest version
                from ...utilities.check_updates import CheckPackageUpdates
                latest_version = CheckPackageUpdates.check_new_package_available(only_once=True)
                if latest_version and not SUPPRESS_UPDATE_MESSAGE_ENV_VAR.get(
                        default=config.get('development.suppress_update_message', False)):
                    if not latest_version[1]:
                        sep = os.linesep
                        self.get_logger().report_text(
                            '{} new package available: UPGRADE to v{} is recommended!\nRelease Notes:\n{}'.format(
                                Session.get_clients()[0][0].upper(), latest_version[0], sep.join(latest_version[2])),
                        )
                    else:
                        self.get_logger().report_text(
                            'ClearML new version available: upgrade to v{} is recommended!'.format(
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
                filepaths=[self._calling_filename, sys.argv[0], ]
                if ScriptInfo.is_running_from_module() else [sys.argv[0], self._calling_filename, ],
                log=self.log, create_requirements=False,
                check_uncommitted=self._store_diff, uncommitted_from_remote=self._store_remote_diff
            )
            for msg in result.warning_messages:
                self.get_logger().report_text(msg)

            # if the git is too large to store on the task, we must store it as artifact:
            if result.auxiliary_git_diff:
                diff_preview = "# git diff too large to handle, storing as artifact. git diff summary:\n"
                diff_preview += '\n'.join(
                    line for line in result.auxiliary_git_diff.split('\n') if line.startswith('diff --git '))
                self._artifacts_manager.upload_artifact(
                    name='auxiliary_git_diff', artifact_object=result.auxiliary_git_diff,
                    preview=diff_preview,
                )

            # store original entry point
            entry_point = result.script.get('entry_point') if result.script else None

            # check if we are running inside a module, then we should set our entry point
            # to the module call including all argv's
            result.script = ScriptInfo.detect_running_module(result.script)

            # Since we might run asynchronously, don't use self.data (let someone else
            # overwrite it before we have a chance to call edit)
            with self._edit_lock:
                self.reload()
                self.data.script = result.script
                self._edit(script=result.script)

            # if jupyter is present, requirements will be created in the background, when saving a snapshot
            if result.script and script_requirements:
                entry_point_filename = None if config.get('development.force_analyze_entire_repo', False) else \
                    os.path.join(result.script['working_dir'], entry_point)
                if self._force_use_pip_freeze:
                    requirements, conda_requirements = pip_freeze(
                        combine_conda_with_pip=config.get('development.detect_with_conda_freeze', True))
                    requirements = '# Python ' + sys.version.replace('\n', ' ').replace('\r', ' ') + '\n\n'\
                                   + requirements
                else:
                    requirements, conda_requirements = script_requirements.get_requirements(
                        entry_point_filename=entry_point_filename)

                if requirements:
                    if not result.script['requirements']:
                        result.script['requirements'] = {}
                    result.script['requirements']['pip'] = requirements
                    result.script['requirements']['conda'] = conda_requirements

                self._update_requirements(result.script.get('requirements') or '')

            # we do not want to wait for the check version thread,
            # because someone might wait for us to finish the repo detection update
        except SystemExit:
            pass
        except Exception as e:
            get_logger('task').debug(str(e))

    def _auto_generate(self, project_name=None, task_name=None, task_type=TaskTypes.training):
        created_msg = make_message('Auto-generated at %(time)s UTC by %(user)s@%(host)s')

        if task_type.value not in (self.TaskTypes.training, self.TaskTypes.testing) and \
                not Session.check_min_api_version('2.8'):
            print('WARNING: Changing task type to "{}" : '
                  'clearml-server does not support task type "{}", '
                  'please upgrade clearml-server.'.format(self.TaskTypes.training, task_type.value))
            task_type = self.TaskTypes.training

        project_id = None
        if project_name:
            project_id = get_or_create_project(self, project_name)

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

        return res.response.id if res else 'offline-{}'.format(str(uuid4()).replace("-", ""))

    def _set_storage_uri(self, value):
        value = value.rstrip('/') if value else None
        self._storage_uri = StorageHelper.conform_url(value)
        self.data.output.destination = self._storage_uri
        self._edit(output_dest=self._storage_uri or ('' if Session.check_min_api_version('2.3') else None))

    @property
    def storage_uri(self):
        # type: () -> Optional[str]
        if self._storage_uri:
            return self._storage_uri
        if running_remotely():
            return self.data.output.destination
        else:
            return None

    @storage_uri.setter
    def storage_uri(self, value):
        # type: (str) -> ()
        self._set_storage_uri(value)

    @property
    def task_id(self):
        # type: () -> str
        return self.id

    @property
    def name(self):
        # type: () -> str
        return self.data.name or ''

    @name.setter
    def name(self, value):
        # type: (str) -> ()
        self.set_name(value)

    @property
    def task_type(self):
        # type: () -> str
        return self.data.type

    @property
    def project(self):
        # type: () -> str
        return self.data.project

    @property
    def parent(self):
        # type: () -> str
        return self.data.parent

    @property
    def input_models_id(self):
        # type: () -> Mapping[str, str]
        if not Session.check_min_api_version("2.13"):
            model_id = self._get_task_property('execution.model', raise_on_error=False)
            return {'Input Model': model_id} if model_id else {}

        input_models = self._get_task_property('models.input', default=[]) or []
        return {m.name: m.model for m in input_models}

    @property
    def output_models_id(self):
        # type: () -> Mapping[str, str]
        if not Session.check_min_api_version("2.13"):
            model_id = self._get_task_property('output.model', raise_on_error=False)
            return {'Output Model': model_id} if model_id else {}

        output_models = self._get_task_property('models.output', default=[]) or []
        return {m.name: m.model for m in output_models}

    @property
    def comment(self):
        # type: () -> str
        return self.data.comment or ''

    @comment.setter
    def comment(self, value):
        # type: (str) -> ()
        self.set_comment(value)

    @property
    def cache_dir(self):
        # type: () -> Path
        """ The cache directory which is used to store the Task related files. """
        return Path(get_cache_dir()) / self.id

    @property
    def status(self):
        # type: () -> str
        """
        The Task's status. To keep the Task updated.
        ClearML reloads the Task status information only, when this value is accessed.

        return str: TaskStatusEnum status
        """
        return self.get_status()

    @property
    def _status(self):
        # type: () -> str
        """ Return the task's cached status (don't reload if we don't have to) """
        return str(self.data.status)

    def reload(self):
        # type: () -> ()
        """
        Reload current Task's state from clearml-server.
        Refresh all task's fields, including artifacts / models / parameters etc.
        """
        return super(Task, self).reload()

    def _get_output_model(self, upload_required=True, model_id=None):
        # type: (bool, Optional[str]) -> Model
        return Model(
            session=self.session,
            model_id=model_id or None,
            cache_dir=self.cache_dir,
            upload_storage_uri=self.storage_uri or self.get_output_destination(
                raise_on_error=upload_required, log_on_error=upload_required),
            upload_storage_suffix=self._get_output_destination_suffix('models'),
            log=self.log)

    @property
    def metrics_manager(self):
        # type: () -> Metrics
        """ A metrics manager used to manage the metrics related to this task """
        return self._get_metrics_manager(self.get_output_destination())

    @property
    def _reporter(self):
        # type: () -> Reporter
        """
        Returns a simple metrics reporter instance.
        """
        if self.__reporter is None:
            self._setup_reporter()
        return self.__reporter

    def _get_metrics_manager(self, storage_uri):
        # type: (str) -> Metrics
        if self._metrics_manager is None:
            self._metrics_manager = Metrics(
                session=self.session,
                task=self,
                storage_uri=storage_uri,
                storage_uri_suffix=self._get_output_destination_suffix('metrics'),
                iteration_offset=self.get_initial_iteration()
            )
        return self._metrics_manager

    def _setup_reporter(self):
        # type: () -> Reporter
        try:
            storage_uri = self.get_output_destination(log_on_error=False)
        except ValueError:
            storage_uri = None
        self.__reporter = Reporter(
            metrics=self._get_metrics_manager(storage_uri=storage_uri), task=self)
        return self.__reporter

    def _get_output_destination_suffix(self, extra_path=None):
        # type: (Optional[str]) -> str
        return '/'.join(quote(x, safe="'[]{}()$^,.; -_+-=") for x in
                        (self.get_project_name(), '%s.%s' % (self.name, self.data.id), extra_path) if x)

    def _reload(self):
        # type: () -> Any
        """ Reload the task object from the backend """
        with self._edit_lock:
            if self._offline_mode:
                # noinspection PyBroadException
                try:
                    with open((self.get_offline_mode_folder() / self._offline_filename).as_posix(), 'rt') as f:
                        stored_dict = json.load(f)
                    stored_data = tasks.Task(**stored_dict)
                    # add missing entries
                    for k, v in stored_dict.items():
                        if not hasattr(stored_data, k):
                            setattr(stored_data, k, v)
                    if stored_dict.get('project_name'):
                        self._project_name = (None, stored_dict.get('project_name'))
                except Exception:
                    stored_data = self._data

                return stored_data or tasks.Task(
                    execution=tasks.Execution(
                        parameters={}, artifacts=[], dataviews=[], model='',
                        model_desc={}, model_labels={}, docker_cmd=''),
                    output=tasks.Output())

            if self._reload_skip_flag and self._data:
                return self._data
            res = self.send(tasks.GetByIdRequest(task=self.id))
            return res.response.task

    def reset(self, set_started_on_success=True):
        # type: (bool) -> ()
        """
        Reset the task. Task will be reloaded following a successful reset.

        :param  set_started_on_success: If True automatically set Task status to started after resetting it.
        """
        self.send(tasks.ResetRequest(task=self.id))
        if set_started_on_success:
            self.started()
        elif self._data:
            # if not started, make sure the current cached state is synced
            self._data.status = self.TaskStatusEnum.created

        self.reload()

    def started(self, ignore_errors=True, force=False):
        # type: (bool, bool) -> ()
        """ The signal that this Task started. """
        return self.send(tasks.StartedRequest(self.id, force=force), ignore_errors=ignore_errors)

    def stopped(self, ignore_errors=True, force=False):
        # type: (bool, bool) -> ()
        """ The signal that this Task stopped. """
        return self.send(tasks.StoppedRequest(self.id, force=force), ignore_errors=ignore_errors)

    def completed(self, ignore_errors=True):
        # type: (bool) -> ()
        """ The signal indicating that this Task completed. """
        if hasattr(tasks, 'CompletedRequest') and callable(tasks.CompletedRequest):
            return self.send(tasks.CompletedRequest(self.id, status_reason='completed'), ignore_errors=ignore_errors)
        return self.send(tasks.StoppedRequest(self.id, status_reason='completed'), ignore_errors=ignore_errors)

    def mark_failed(self, ignore_errors=True, status_reason=None, status_message=None, force=False):
        # type: (bool, Optional[str], Optional[str], bool) -> ()
        """ The signal that this Task stopped. """
        return self.send(
            tasks.FailedRequest(
                task=self.id, status_reason=status_reason, status_message=status_message, force=force),
            ignore_errors=ignore_errors,
        )

    def publish(self, ignore_errors=True):
        # type: (bool) -> ()
        """ The signal that this Task will be published """
        if str(self.status) not in (str(tasks.TaskStatusEnum.stopped), str(tasks.TaskStatusEnum.completed)):
            raise ValueError("Can't publish, Task is not stopped")
        resp = self.send(tasks.PublishRequest(self.id), ignore_errors=ignore_errors)
        assert isinstance(resp.response, tasks.PublishResponse)
        return resp

    def _delete(
        self,
        delete_artifacts_and_models=True,
        skip_models_used_by_other_tasks=True,
        raise_on_error=False,
    ):
        # type: (bool, bool, bool) -> bool
        """
        Delete the task as well as it's output models and artifacts.
        Models and artifacts are deleted from their storage locations, each using its URI.

        Note: in order to delete models and artifacts using their URI, make sure the proper storage credentials are
        configured in your configuration file (e.g. if an artifact is stored in S3, make sure sdk.aws.s3.credentials
        are properly configured and that you have delete permission in the related buckets).

        :param delete_artifacts_and_models: If True, artifacts and models would also be deleted (default True)
        :param skip_models_used_by_other_tasks: If True, models used by other tasks would not be deleted (default True)
        :param raise_on_error: If True an exception will be raised when encountering an error.
                               If False an error would be printed and no exception will be raised.
        :return: True if the task was deleted successfully.
        """

        try:
            res = self.send(tasks.GetByIdRequest(self.task_id))
            task = res.response.task
            if task.status == Task.TaskStatusEnum.published:
                if raise_on_error:
                    raise self.DeleteError("Cannot delete published task {}".format(self.task_id))
                self.log.error("Cannot delete published task {}".format(self.task_id))
                return False

            execution = {}
            models_res = []
            if delete_artifacts_and_models:
                execution = task.execution.to_dict() if task.execution else {}
                models_res = self.send(
                    models.GetAllRequest(
                        task=[task.id], only_fields=["id", "uri"]
                    )
                ).response.models

            event_uris = list(self._get_all_events(
                event_type="training_debug_image", unique_selector=itemgetter("url"), batch_size=10000
            ))
            event_uris.extend(self._get_image_plot_uris())

            task_deleted = self.send(tasks.DeleteRequest(self.task_id, force=True))
            if not task_deleted:
                if raise_on_error:
                    raise self.DeleteError("Failed deleting task {}".format(self.task_id))
                self.log.error("Failed deleting task {}".format(self.task_id))
                return False

        except self.DeleteError:
            raise
        except Exception as ex:
            if raise_on_error:
                raise self.DeleteError("Task deletion failed: {}".format(ex))
            self.log.error("Task deletion failed: {}".format(ex))
            return False

        failures = []
        if delete_artifacts_and_models:
            for e in execution["artifacts"]:
                if e["mode"] == "output" and not self._delete_uri(e["uri"]):
                    failures.append(e["uri"])

            for m in models_res:
                # noinspection PyBroadException
                try:
                    is_output_model = task.output and (m.id == task.output.model)
                    res = self.send(
                        models.DeleteRequest(m.id, force=(not skip_models_used_by_other_tasks)),
                        ignore_errors=is_output_model
                    )
                    # Should delete if model was deleted or if this was the output model (which was already deleted
                    # by DeleteRequest, and it's URI is dangling
                    should_delete = is_output_model or res.response.deleted
                except SendError as ex:
                    if (ex.result.meta.result_code, ex.result.meta.result_subcode) == (400, 201):
                        # Model not found, already deleted by DeleteRequest
                        should_delete = True
                    else:
                        failures.append("model id: {}".format(m.id))
                        continue
                except Exception:
                    failures.append("model id: {}".format(m.id))
                    continue
                if should_delete and not self._delete_uri(m.uri):
                    failures.append(m.uri)

        event_uris = list(filter(None, event_uris))
        for uri in event_uris:
            if not self._delete_uri(uri):
                failures.append(uri)

        failures = list(filter(None, failures))
        if len(failures):
            error = "Failed deleting the following URIs:\n{}".format(
                "\n".join(failures)
            )
            if raise_on_error:
                raise self.DeleteError(error)
            self.log.error(error)

        return task_deleted

    def _delete_uri(self, uri):
        # type: (str) -> bool
        # noinspection PyBroadException
        try:
            deleted = StorageHelper.get(uri).delete(uri)
            if deleted:
                self.log.debug("Deleted file: {}".format(uri))
                return True
        except Exception as ex:
            self.log.error("Failed deleting {}: {}".format(uri, str(ex)))
            return False
        return False

    def _get_image_plot_uris(self):
        # type: () -> Set[str]

        def image_source_selector(d):
            plot = d.get("plot_str")
            if plot:
                # noinspection PyBroadException
                try:
                    plot = json.loads(plot)
                    return next(
                        filter(None, (image.get("source") for image in plot.get("layout", {}).get("images", []))),
                        None
                    )
                except Exception:
                    pass

        return self._get_all_events(
            event_type="plot",
            unique_selector=image_source_selector,
            batch_size=10000
        )

    def update_model_desc(self, new_model_desc_file=None):
        # type: (Optional[str]) -> ()
        """ Change the Task's model description. """
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

    def update_output_model(
            self,
            model_path,  # type: str
            name=None,  # type: Optional[str]
            comment=None,  # type: Optional[str]
            tags=None,  # type: Optional[Sequence[str]]
            model_name=None,  # type: Optional[str]
            iteration=None,  # type: Optional[int]
    ):
        # type: (...) -> str
        """
        Update the Task's output model weights file. First, ClearML uploads the file to the preconfigured output
        destination (see the Task's ``output.destination`` property or call the ``setup_upload`` method),
        then ClearML updates the model object associated with the Task an API call. The API call uses with the URI
        of the uploaded file, and other values provided by additional arguments.

        :param model_path: A local weights file or folder to be uploaded.
            If remote URI is provided (e.g. http:// or s3: // etc) then the URI is stored as is, without any upload
        :param name: The updated model name.
            If not provided, the name is the model weights file filename without the extension.
        :param comment: The updated model description. (Optional)
        :param tags: The updated model tags. (Optional)
        :param model_name: If provided the model name as it will appear in the model artifactory. (Optional)
            Default: Task.name - name
        :param iteration: iteration number for the current stored model (Optional)

        :return: The URI of the uploaded weights file.
            Notice: upload is done is a background thread, while the function call returns immediately
        """
        from ...model import OutputModel
        output_model = OutputModel(
            task=self,
            name=model_name or ('{} - {}'.format(self.name, name) if name else self.name),
            tags=tags,
            comment=comment
        )
        output_model.connect(task=self, name=name)
        url = output_model.update_weights(weights_filename=model_path, iteration=iteration)
        return url

    @property
    def labels_stats(self):
        # type: () -> dict
        """ Get accumulated label stats for the current/last frames iteration """
        return self._curr_label_stats

    def _accumulate_label_stats(self, roi_stats, reset=False):
        # type: (dict, bool) -> ()
        if reset:
            self._curr_label_stats = {}
        for label in roi_stats:
            if label in self._curr_label_stats:
                self._curr_label_stats[label] += roi_stats[label]
            else:
                self._curr_label_stats[label] = roi_stats[label]

    def set_input_model(
            self,
            model_id=None,
            model_name=None,
            update_task_design=True,
            update_task_labels=True,
            name=None
    ):
        # type: (str, Optional[str], bool, bool, Optional[str]) -> ()
        """
        Set a new input model for the Task. The model must be "ready" (status is ``Published``) to be used as the
        Task's input model.

        :param model_id: The Id of the model on the **ClearML Server** (backend). If ``model_name`` is not specified,
            then ``model_id`` must be specified.
        :param model_name: The model name in the artifactory. The model_name is used to locate an existing model
            in the **ClearML Server** (backend). If ``model_id`` is not specified,
            then ``model_name`` must be specified.
        :param update_task_design: Update the Task's design

            - ``True`` - ClearML copies the Task's model design from the input model.
            - ``False`` - ClearML does not copy the Task's model design from the input model.

        :param update_task_labels: Update the Task's label enumeration

            - ``True`` - ClearML copies the Task's label enumeration from the input model.
            - ``False`` - ClearML does not copy the Task's label enumeration from the input model.

        :param name: Model section name to be stored on the Task (unrelated to the model object name itself)
            Default: the the model weight filename is used (excluding file extension)
        """
        if model_id is None and not model_name:
            raise ValueError('Expected one of [model_id, model_name]')

        if model_name and not model_id:
            # Try getting the model by name. Limit to 10 results.
            res = self.send(
                models.GetAllRequest(
                    name=exact_match_regex(model_name),
                    ready=True,
                    page=0,
                    page_size=10,
                    order_by=['-created'],
                    only_fields=['id', 'created', 'uri']
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
            name = name or Path(model.uri).stem
        else:
            # clear the input model
            model = None
            model_id = ''
            name = name or 'Input Model'

        with self._edit_lock:
            self.reload()
            # store model id
            if Session.check_min_api_version("2.13"):
                self.send(tasks.AddOrUpdateModelRequest(
                    task=self.id, name=name, model=model_id, type=tasks.ModelTypeEnum.input
                ))
            else:
                # backwards compatibility
                self._set_task_property("execution.model", model_id, raise_on_error=False, log_on_error=False)

            # Auto populate from model, if empty
            if update_task_labels and not self.data.execution.model_labels:
                self.data.execution.model_labels = model.labels if model else {}

            self._edit(execution=self.data.execution)

    def get_parameters(self, backwards_compatibility=True):
        # type: (bool) -> (Optional[dict])
        """
        Get the parameters for a Task. This method returns a complete group of key-value parameter pairs, but does not
        support parameter descriptions (the result is a dictionary of key-value pairs).
        Notice the returned parameter dict is flat:
        i.e. {'Args/param': 'value'} is the argument "param" from section "Args"

        :param backwards_compatibility: If True (default) parameters without section name
            (API version < 2.9, clearml-server < 0.16) will be at dict root level.
            If False, parameters without section name, will be nested under "Args/" key.

        :return: dict of the task parameters, all flattened to key/value.
            Different sections with key prefix "section/"
        """
        if not Session.check_min_api_version('2.9'):
            return self._get_task_property('execution.parameters')

        # API will makes sure we get old parameters with type legacy on top level (instead of nested in Args)
        parameters = dict()
        hyperparams = self._get_task_property('hyperparams') or {}
        if not backwards_compatibility:
            for section in hyperparams:
                for key, section_param in hyperparams[section].items():
                    parameters['{}/{}'.format(section, key)] = section_param.value
        else:
            for section in hyperparams:
                for key, section_param in hyperparams[section].items():
                    if section_param.type == 'legacy' and section in (self._legacy_parameters_section_name, ):
                        parameters['{}'.format(key)] = section_param.value
                    else:
                        parameters['{}/{}'.format(section, key)] = section_param.value

        return parameters

    def set_parameters(self, *args, **kwargs):
        # type: (*dict, **Any) -> ()
        """
        Set the parameters for a Task. This method sets a complete group of key-value parameter pairs, but does not
        support parameter descriptions (the input is a dictionary of key-value pairs).
        Notice the parameter dict is flat:
        i.e. {'Args/param': 'value'} will set the argument "param" in section "Args" to "value"

        :param args: Positional arguments, which are one or more dictionary or (key, value) iterable. They are
            merged into a single key-value pair dictionary.
        :param kwargs: Key-value pairs, merged into the parameters dictionary created from ``args``.
        """
        return self._set_parameters(*args, __update=False, **kwargs)

    def _set_parameters(self, *args, **kwargs):
        # type: (*dict, **Any) -> ()
        """
        Set the parameters for a Task. This method sets a complete group of key-value parameter pairs, but does not
        support parameter descriptions (the input is a dictionary of key-value pairs).

        :param args: Positional arguments, which are one or more dictionary or (key, value) iterable. They are
            merged into a single key-value pair dictionary.
        :param kwargs: Key-value pairs, merged into the parameters dictionary created from ``args``.
        """
        def stringify(value):
            # return empty string if value is None
            if value is None:
                return ""

            str_value = str(value)
            if isinstance(value, (tuple, list, dict)) and 'None' in re.split(r'[ ,\[\]{}()]', str_value):
                # If we have None in the string we have to use json to replace it with null,
                # otherwise we end up with None as string when running remotely
                try:
                    str_json = json.dumps(value)
                    # verify we actually have a null in the string, otherwise prefer the str cast
                    # This is because we prefer to have \' as in str and not \" used in json
                    if 'null' in re.split(r'[ ,\[\]{}()]', str_json):
                        return str_json
                except TypeError:
                    # if we somehow failed to json serialize, revert to previous std casting
                    pass
            return str_value

        if not all(isinstance(x, (dict, Iterable)) for x in args):
            raise ValueError('only dict or iterable are supported as positional arguments')

        prefix = kwargs.pop('__parameters_prefix', None)
        descriptions = kwargs.pop('__parameters_descriptions', None) or dict()
        params_types = kwargs.pop('__parameters_types', None) or dict()
        update = kwargs.pop('__update', False)

        # new parameters dict
        new_parameters = dict(itertools.chain.from_iterable(x.items() if isinstance(x, dict) else x for x in args))
        new_parameters.update(kwargs)
        if prefix:
            prefix = prefix.strip('/')
            new_parameters = dict(('{}/{}'.format(prefix, k), v) for k, v in new_parameters.items())

        # verify parameters type:
        not_allowed = {
            k: type(v).__name__
            for k, v in new_parameters.items()
            if not verify_basic_type(v, self._parameters_allowed_types)
        }
        if not_allowed:
            self.log.warning(
                "Skipping parameter: {}, only builtin types are supported ({})".format(
                    ', '.join('%s[%s]' % p for p in not_allowed.items()),
                    ', '.join(t.__name__ for t in self._parameters_allowed_types))
            )
            new_parameters = {k: v for k, v in new_parameters.items() if k not in not_allowed}

        use_hyperparams = Session.check_min_api_version('2.9')

        with self._edit_lock:
            self.reload()
            # if we have a specific prefix and we use hyperparameters, and we use set.
            # overwrite only the prefix, leave the rest as is.
            if not update and prefix:
                parameters = copy(self.get_parameters() or {})
                parameters = dict((k, v) for k, v in parameters.items() if not k.startswith(prefix+'/'))
            elif update:
                parameters = copy(self.get_parameters() or {})
            else:
                parameters = dict()

            parameters.update(new_parameters)

            # force cast all variables to strings (so that we can later edit them in UI)
            parameters = {k: stringify(v) for k, v in parameters.items()}

            if use_hyperparams:
                # build nested dict from flat parameters dict:
                org_hyperparams = self.data.hyperparams or {}
                hyperparams = dict()
                # if the task is a legacy task, we should put everything back under Args/key with legacy type
                legacy_name = self._legacy_parameters_section_name
                org_legacy_section = org_hyperparams.get(legacy_name, dict())

                for k, v in parameters.items():
                    # legacy variable
                    if org_legacy_section.get(k, tasks.ParamsItem()).type == 'legacy':
                        section = hyperparams.get(legacy_name, dict())
                        section[k] = copy(org_legacy_section[k])
                        section[k].value = str(v) if v else v
                        description = descriptions.get(k)
                        if description:
                            section[k].description = description
                        hyperparams[legacy_name] = section
                        continue

                    org_k = k
                    if '/' not in k:
                        k = '{}/{}'.format(self._default_configuration_section_name, k)
                    section_name, key = k.split('/', 1)
                    section = hyperparams.get(section_name, dict())
                    org_param = org_hyperparams.get(section_name, dict()).get(key, tasks.ParamsItem())
                    param_type = params_types[org_k] if org_k in params_types else org_param.type
                    if param_type and not isinstance(param_type, str):
                        param_type = param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)

                    section[key] = tasks.ParamsItem(
                        section=section_name, name=key,
                        value=str(v) if v else v,
                        description=descriptions[org_k] if org_k in descriptions else org_param.description,
                        type=param_type,
                    )
                    hyperparams[section_name] = section

                self._edit(hyperparams=hyperparams)
                self.data.hyperparams = hyperparams
            else:
                execution = self.data.execution
                if execution is None:
                    execution = tasks.Execution(
                        parameters=parameters, artifacts=[], dataviews=[], model='',
                        model_desc={}, model_labels={}, docker_cmd='')
                else:
                    execution.parameters = parameters
                self._edit(execution=execution)

    def set_parameter(self, name, value, description=None, value_type=None):
        # type: (str, str, Optional[str], Optional[Any]) -> ()
        """
        Set a single Task parameter. This overrides any previous value for this parameter.

        :param name: The parameter name.
        :param value: The parameter value.
        :param description: The parameter description.
        :param value_type: The type of the parameters (cast to string and store)
        """
        if not Session.check_min_api_version('2.9'):
            # not supported yet
            description = None
            value_type = None

        self._set_parameters(
            {name: value}, __update=True,
            __parameters_descriptions={name: description},
            __parameters_types={name: value_type}
        )

    def get_parameter(self, name, default=None):
        # type: (str, Any) -> Any
        """
        Get a value for a parameter.

        :param name: Parameter name
        :param default: Default value
        :return: The Parameter value (or default value if parameter is not defined).
        """
        params = self.get_parameters()
        return params.get(name, default)

    def delete_parameter(self, name):
        # type: (str) -> bool
        """
        Delete a parameter byt it's full name Section/name.

        :param name: Parameter name in full, i.e. Section/name. For example, 'Args/batch_size'
        :return: True if the parameter was deleted successfully
        """
        if not Session.check_min_api_version('2.9'):
            raise ValueError(
                "Delete hyper-parameter is not supported by your clearml-server, "
                "upgrade to the latest version")

        with self._edit_lock:
            paramkey = tasks.ParamKey(section=name.split('/', 1)[0], name=name.split('/', 1)[1])
            res = self.send(tasks.DeleteHyperParamsRequest(
                task=self.id, hyperparams=[paramkey]), raise_on_errors=False)
            self.reload()

        return res.ok()

    def update_parameters(self, *args, **kwargs):
        # type: (*dict, **Any) -> ()
        """
        Update the parameters for a Task. This method updates a complete group of key-value parameter pairs, but does
        not support parameter descriptions (the input is a dictionary of key-value pairs).
        Notice the parameter dict is flat:
        i.e. {'Args/param': 'value'} will set the argument "param" in section "Args" to "value"

        :param args: Positional arguments, which are one or more dictionary or (key, value) iterable. They are
            merged into a single key-value pair dictionary.
        :param kwargs: Key-value pairs, merged into the parameters dictionary created from ``args``.
        """
        self._set_parameters(*args, __update=True, **kwargs)

    def set_model_label_enumeration(self, enumeration=None):
        # type: (Mapping[str, int]) -> ()
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
        # type: () -> ()
        if not DOCKER_IMAGE_ENV_VAR.exists() and not DOCKER_BASH_SETUP_ENV_VAR.exists():
            return
        self.set_base_docker(
            docker_cmd=DOCKER_IMAGE_ENV_VAR.get(default=""),
            docker_setup_bash_script=DOCKER_BASH_SETUP_ENV_VAR.get(default=""))

    def set_base_docker(self, docker_cmd, docker_arguments=None, docker_setup_bash_script=None):
        # type: (str, Optional[Union[str, Sequence[str]]], Optional[Union[str, Sequence[str]]]) -> ()
        """
        Set the base docker image for this experiment
        If provided, this value will be used by clearml-agent to execute this experiment
        inside the provided docker image.
        When running remotely the call is ignored

        :param docker_cmd: docker container image (example: 'nvidia/cuda:11.1')
        :param docker_arguments: docker execution parameters (example: '-e ENV=1')
        :param docker_setup_bash_script: bash script to run at the
            beginning of the docker before launching the Task itself. example: ['apt update', 'apt-get install -y gcc']
        """
        image = docker_cmd.split(' ')[0] if docker_cmd else ''
        if not docker_arguments and docker_cmd:
            docker_arguments = docker_cmd.split(' ')[1:] if len(docker_cmd.split(' ')) > 1 else ''

        arguments = (docker_arguments if isinstance(docker_arguments, str) else ' '.join(docker_arguments)) \
            if docker_arguments else ''

        if docker_setup_bash_script:
            setup_shell_script = docker_setup_bash_script \
                if isinstance(docker_setup_bash_script, str) else '\n'.join(docker_setup_bash_script)
        else:
            setup_shell_script = ''

        with self._edit_lock:
            self.reload()
            if Session.check_min_api_version("2.13"):
                self.data.container = dict(image=image, arguments=arguments, setup_shell_script=setup_shell_script)
                self._edit(container=self.data.container)
            else:
                if setup_shell_script:
                    raise ValueError(
                        "Your ClearML-server does not support docker bash script feature, please upgrade.")
                execution = self.data.execution
                execution.docker_cmd = image + (' {}'.format(arguments) if arguments else '')
                self._edit(execution=execution)

    def get_base_docker(self):
        # type: () -> str
        """Get the base Docker command (image) that is set for this experiment."""
        if Session.check_min_api_version("2.13"):
            # backwards compatibility
            container = self._get_task_property(
                "container", raise_on_error=False, log_on_error=False, default={})
            return (container.get('image', '') +
                    (' {}'.format(container['arguments']) if container.get('arguments', '') else '')) or None
        else:
            return self._get_task_property("execution.docker_cmd", raise_on_error=False, log_on_error=False)

    def set_artifacts(self, artifacts_list=None):
        # type: (Sequence[tasks.Artifact]) -> Optional[List[tasks.Artifact]]
        """
        List of artifacts (tasks.Artifact) to update the task

        :param list artifacts_list: list of artifacts (type tasks.Artifact)
        :return: List of current Task's Artifacts or None if error.
        """
        if not Session.check_min_api_version('2.3'):
            return None
        if not (isinstance(artifacts_list, (list, tuple))
                and all(isinstance(a, tasks.Artifact) for a in artifacts_list)):
            raise ValueError('Expected artifacts as List[tasks.Artifact]')
        with self._edit_lock:
            self.reload()
            execution = self.data.execution
            keys = [a.key for a in artifacts_list]
            execution.artifacts = [a for a in execution.artifacts or [] if a.key not in keys] + artifacts_list
            self._edit(execution=execution)
        return execution.artifacts or []

    def _add_artifacts(self, artifacts_list):
        # type: (Sequence[tasks.Artifact]) -> Optional[List[tasks.Artifact]]
        """
        List of artifacts (tasks.Artifact) to add to the the task
        If an artifact by the same name already exists it will overwrite the existing artifact.

        :param list artifacts_list: list of artifacts (type tasks.Artifact)
        :return: List of current Task's Artifacts
        """
        if not Session.check_min_api_version('2.3'):
            return None
        if not (isinstance(artifacts_list, (list, tuple))
                and all(isinstance(a, tasks.Artifact) for a in artifacts_list)):
            raise ValueError('Expected artifacts as List[tasks.Artifact]')

        with self._edit_lock:
            if Session.check_min_api_version("2.13") and not self._offline_mode:
                req = tasks.AddOrUpdateArtifactsRequest(task=self.task_id, artifacts=artifacts_list, force=True)
                res = self.send(req, raise_on_errors=False)
                if not res or not res.response or not res.response.updated:
                    return None
                self.reload()
            else:
                self.reload()
                execution = self.data.execution
                keys = [a.key for a in artifacts_list]
                execution.artifacts = [a for a in execution.artifacts or [] if a.key not in keys] + artifacts_list
                self._edit(execution=execution)
        return self.data.execution.artifacts or []

    def _set_model_design(self, design=None):
        # type: (str) -> ()
        with self._edit_lock:
            self.reload()
            if Session.check_min_api_version('2.9'):
                configuration = self._get_task_property(
                    "configuration", default={}, raise_on_error=False, log_on_error=False) or {}
                configuration[self._default_configuration_section_name] = tasks.ConfigurationItem(
                    name=self._default_configuration_section_name, value=str(design))
                self._edit(configuration=configuration)
            else:
                execution = self.data.execution
                if design is not None:
                    # noinspection PyProtectedMember
                    execution.model_desc = Model._wrap_design(design)

                self._edit(execution=execution)

    def get_labels_enumeration(self):
        # type: () -> Mapping[str, int]
        """
        Get the label enumeration dictionary label enumeration dictionary of string (label) to integer (value) pairs.

        :return: A dictionary containing the label enumeration.
        """
        if not self.data or not self.data.execution:
            return {}
        return self.data.execution.model_labels

    def get_model_design(self):
        # type: () -> str
        """
        Get the model configuration as blob of text.

        :return: The model configuration as blob of text.
        """
        if Session.check_min_api_version('2.9'):
            design = self._get_task_property(
                "configuration", default={}, raise_on_error=False, log_on_error=False) or {}
            if design:
                design = design.get(sorted(design.keys())[0]).value or ''
        else:
            design = self._get_task_property(
                "execution.model_desc", default={}, raise_on_error=False, log_on_error=False)

        # noinspection PyProtectedMember
        return Model._unwrap_design(design)

    def get_random_seed(self):
        # type: () -> int
        # fixed seed for the time being
        return 1337

    def set_random_seed(self, random_seed):
        # type: (int) -> ()
        # fixed seed for the time being
        pass

    def set_project(self, project_id=None, project_name=None):
        # type: (Optional[str], Optional[str]) -> ()

        # if running remotely and we are the main task, skip setting ourselves.
        if self._is_remote_main_task():
            return

        if not project_id:
            assert isinstance(project_name, six.string_types)
            res = self.send(projects.GetAllRequest(name=exact_match_regex(project_name)), raise_on_errors=False)
            if not res or not res.response or not res.response.projects or len(res.response.projects) != 1:
                return False
            project_id = res.response.projects[0].id

        assert isinstance(project_id, six.string_types)
        self._set_task_property("project", project_id)
        self._edit(project=project_id)

    def get_project_name(self):
        # type: () -> Optional[str]
        if self.project is None:
            return self._project_name[1] if self._project_name and len(self._project_name) > 1 else None

        if self._project_name and self._project_name[1] is not None and self._project_name[0] == self.project:
            return self._project_name[1]

        res = self.send(projects.GetByIdRequest(project=self.project), raise_on_errors=False)
        if not res or not res.response or not res.response.project:
            return None
        self._project_name = (self.project, res.response.project.name)
        return self._project_name[1]

    def get_tags(self):
        # type: () -> Sequence[str]
        return self._get_task_property("tags")

    def set_system_tags(self, tags):
        # type: (Sequence[str]) -> ()
        assert isinstance(tags, (list, tuple))
        if Session.check_min_api_version('2.3'):
            self._set_task_property("system_tags", tags)
            self._edit(system_tags=self.data.system_tags)
        else:
            self._set_task_property("tags", tags)
            self._edit(tags=self.data.tags)

    def get_system_tags(self):
        # type: () -> Sequence[str]
        return self._get_task_property("system_tags" if Session.check_min_api_version('2.3') else "tags")

    def set_tags(self, tags):
        # type: (Sequence[str]) -> ()
        assert isinstance(tags, (list, tuple))
        if not Session.check_min_api_version('2.3'):
            # not supported
            return
        self._set_task_property("tags", tags)
        self._edit(tags=self.data.tags)

    def set_name(self, name):
        # type: (str) -> ()
        """
        Set the Task name.

        :param name: The name of the Task.
        :type name: str
        """
        name = name or ''
        self._set_task_property("name", str(name))
        self._edit(name=self.data.name)

    def set_parent(self, parent):
        # type: (Optional[Union[str, Task]]) -> ()
        """
        Set the parent task for the Task.
        :param parent: The parent task id (or parent Task object) for the Task. Set None for no parent.
        :type parent: str or Task
        """
        if parent:
            assert isinstance(parent, (str, Task))
            if isinstance(parent, Task):
                parent = parent.parent
            assert parent != self.id
        self._set_task_property("parent", str(parent) if parent else None)
        self._edit(parent=self.data.parent)

    def set_comment(self, comment):
        # type: (str) -> ()
        """
        Set a comment / description for the Task.

        :param comment: The comment / description for the Task.
        :type comment: str
        """
        comment = comment or ''
        self._set_task_property("comment", str(comment))
        self._edit(comment=str(comment))

    def set_task_type(self, task_type):
        # type: (Union[str, Task.TaskTypes]) -> ()
        """
        Set the task_type for the Task.

        :param task_type: The task_type of the Task (see optional values in TaskTypes).
        :type task_type: str or TaskTypes
        """
        if not isinstance(task_type, self.TaskTypes):
            task_type = self.TaskTypes(task_type)

        self._set_task_property("task_type", str(task_type))
        self._edit(type=task_type)

    def set_archived(self, archive):
        # type: (bool) -> ()
        """
        Archive the Task or remove it from the archived folder.

        :param archive: If True archive the Task, If False make sure it is removed from the archived folder
        """
        with self._edit_lock:
            system_tags = list(set(self.get_system_tags()) | {self.archived_tag}) \
                if archive else list(set(self.get_system_tags()) - {self.archived_tag})
            self.set_system_tags(system_tags)

    def get_archived(self):
        # type: () -> bool
        """
        Return the Archive state of the Task

        :return: If True the Task is archived, otherwise it is not.
        """
        return self.archived_tag in self.get_system_tags()

    def set_initial_iteration(self, offset=0):
        # type: (int) -> int
        """
        Set the initial iteration offset. The default value is ``0``. This method is useful when continuing training
        from previous checkpoints.

        For example, to start on iteration 100000, including scalars and plots:

        ..code-block:: py

          task.set_initial_iteration(100000)

        Task.set_initial_iteration(100000)

        :param int offset: Initial iteration (at starting point)
        :return: A newly set initial offset.
        """
        if not isinstance(offset, int):
            raise ValueError("Initial iteration offset must be an integer")

        self._initial_iteration_offset = offset
        if self._metrics_manager:
            self._metrics_manager.set_iteration_offset(self._initial_iteration_offset)
        return self._initial_iteration_offset

    def get_initial_iteration(self):
        # type: () -> int
        """
        Get the initial iteration offset. The default value is ``0``. This method is useful when continuing training
        from previous checkpoints.

        :return: The initial iteration offset.
        """
        return self._initial_iteration_offset

    def get_status(self):
        # type: () -> str
        """
        Return The task status without refreshing the entire Task object object (only the status property)

        TaskStatusEnum: ["created", "in_progress", "stopped", "closed", "failed", "completed",
                         "queued", "published", "publishing", "unknown"]

        :return: str: Task status as string (TaskStatusEnum)
        """
        status = self._get_status()[0]
        if self._data:
            self._data.status = status
        return str(status)

    def get_output_log_web_page(self):
        # type: () -> str
        """
        Return the Task results & outputs web page address.
        For example: https://demoapp.demo.clear.ml/projects/216431/experiments/60763e04/output/log

        :return: http/s URL link.
        """
        return '{}/projects/{}/experiments/{}/output/log'.format(
            self._get_app_server(),
            self.project if self.project is not None else '*',
            self.id,
        )

    def get_reported_scalars(
            self,
            max_samples=0,  # type: int
            x_axis='iter'  # type: str
    ):
        # type: (...) -> Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]]
        """
        Return a nested dictionary for the scalar graphs,
        where the first key is the graph title and the second is the series name.
        Value is a dict with 'x': values and 'y': values

        .. note::
           This call is not cached, any call will retrieve all the scalar reports from the back-end.
           If the Task has many scalars reported, it might take long for the call to return.

        Example:

        .. code-block:: py

          {'title': {'series': {
                      'x': [0, 1 ,2],
                      'y': [10, 11 ,12],
          }}}

        :param int max_samples: Maximum samples per series to return. Default is 0 returning all scalars.
            With sample limit, average scalar values inside sampling window.
        :param str x_axis: scalar x_axis, possible values:
            'iter': iteration (default), 'timestamp': seconds from start, 'iso_time': absolute time
        :return: dict: Nested scalar graphs: dict[title(str), dict[series(str), dict[axis(str), list(float)]]]
        """
        if x_axis not in ('iter', 'timestamp', 'iso_time'):
            raise ValueError("Scalar x-axis supported values are: 'iter', 'timestamp', 'iso_time'")

        # send request
        res = self.send(
            events.ScalarMetricsIterHistogramRequest(
                task=self.id, key=x_axis, samples=max(1, max_samples) if max_samples else None),
            raise_on_errors=False,
            ignore_errors=True,
        )
        if not res:
            return {}
        response = res.wait()
        if not response.ok() or not response.response_data:
            return {}

        return response.response_data

    def get_reported_console_output(self, number_of_reports=1):
        # type: (int) -> Sequence[str]
        """
        Return a list of console outputs reported by the Task. Retrieved outputs are the most updated console outputs.

        :param int number_of_reports: The number of reports to return. The default value is ``1``, indicating the
            last (most updated) console output
        :return: A list of strings, each entry corresponds to one report.
        """
        if Session.check_min_api_version('2.9'):
            request = events.GetTaskLogRequest(
                task=self.id,
                order='asc',
                navigate_earlier=True,
                batch_size=number_of_reports)
        else:
            request = events.GetTaskLogRequest(
                task=self.id,
                order='asc',
                from_='tail',
                batch_size=number_of_reports)
        res = self.send(request)
        response = res.wait()
        if not response.ok() or not response.response_data.get('events'):
            return []

        lines = [r.get('msg', '') for r in response.response_data['events']]
        return lines

    def get_configuration_object(self, name):
        # type: (str) -> Optional[str]
        """
        Get the Task's configuration object section as a blob of text
        Use only for automation (externally), otherwise use `Task.connect_configuration`.

        :param str name: Configuration section name
        :return: The Task's configuration as a text blob (unconstrained text string)
            return None if configuration name is not valid
        """
        return self._get_configuration_text(name)

    def get_configuration_objects(self):
        # type: () -> Optional[Mapping[str, str]]
        """
        Get the Task's configuration object section as a blob of text
        Use only for automation (externally), otherwise use `Task.connect_configuration`.

        :return: The Task's configurations as a
        dict (config name as key) and text blob as value (unconstrained text string)
        """
        if not Session.check_min_api_version('2.9'):
            raise ValueError(
                "Multiple configurations are not supported with the current 'clearml-server', "
                "please upgrade to the latest version")

        configuration = self.data.configuration or {}
        return {k: v.value for k, v in configuration.items()}

    def set_configuration_object(self, name, config_text=None, description=None, config_type=None):
        # type: (str, Optional[str], Optional[str], Optional[str]) -> None
        """
        Set the Task's configuration object as a blob of text.
        Use only for automation (externally), otherwise use `Task.connect_configuration`.

        :param str name: Configuration section name
        :param config_text: configuration as a blob of text (unconstrained text string)
            usually the content of a configuration file of a sort
        :param str description: Configuration section description
        :param str config_type: Optional configuration format type
        """
        return self._set_configuration(
            name=name, description=description, config_type=config_type, config_text=config_text)

    @classmethod
    def get_projects(cls):
        # type: () -> (List['projects.Project'])
        """
        Return a list of projects in the system, sorted by last updated time

        :return: A list of all the projects in the system. Each entry is a `services.projects.Project` object.
        """
        res = cls._send(
            cls._get_default_session(),
            projects.GetAllRequest(order_by=['last_update']), raise_on_errors=True)
        if res and res.response and res.response.projects:
            return [projects.Project(**p.to_dict()) for p in res.response.projects]
        return []

    @classmethod
    def get_project_id(cls, project_name):
        # type: (str) -> Optional[str]
        """
        Return a the project unique id (str).
        If for than one project match the project_name, return the last updated project
        If no project matched the requested name, returns None

        :return: Project unique ID (str), or None if no project was found.
        """
        assert project_name
        assert isinstance(project_name, str)
        res = cls._send(
            cls._get_default_session(),
            projects.GetAllRequest(order_by=['last_update'], name=exact_match_regex(project_name)),
            raise_on_errors=False)
        if res and res.response and res.response.projects:
            return [projects.Project(**p.to_dict()).id for p in res.response.projects][0]
        return None

    @staticmethod
    def running_locally():
        # type: () -> bool
        """
        Is the task running locally (i.e., ``clearml-agent`` is not executing it)

        :return: True, if the task is running locally. False, if the task is not running locally.

        """
        return not running_remotely()

    @classmethod
    def add_requirements(cls, package_name, package_version=None):
        # type: (str, Optional[str]) -> None
        """
        Force the adding of a package to the requirements list. If ``package_version`` is None, use the
        installed package version, if found.
        Example: Task.add_requirements('tensorflow', '2.4.0')
        Example: Task.add_requirements('tensorflow', '>=2.4')
        Example: Task.add_requirements('tensorflow') -> use the installed tensorflow version
        Example: Task.add_requirements('tensorflow', '') -> no version limit

        :param str package_name: The package name to add to the "Installed Packages" section of the task.
        :param package_version: The package version requirements. If ``None``, then  use the installed version.
        """
        if not running_remotely() and hasattr(cls, 'current_task') and cls.current_task():
            get_logger('task').warning(
                'Requirement ignored, Task.add_requirements() must be called before Task.init()')
        cls._force_requirements[str(package_name)] = package_version

    @classmethod
    def ignore_requirements(cls, package_name):
        # type: (str) -> None
        """
        Ignore a specific package when auto generating the requirements list.
        Example: Task.ignore_requirements('pywin32')

        :param str package_name: The package name to remove/ignore from the "Installed Packages" section of the task.
        """
        if not running_remotely() and hasattr(cls, 'current_task') and cls.current_task():
            get_logger('task').warning(
                'Requirement ignored, Task.ignore_requirements() must be called before Task.init()')
        cls._ignore_requirements.add(str(package_name))

    @classmethod
    def force_requirements_env_freeze(cls, force=True):
        # type: (bool) -> None
        """
        Force using `pip freeze` / `conda list` to store the full requirements of the active environment
        (instead of statically analyzing the running code and listing directly imported packages)
        Notice: Must be called before `Task.init` !

        :param force: Set force using `pip freeze` flag on/off
        """
        cls._force_use_pip_freeze = bool(force)

    def _get_default_report_storage_uri(self):
        # type: () -> str
        if self._offline_mode:
            return str(self.get_offline_mode_folder() / 'data')

        if not self._files_server:
            self._files_server = Session.get_files_server_host()
        return self._files_server

    def _get_status(self):
        # type: () -> (Optional[str], Optional[str])
        if self._offline_mode:
            return tasks.TaskStatusEnum.created, 'offline'

        # noinspection PyBroadException
        try:
            all_tasks = self.send(
                tasks.GetAllRequest(id=[self.id], only_fields=['status', 'status_message']),
            ).response.tasks
            return all_tasks[0].status, all_tasks[0].status_message
        except Exception:
            return None, None

    def _get_last_update(self):
        # type: () -> (Optional[datetime])
        if self._offline_mode:
            return None

        # noinspection PyBroadException
        try:
            all_tasks = self.send(
                tasks.GetAllRequest(id=[self.id], only_fields=['last_update']),
            ).response.tasks
            return all_tasks[0].last_update
        except Exception:
            return None

    def _reload_last_iteration(self):
        # type: () -> ()
        # noinspection PyBroadException
        try:
            all_tasks = self.send(
                tasks.GetAllRequest(id=[self.id], only_fields=['last_iteration']),
            ).response.tasks
            self.data.last_iteration = all_tasks[0].last_iteration
        except Exception:
            return None

    def _clear_task(self, system_tags=None, comment=None):
        # type: (Optional[Sequence[str]], Optional[str]) -> ()
        self._data.script = tasks.Script(
            binary='', repository='', tag='', branch='', version_num='', entry_point='',
            working_dir='', requirements={}, diff='',
        )
        if Session.check_min_api_version("2.13"):
            self._data.models = tasks.TaskModels(input=[], output=[])
            self._data.container = dict()

        self._data.execution = tasks.Execution(
            artifacts=[], dataviews=[], model='', model_desc={}, model_labels={}, parameters={}, docker_cmd='')

        self._data.comment = str(comment)

        self._storage_uri = None
        self._data.output.destination = self._storage_uri

        self._update_requirements('')

        if Session.check_min_api_version('2.13'):
            self._set_task_property("system_tags", system_tags)
            self._edit(system_tags=self._data.system_tags, comment=self._data.comment,
                       script=self._data.script, execution=self._data.execution, output_dest='',
                       hyperparams=dict(), configuration=dict(),
                       container=self._data.container, models=self._data.models)
        elif Session.check_min_api_version('2.9'):
            self._set_task_property("system_tags", system_tags)
            self._edit(system_tags=self._data.system_tags, comment=self._data.comment,
                       script=self._data.script, execution=self._data.execution, output_dest='',
                       hyperparams=dict(), configuration=dict())
        elif Session.check_min_api_version('2.3'):
            self._set_task_property("system_tags", system_tags)
            self._edit(system_tags=self._data.system_tags, comment=self._data.comment,
                       script=self._data.script, execution=self._data.execution, output_dest='')
        else:
            self._set_task_property("tags", system_tags)
            self._edit(tags=self._data.tags, comment=self._data.comment,
                       script=self._data.script, execution=self._data.execution, output_dest=None)

    @classmethod
    def _get_api_server(cls):
        # type: () -> ()
        return Session.get_api_server_host()

    def _get_app_server(self):
        # type: () -> str
        if not self._app_server:
            self._app_server = Session.get_app_server_host()
        return self._app_server

    def _is_remote_main_task(self):
        # type: () -> bool
        """
        :return: return True if running remotely and this Task is the registered main task
        """
        return running_remotely() and get_remote_task_id() == self.id

    def _edit(self, **kwargs):
        # type: (**Any) -> Any
        with self._edit_lock:
            if self._offline_mode:
                for k, v in kwargs.items():
                    setattr(self.data, k, v)
                Path(self.get_offline_mode_folder()).mkdir(parents=True, exist_ok=True)
                with open((self.get_offline_mode_folder() / self._offline_filename).as_posix(), 'wt') as f:
                    export_data = self.data.to_dict()
                    export_data['project_name'] = self.get_project_name()
                    export_data['offline_folder'] = self.get_offline_mode_folder().as_posix()
                    json.dump(export_data, f, ensure_ascii=True, sort_keys=True)
                return None

            # Since we ae using forced update, make sure he task status is valid
            status = self._data.status if self._data and self._reload_skip_flag else self.data.status
            if status not in (tasks.TaskStatusEnum.created, tasks.TaskStatusEnum.in_progress):
                # the exception being name/comment that we can always change.
                if kwargs and all(k in ('name', 'comment', 'tags', 'system_tags', 'runtime') for k in kwargs.keys()):
                    pass
                else:
                    raise ValueError('Task object can only be updated if created or in_progress')

            res = self.send(tasks.EditRequest(task=self.id, force=True, **kwargs), raise_on_errors=False)
            return res

    def _update_requirements(self, requirements):
        # type: (Union[dict, str]) -> ()
        if not isinstance(requirements, dict):
            requirements = {'pip': requirements}

        # make sure we have str as values:
        for key in requirements.keys():
            if requirements[key] and not isinstance(requirements[key], str):
                requirements[key] = '\n'.join(requirements[key])

        # protection, Old API might not support it
        # noinspection PyBroadException
        try:
            with self._edit_lock:
                self.reload()
                self.data.script.requirements = requirements
                if self._offline_mode:
                    self._edit(script=self.data.script)
                else:
                    self.send(tasks.SetRequirementsRequest(task=self.id, requirements=requirements))
        except Exception:
            pass

    def _update_script(self, script):
        # type: (dict) -> ()
        with self._edit_lock:
            self.reload()
            self.data.script = script
            self._edit(script=script)

    def _set_configuration(self, name, description=None, config_type=None, config_text=None, config_dict=None):
        # type: (str, Optional[str], Optional[str], Optional[str], Optional[Mapping]) -> None
        """
        Set Task configuration text/dict. Multiple configurations are supported.

        :param str name: Configuration name.
        :param str description: Configuration section description.
        :param str config_type: Optional configuration format type (str).
        :param config_text: model configuration (unconstrained text string). usually the content
            of a configuration file. If `config_text` is not None, `config_dict` must not be provided.
        :param config_dict: model configuration parameters dictionary.
            If `config_dict` is not None, `config_text` must not be provided.
        """
        # make sure we have wither dict or text
        mutually_exclusive(config_dict=config_dict, config_text=config_text, _check_none=True)

        if not Session.check_min_api_version('2.9'):
            raise ValueError("Multiple configurations are not supported with the current 'clearml-server', "
                             "please upgrade to the latest version")

        if description:
            description = str(description)
        # support empty string
        a_config = config_dict_to_text(config_dict if config_text is None else config_text)
        with self._edit_lock:
            self.reload()
            configuration = self.data.configuration or {}
            configuration[name] = tasks.ConfigurationItem(
                name=name, value=a_config, description=description or None, type=config_type or None)
            self._edit(configuration=configuration)

    def _get_configuration_text(self, name):
        # type: (str) -> Optional[str]
        """
        Get Task configuration section as text

        :param str name: Configuration name.
        :return: The Task configuration as text (unconstrained text string).
            return None if configuration name is not valid.
        """
        if not Session.check_min_api_version('2.9'):
            raise ValueError("Multiple configurations are not supported with the current 'clearml-server', "
                             "please upgrade to the latest version")

        configuration = self.data.configuration or {}
        if not configuration.get(name):
            return None
        return configuration[name].value

    def _get_configuration_dict(self, name):
        # type: (str) -> Optional[dict]
        """
        Get Task configuration section as dictionary

        :param str name: Configuration name.
        :return: The Task configuration as dictionary.
            return None if configuration name is not valid.
        """
        config_text = self._get_configuration_text(name)
        if not config_text:
            return None
        return text_to_config_dict(config_text)

    def get_offline_mode_folder(self):
        # type: () -> (Optional[Path])
        """
        Return the folder where all the task outputs and logs are stored in the offline session.
        :return: Path object, local folder, later to be used with `report_offline_session()`
        """
        if not self._offline_mode:
            return None
        return get_offline_dir(task_id=self.task_id)

    @classmethod
    def _clone_task(
            cls,
            cloned_task_id,  # type: str
            name=None,   # type: Optional[str]
            comment=None,   # type: Optional[str]
            execution_overrides=None,   # type: Optional[dict]
            tags=None,   # type: Optional[Sequence[str]]
            parent=None,   # type: Optional[str]
            project=None,   # type: Optional[str]
            log=None,    # type: Optional[logging.Logger]
            session=None,    # type: Optional[Session]
    ):
        # type: (...) -> str
        """
        Clone a task

        :param str cloned_task_id: Task ID for the task to be cloned
        :param str name: New for the new task
        :param str comment: Optional comment for the new task
        :param dict execution_overrides: Task execution overrides. Applied over the cloned task's execution
            section, useful for overriding values in the cloned task.
        :param list tags: Optional updated model tags
        :param str parent: Optional parent Task ID of the new task.
        :param str project: Optional project ID of the new task.
            If None, the new task will inherit the cloned task's project.
        :param logging.Logger log: Log object used by the infrastructure.
        :param Session session: Session object used for sending requests to the API
        :return: The new tasks's ID.
        """

        session = session if session else cls._get_default_session()
        use_clone_api = Session.check_min_api_version('2.9')
        if use_clone_api:
            res = cls._send(
                session=session, log=log,
                req=tasks.CloneRequest(
                    task=cloned_task_id,
                    new_task_name=name,
                    new_task_tags=tags,
                    new_task_comment=comment,
                    new_task_parent=parent,
                    new_task_project=project,
                    execution_overrides=execution_overrides,
                )
            )
            cloned_task_id = res.response.id
            return cloned_task_id

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

        if not hasattr(task, 'system_tags') and not tags and task.tags:
            tags = [t for t in task.tags if t != cls._development_tag]

        extra = {}
        if hasattr(task, 'hyperparams'):
            extra['hyperparams'] = task.hyperparams
        if hasattr(task, 'configuration'):
            extra['configuration'] = task.configuration
        if getattr(task, 'system_tags', None):
            extra['system_tags'] = [t for t in task.system_tags if t not in (cls._development_tag, cls.archived_tag)]

        req = tasks.CreateRequest(
            name=name or task.name,
            type=task.type,
            input=task.input if hasattr(task, 'input') else {'view': {}},
            tags=tags,
            comment=comment if comment is not None else task.comment,
            parent=parent,
            project=project if project else task.project,
            output_dest=output_dest,
            execution=execution.as_plain_ordered_dict(),
            script=task.script,
            **extra
        )
        res = cls._send(session=session, log=log, req=req)
        cloned_task_id = res.response.id

        if task.script and task.script.requirements:
            cls._send(session=session, log=log, req=tasks.SetRequirementsRequest(
                task=cloned_task_id, requirements=task.script.requirements))
        return cloned_task_id

    @classmethod
    def get_all(cls, session=None, log=None, **kwargs):
        # type: (Optional[Session], Optional[logging.Logger], **Any) -> Any
        """
        List all the Tasks based on specific projection.

        :param Session session: The session object used for sending requests to the API.
        :param logging.Logger log: The Log object.
        :param kwargs: Keyword args passed to the GetAllRequest
            (see :class:`.backend_api.services.v2_5.tasks.GetAllRequest`)

            For example:

            .. code-block:: bash

               status='completed', 'search_text'='specific_word', 'user'='user_id', 'project'='project_id'

        :type kwargs: dict

        :return: The API response.
        """
        session = session if session else cls._get_default_session()
        req = tasks.GetAllRequest(**kwargs)
        res = cls._send(session=session, req=req, log=log)
        return res

    @classmethod
    def get_by_name(cls, task_name):
        # type: (str) -> Task
        res = cls._send(cls._get_default_session(), tasks.GetAllRequest(name=exact_match_regex(task_name)))

        task = get_single_result(entity='task', query=task_name, results=res.response.tasks)
        return cls(task_id=task.id)

    @classmethod
    def _get_project_name(cls, project_id):
        res = cls._send(cls._get_default_session(), projects.GetByIdRequest(project=project_id), raise_on_errors=False)
        if not res or not res.response or not res.response.project:
            return None
        return res.response.project.name

    def _get_all_events(
        self, max_events=100, batch_size=500, order='asc', event_type=None, unique_selector=itemgetter("url")
    ):
        # type: (int, int, str, str, Callable[[dict], Any]) -> Union[List[Any], Set[Any]]
        """
        Get a list of all reported events.

        Warning: Debug only. Do not use outside of testing.

        :param max_events: The maximum events the function will return. Pass None
            to return all the reported events.
        :param batch_size: The maximum number of events retrieved by each internal call performed by this method.
        :param order: Events order (by timestamp) - "asc" for ascending, "desc" for descending.
        :param event_type: Event type. Pass None to get all event types.
        :param unique_selector: If provided, used to select a value from each event, only a unique set of these
            values will be returned by this method.

        :return: A list of events from the task. If unique_selector was provided, a set of values selected from events
            of the task.
        """
        batch_size = max_events or batch_size

        log_events = self.send(events.GetTaskEventsRequest(
            task=self.id,
            order=order,
            batch_size=batch_size,
            event_type=event_type,
        ))

        returned_count = log_events.response.returned
        total_events = log_events.response.total
        scroll = log_events.response.scroll_id
        if unique_selector:
            events_list = set(map(unique_selector, log_events.response.events))
        else:
            events_list = log_events.response.events

        while returned_count < total_events and (max_events is None or len(events_list) < max_events):
            log_events = self.send(events.GetTaskEventsRequest(
                task=self.id,
                order=order,
                batch_size=batch_size,
                event_type=event_type,
                scroll_id=scroll,
            ))
            scroll = log_events.response.scroll_id
            returned_count += log_events.response.returned
            if unique_selector:
                events_list.update(log_events.response.events)
            else:
                events_list.extend(log_events.response.events)

        return events_list

    @property
    def _edit_lock(self):
        # type: () -> ()

        # skip the actual lock, this one-time lock will always enter
        # only used on shutdown process to avoid deadlocks
        if self.__edit_lock is False:
            return RLock()

        if self.__edit_lock:
            return self.__edit_lock
        if not PROC_MASTER_ID_ENV_VAR.get() or len(PROC_MASTER_ID_ENV_VAR.get().split(':')) < 2:
            self.__edit_lock = RLock()
        elif PROC_MASTER_ID_ENV_VAR.get().split(':')[1] == str(self.id):
            filename = os.path.join(gettempdir(), 'clearml_{}.lock'.format(self.id))
            # no need to remove previous file lock if we have a dead process, it will automatically release the lock.
            # # noinspection PyBroadException
            # try:
            #     os.unlink(filename)
            # except Exception:
            #     pass
            # create a new file based lock
            self.__edit_lock = FileRLock(filename=filename)
        else:
            self.__edit_lock = RLock()
        return self.__edit_lock

    @_edit_lock.setter
    def _edit_lock(self, value):
        # type: (RLock) -> ()
        self.__edit_lock = value

    @classmethod
    def __update_master_pid_task(cls, pid=None, task=None):
        # type: (Optional[int], Union[str, Task]) -> None
        pid = pid or os.getpid()
        if not task:
            PROC_MASTER_ID_ENV_VAR.set(str(pid) + ':')
        elif isinstance(task, str):
            PROC_MASTER_ID_ENV_VAR.set(str(pid) + ':' + task)
        else:
            # noinspection PyUnresolvedReferences
            PROC_MASTER_ID_ENV_VAR.set(str(pid) + ':' + str(task.id))
            # make sure we refresh the edit lock next time we need it,
            task._edit_lock = None

    @classmethod
    def __get_master_id_task_id(cls):
        # type: () -> Optional[str]
        master_task_id = PROC_MASTER_ID_ENV_VAR.get().split(':')
        # we could not find a task ID, revert to old stub behaviour
        if len(master_task_id) < 2 or not master_task_id[1]:
            return None
        return master_task_id[1]

    @classmethod
    def __get_master_process_id(cls):
        # type: () -> Optional[str]
        master_task_id = PROC_MASTER_ID_ENV_VAR.get().split(':')
        # we could not find a task ID, revert to old stub behaviour
        if len(master_task_id) < 2 or not master_task_id[1]:
            return None
        return master_task_id[0]

    @classmethod
    def __is_subprocess(cls):
        # type: () -> bool
        # notice this class function is called from Task.ExitHooks, do not rename/move it.
        is_subprocess = PROC_MASTER_ID_ENV_VAR.get() and \
            PROC_MASTER_ID_ENV_VAR.get().split(':')[0] != str(os.getpid())
        return is_subprocess

    @classmethod
    def set_offline(cls, offline_mode=False):
        # type: (bool) -> None
        """
        Set offline mode, where all data and logs are stored into local folder, for later transmission

        :param offline_mode: If True, offline-mode is turned on, and no communication to the backend is enabled.
        :return:
        """
        if not running_remotely():
            ENV_OFFLINE_MODE.set(offline_mode)
            InterfaceBase._offline_mode = bool(offline_mode)
            Session._offline_mode = bool(offline_mode)

    @classmethod
    def is_offline(cls):
        # type: () -> bool
        """
        Return offline-mode state, If in offline-mode, no communication to the backend is enabled.

        :return: boolean offline-mode state
        """
        return cls._offline_mode

    @classmethod
    def _get_task_status(cls, task_id):
        # type: (str) -> (Optional[str], Optional[str])
        if cls._offline_mode:
            return tasks.TaskStatusEnum.created, 'offline'

        # noinspection PyBroadException
        try:
            all_tasks = cls._get_default_session().send(
                tasks.GetAllRequest(id=[task_id], only_fields=['status', 'status_message']),
            ).response.tasks
            return all_tasks[0].status, all_tasks[0].status_message
        except Exception:
            return None, None
