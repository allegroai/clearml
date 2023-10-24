import hashlib
import json
import os
import subprocess
import sys
import tempfile
import warnings
from copy import deepcopy
from datetime import datetime
from logging import getLogger
from time import time, sleep
from typing import Optional, Mapping, Sequence, Any, Callable, Union

from pathlib2 import Path

from ..backend_api import Session
from ..backend_interface.util import get_or_create_project, exact_match_regex
from ..storage.util import hash_dict
from ..task import Task
from ..backend_api.services import tasks as tasks_service
from ..utilities.proxy_object import verify_basic_type, get_basic_type


logger = getLogger('clearml.automation.job')


class BaseJob(object):
    _job_hash_description = 'job_hash={}'
    _job_hash_property = 'pipeline_job_hash'
    _hashing_callback = None
    _last_batch_status_update_ts = 0

    def __init__(self):
        # type: () -> ()
        """
        Base Job is an abstract CLearML Job
        """
        self._is_cached_task = False
        self._worker = None
        self.task_parameter_override = None
        self.task = None
        self.task_started = False
        self._last_status_ts = 0
        self._last_status = None

    def get_metric(self, title, series):
        # type: (str, str) -> (float, float, float)
        """
        Retrieve a specific scalar metric from the running Task.

        :param str title: Graph title (metric)
        :param str series: Series on the specific graph (variant)
        :return: A tuple of min value, max value, last value
        """
        metrics, title, series, values = self.get_metric_req_params(title, series)

        res = self.task.send(
            tasks_service.GetAllRequest(
                id=[self.task.id],
                page=0,
                page_size=1,
                only_fields=['id', ] + metrics
            )
        )
        response = res.wait()

        return tuple(response.response_data['tasks'][0]['last_metrics'][title][series][v] for v in values)

    @staticmethod
    def get_metric_req_params(title, series):
        title = hashlib.md5(str(title).encode('utf-8')).hexdigest()
        series = hashlib.md5(str(series).encode('utf-8')).hexdigest()
        metric = 'last_metrics.{}.{}.'.format(title, series)
        values = ['min_value', 'max_value', 'value']
        metrics = [metric + v for v in values]
        return metrics, title, series, values

    def launch(self, queue_name=None):
        # type: (str) -> bool
        """
        Send Job for execution on the requested execution queue

        :param str queue_name:

        :return False if Task is not in "created" status (i.e. cannot be enqueued) or cannot be enqueued
        """
        if self._is_cached_task:
            return False
        try:
            Task.enqueue(task=self.task, queue_name=queue_name)
            return True
        except Exception as ex:
            logger.warning('Error enqueuing Task {} to {}: {}'.format(self.task, queue_name, ex))
        return False

    def abort(self):
        # type: () -> ()
        """
        Abort currently running job (can be called multiple times)
        """
        if not self.task or self._is_cached_task:
            return

        if self.task.status == Task.TaskStatusEnum.queued:
            Task.dequeue(self.task)

        elif self.task.status == Task.TaskStatusEnum.in_progress:
            try:
                self.task.stopped()
            except Exception as ex:
                logger.warning(ex)

    def elapsed(self):
        # type: () -> float
        """
        Return the time in seconds since job started. Return -1 if job is still pending

        :return: Seconds from start.
        """
        if not self.task_started and str(self.task.status) != Task.TaskStatusEnum.in_progress:
            return -1
        self.task_started = True
        if not self.task.data.started:
            self.task.reload()
            if not self.task.data.started:
                return -1
        return (datetime.now(tz=self.task.data.started.tzinfo) - self.task.data.started).total_seconds()

    def iterations(self):
        # type: () -> int
        """
        Return the last iteration value of the current job. -1 if job has not started yet

        :return: Task last iteration.
        """
        if not self.task_started and self.task.status != Task.TaskStatusEnum.in_progress:
            return -1
        self.task_started = True
        return self.task.get_last_iteration()

    def task_id(self):
        # type: () -> str
        """
        Return the Task id.

        :return: The Task ID.
        """
        return self.task.id

    def status(self, force=False):
        # type: (bool) -> str
        """
        Return the Job Task current status. Options are: "created", "queued", "in_progress", "stopped", "published",
        "publishing", "closed", "failed", "completed", "unknown".

        :param force: Force status update, otherwise, only refresh state every 1 sec

        :return: Task status Task.TaskStatusEnum in string.
        """
        if self._last_status and not force and time() - self._last_status_ts < 1.:
            return self._last_status

        self._last_status = self.task.status
        # update timestamp after api call status()
        self._last_status_ts = time()
        return self._last_status

    def status_message(self):
        # type: () -> str
        """
        Gets the status message of the task. Note that the message is updated only after `BaseJob.status()`
        is called

        :return: The status message of the corresponding task as a string
        """
        return str(self.task.data.status_message)

    @classmethod
    def update_status_batch(cls, jobs):
        # type: (Sequence[BaseJob]) -> ()
        """
        Update the status of jobs, in batch_size

        :param jobs: The jobs to update the status of
        """
        have_job_with_no_status = False
        id_map = {}
        for job in jobs:
            if not job.task:
                continue
            id_map[job.task.id] = job
            # noinspection PyProtectedMember
            if not job._last_status:
                have_job_with_no_status = True
        if not id_map or (time() - cls._last_batch_status_update_ts < 1 and not have_job_with_no_status):
            return
        # noinspection PyProtectedMember
        batch_status = Task._get_tasks_status(list(id_map.keys()))
        last_batch_update_ts = time()
        cls._last_batch_status_update_ts = last_batch_update_ts
        for status, message, task_id in batch_status:
            if not status or not task_id:
                continue
            # noinspection PyProtectedMember
            id_map[task_id]._last_status = status
            # noinspection PyProtectedMember
            id_map[task_id]._last_status_ts = last_batch_update_ts

    def wait(self, timeout=None, pool_period=30., aborted_nonresponsive_as_running=False):
        # type: (Optional[float], float, bool) -> bool
        """
        Wait until the task is fully executed (i.e., aborted/completed/failed)

        :param timeout: maximum time (minutes) to wait for Task to finish
        :param pool_period: check task status every pool_period seconds
        :param aborted_nonresponsive_as_running: (default: False) If True, ignore the stopped state if the backend
            non-responsive watchdog sets this Task to stopped. This scenario could happen if
            an instance running the job is killed without warning (e.g. spot instances)
        :return: True, if Task finished.
        """
        tic = time()
        while timeout is None or time() - tic < timeout * 60.:
            if self.is_stopped(aborted_nonresponsive_as_running=aborted_nonresponsive_as_running):
                return True
            sleep(pool_period)

        return self.is_stopped(aborted_nonresponsive_as_running=aborted_nonresponsive_as_running)

    def get_console_output(self, number_of_reports=1):
        # type: (int) -> Sequence[str]
        """
        Return a list of console outputs reported by the Task.
        Returned console outputs are retrieved from the most updated console outputs.

        :param int number_of_reports: number of reports to return, default 1, the last (most updated) console output
        :return: List of strings each entry corresponds to one report.
        """
        return self.task.get_reported_console_output(number_of_reports=number_of_reports)

    def worker(self):
        # type: () -> Optional[str]
        """
        Return the current worker ID executing this Job. If job is pending, returns None

        :return: ID of the worker executing / executed the job, or None if job is still pending.
        """
        if self.is_pending():
            return self._worker

        if self._worker is None:
            self.task.reload()
            self._worker = self.task.last_worker

        return self._worker

    def is_running(self):
        # type: () -> bool
        """
        Return True, if job is currently running (pending is considered False)

        :return: True, if the task is currently in progress.
        """
        return self.status() == Task.TaskStatusEnum.in_progress

    def is_stopped(self, aborted_nonresponsive_as_running=False):
        # type: (bool) -> bool
        """
        Return True, if job finished executing (for any reason)

        :param aborted_nonresponsive_as_running: (default: False) If True, ignore the stopped state if the backend
            non-responsive watchdog sets this Task to stopped. This scenario could happen if
            an instance running the job is killed without warning (e.g. spot instances)

        :return: True the task is currently one of these states, stopped / completed / failed / published.
        """
        task_status = self.status()
        # check if we are Not in any of the non-running states
        if task_status not in (Task.TaskStatusEnum.stopped, Task.TaskStatusEnum.completed,
                               Task.TaskStatusEnum.failed, Task.TaskStatusEnum.published):
            return False

        # notice the status update also refresh the "status_message" field on the Task

        # if we are stopped but the message says "non-responsive" it means for some reason the
        # Task's instance was killed, we should ignore it if requested because we assume someone will bring it back
        if aborted_nonresponsive_as_running and task_status == Task.TaskStatusEnum.stopped and \
                str(self.task.data.status_message).lower() == "forced stop (non-responsive)":
            # if we are here it means the state is "stopped" but we should ignore it
            # because the non-responsive watchdog set it. We assume someone (autoscaler) will relaunch it.
            return False
        else:
            # if we do not need to ignore the nonactive state, it means this Task stopped
            return True

    def is_failed(self):
        # type: () -> bool
        """
        Return True, if job is has executed and failed

        :return: True the task is currently in failed state
        """
        return self.status() in (Task.TaskStatusEnum.failed, )

    def is_completed(self):
        # type: () -> bool
        """
        Return True, if job is has executed and completed successfully

        :return: True the task is currently in completed or published state
        """
        return self.status() in (Task.TaskStatusEnum.completed, Task.TaskStatusEnum.published)

    def is_aborted(self):
        # type: () -> bool
        """
        Return True, if job is has executed and aborted

        :return: True the task is currently in aborted state
        """
        return self.status() in (Task.TaskStatusEnum.stopped, )

    def is_pending(self):
        # type: () -> bool
        """
        Return True, if job is waiting for execution

        :return: True if the task is currently queued.
        """
        return self.status() in (Task.TaskStatusEnum.queued, Task.TaskStatusEnum.created)

    def started(self):
        # type: () -> bool
        """
        Return True, if job already started, or ended. False, if created/pending.

        :return: False, if the task is currently in draft mode or pending.
        """
        if not self.task_started and self.task.status in (
                Task.TaskStatusEnum.in_progress, Task.TaskStatusEnum.created):
            return False

        self.task_started = True
        return True

    def delete(self):
        # type: () -> bool
        """
        Delete the current temporary job (before launching)
        Return False if the Job/Task could not deleted
        """
        if not self.task or self._is_cached_task:
            return False

        if self.task.delete():
            self.task = None
            return True

        return False

    def is_cached_task(self):
        # type: () -> bool
        """
        :return: True if the internal Task is a cached one, False otherwise.
        """
        return self._is_cached_task

    @classmethod
    def register_hashing_callback(cls, a_function):
        # type: (Callable[[dict], dict]) -> None
        """
        Allow to customize the dict used for hashing the Task.
        Provided function will be called with a dict representing a Task,
        allowing to return a modified version of the representation dict.

        :param a_function:  Function manipulating the representation dict of a function
        """
        assert callable(a_function)
        cls._hashing_callback = a_function

    @classmethod
    def _create_task_hash(
            cls,
            task,
            section_overrides=None,
            params_override=None,
            configurations_override=None,
            explicit_docker_image=None
    ):
        # type: (Task, Optional[dict], Optional[dict], Optional[dict], Optional[str]) -> Optional[str]
        """
        Create Hash (str) representing the state of the Task

        :param task: A Task to hash
        :param section_overrides: optional dict (keys are Task's section names) with task overrides.
        :param params_override: Alternative to the entire Task's hyper parameters section
        (notice this should not be a nested dict but a flat key/value)
        :param configurations_override: dictionary of configuration override objects (tasks.ConfigurationItem)
        :param explicit_docker_image: The explicit docker image. Used to invalidate the hash when the docker image
            was explicitly changed

        :return: str hash of the Task configuration
        """
        if not task:
            return None
        if section_overrides and section_overrides.get("script"):
            script = section_overrides["script"]
            if not isinstance(script, dict):
                script = script.to_dict()
        else:
            script = task.data.script.to_dict() if task.data.script else {}

        # if we have a repository, we must make sure we have a specific version_num to ensure consistency
        if script.get("repository") and not script.get("version_num") and not script.get("tag"):
            return None

        # we need to ignore `requirements` section because ir might be changing from run to run
        script = deepcopy(script)
        script.pop("requirements", None)

        hyper_params = deepcopy(task.get_parameters() if params_override is None else params_override)
        hyper_params_to_change = {}
        task_cache = {}
        for key, value in hyper_params.items():
            if key.startswith("kwargs_artifacts/"):
                # noinspection PyBroadException
                try:
                    # key format is <task_id>.<artifact_name>
                    task_id, artifact = value.split(".", 1)
                    task_ = task_cache.setdefault(task_id, Task.get_task(task_id))
                    # set the value of the hyper parameter to the hash of the artifact
                    # because the task ID might differ, but the artifact might be the same
                    hyper_params_to_change[key] = task_.artifacts[artifact].hash
                except Exception:
                    pass
        hyper_params.update(hyper_params_to_change)
        configs = task.get_configuration_objects() if configurations_override is None else configurations_override
        # currently we do not add the docker image to the hash (only args and setup script),
        # because default docker image will cause the step to change
        docker = None
        if hasattr(task.data, "container"):
            docker = dict(**(task.data.container or dict()))
            docker.pop("image", None)
            if explicit_docker_image:
                docker["image"] = explicit_docker_image

        hash_func = "md5" if Session.check_min_api_version("2.13") else "crc32"

        # make sure that if we only have docker args/bash,
        # we use encode it, otherwise we revert to the original encoding (excluding docker altogether)
        repr_dict = dict(script=script, hyper_params=hyper_params, configs=configs)
        if docker:
            repr_dict["docker"] = docker

        # callback for modifying the representation dict
        if cls._hashing_callback:
            repr_dict = cls._hashing_callback(deepcopy(repr_dict))

        return hash_dict(repr_dict, hash_func=hash_func)

    @classmethod
    def _get_cached_task(cls, task_hash):
        # type: (str) -> Optional[Task]
        """

        :param task_hash:
        :return: A task matching the requested task hash
        """
        if not task_hash:
            return None
        if Session.check_min_api_version('2.13'):
            # noinspection PyProtectedMember
            potential_tasks = Task._query_tasks(
                status=['completed', 'published'],
                system_tags=['-{}'.format(Task.archived_tag)],
                _all_=dict(fields=['runtime.{}'.format(cls._job_hash_property)],
                           pattern=exact_match_regex(task_hash)),
                only_fields=['id'],
            )
        else:
            # noinspection PyProtectedMember
            potential_tasks = Task._query_tasks(
                status=['completed', 'published'],
                system_tags=['-{}'.format(Task.archived_tag)],
                _all_=dict(fields=['comment'], pattern=cls._job_hash_description.format(task_hash)),
                only_fields=['id'],
            )
        for obj in potential_tasks:
            task = Task.get_task(task_id=obj.id)
            return task
        return None

    @classmethod
    def _set_task_cache_hash(cls, task, task_hash=None):
        # type: (Task, Optional[str]) -> ()
        """
        Store the task state hash for later querying
        :param task: The Task object that was created
        :param task_hash: The Task Hash (string) to store, if None generate a new task_hash from the Task
        """
        if not task:
            return
        if not task_hash:
            task_hash = cls._create_task_hash(task=task)
        if Session.check_min_api_version('2.13'):
            # noinspection PyProtectedMember
            task._set_runtime_properties(runtime_properties={cls._job_hash_property: str(task_hash)})
        else:
            hash_comment = cls._job_hash_description.format(task_hash) + '\n'
            task.set_comment(task.comment + '\n' + hash_comment if task.comment else hash_comment)


class ClearmlJob(BaseJob):

    def __init__(
            self,
            base_task_id,  # type: str
            parameter_override=None,  # type: Optional[Mapping[str, str]]
            task_overrides=None,  # type: Optional[Mapping[str, str]]
            configuration_overrides=None,  # type: Optional[Mapping[str, Union[str, Mapping]]]
            tags=None,  # type: Optional[Sequence[str]]
            parent=None,  # type: Optional[str]
            disable_clone_task=False,  # type: bool
            allow_caching=False,  # type: bool
            target_project=None,  # type: Optional[str]
            output_uri=None,  # type: Optional[Union[str, bool]]
            **kwargs  # type: Any
    ):
        # type: (...) -> ()
        """
        Create a new Task based on a base_task_id with a different set of parameters

        :param str base_task_id: base task ID to clone from
        :param dict parameter_override: dictionary of parameters and values to set fo the cloned task
        :param dict task_overrides:  Task object specific overrides.
            for example {'script.version_num': None, 'script.branch': 'main'}
        :param configuration_overrides: Optional, override Task configuration objects.
            Expected dictionary of configuration object name and configuration object content.
            Examples:
                {'config_section': dict(key='value')}
                {'config_file': 'configuration file content'}
                {'OmegaConf': YAML.dumps(full_hydra_dict)}
        :param list tags: additional tags to add to the newly cloned task
        :param str parent: Set newly created Task parent task field, default: base_tak_id.
        :param dict kwargs: additional Task creation parameters
        :param bool disable_clone_task: if False (default), clone base task id.
            If True, use the base_task_id directly (base-task must be in draft-mode / created),
        :param bool allow_caching: If True, check if we have a previously executed Task with the same specification.
            If we do, use it and set internal is_cached flag. Default False (always create new Task).
        :param Union[str, bool] output_uri: The storage / output url for this job. This is the default location for
            output models and other artifacts. Check Task.init reference docs for more info (output_uri is a parameter).
        :param str target_project: Optional, Set the target project name to create the cloned Task in.
        """
        super(ClearmlJob, self).__init__()
        base_temp_task = Task.get_task(task_id=base_task_id)
        if disable_clone_task:
            self.task = base_temp_task
            task_status = self.task.status
            if task_status != Task.TaskStatusEnum.created:
                logger.warning('Task cloning disabled but requested Task [{}] status={}. '
                               'Reverting to clone Task'.format(base_task_id, task_status))
                disable_clone_task = False
                self.task = None
            elif parent:
                self.task.set_parent(parent)
        else:
            self.task = None

        self.task_parameter_override = None
        task_params = None
        if parameter_override:
            task_params = base_temp_task.get_parameters(backwards_compatibility=False)
            task_params.update(parameter_override)
            self.task_parameter_override = dict(**parameter_override)

        task_configurations = None
        if configuration_overrides:
            task_configurations = deepcopy(base_temp_task.data.configuration or {})
            for k, v in configuration_overrides.items():
                if not isinstance(v, (str, dict)):
                    raise ValueError('Configuration override dictionary value must be wither str or dict, '
                                     'got {} instead'.format(type(v)))
                value = v if isinstance(v, str) else json.dumps(v)
                if k in task_configurations:
                    task_configurations[k].value = value
                else:
                    task_configurations[k] = tasks_service.ConfigurationItem(
                        name=str(k), value=value, description=None, type='json' if isinstance(v, dict) else None
                    )
            configuration_overrides = {k: v.value for k, v in task_configurations.items()}

        sections = {}
        if task_overrides:
            # set values inside the Task
            for k, v in task_overrides.items():
                # notice we can allow ourselves to change the base-task object as we will not use it any further
                # noinspection PyProtectedMember
                base_temp_task._set_task_property(k, v, raise_on_error=False, log_on_error=True)
                section = k.split('.')[0]
                sections[section] = getattr(base_temp_task.data, section, None)

        # check cached task
        self._is_cached_task = False
        task_hash = None
        if allow_caching:
            # look for a cached copy of the Task
            # get parameters + task_overrides + as dict and hash it.
            task_hash = self._create_task_hash(
                base_temp_task,
                section_overrides=sections,
                params_override=task_params,
                configurations_override=configuration_overrides or None,
                explicit_docker_image=kwargs.get("explicit_docker_image")
            )
            task = self._get_cached_task(task_hash)
            # if we found a task, just use
            if task:
                if disable_clone_task and self.task and self.task.status == self.task.TaskStatusEnum.created:
                    # if the base task at is in draft mode, and we are using cached task
                    # we assume the base Task was created adhoc and we can delete it.
                    pass  # self.task.delete()

                self._is_cached_task = True
                self.task = task
                self.task_started = True
                self._worker = None
                return

        # if we have target_project, remove project from kwargs if we have it.
        if target_project and 'project' in kwargs:
            logger.info(
                'target_project={} and project={} passed, using target_project.'.format(
                    target_project, kwargs['project']))
            kwargs.pop('project', None)

        # check again if we need to clone the Task
        if not disable_clone_task:
            # noinspection PyProtectedMember
            self.task = Task.clone(
                base_task_id, parent=parent or base_task_id,
                project=get_or_create_project(
                    session=Task._get_default_session(), project_name=target_project
                ) if target_project else kwargs.pop('project', None),
                **kwargs
            )

        if tags:
            self.task.set_tags(list(set(self.task.get_tags()) | set(tags)))

        if task_params:
            param_types = {}
            for key, value in task_params.items():
                if verify_basic_type(value):
                    param_types[key] = get_basic_type(value)
            self.task.set_parameters(task_params, __parameters_types=param_types)

        # store back Task configuration object into backend
        if task_configurations:
            # noinspection PyProtectedMember
            self.task._edit(configuration=task_configurations)

        if task_overrides and sections:
            # store back Task parameters into backend
            # noinspection PyProtectedMember
            self.task._edit(**sections)

        if output_uri is not None:
            self.task.output_uri = output_uri
        self._set_task_cache_hash(self.task, task_hash)
        self.task_started = False
        self._worker = None


class LocalClearmlJob(ClearmlJob):
    """
    Run jobs locally as a sub-process, use only when no agents are available (this will not use queues)
    or for debug purposes.
    """
    def __init__(self, *args, **kwargs):
        super(LocalClearmlJob, self).__init__(*args, **kwargs)
        self._job_process = None
        self._local_temp_file = None

    def launch(self, queue_name=None):
        # type: (str) -> bool
        """
        Launch job as a subprocess, ignores "queue_name"

        :param queue_name: Ignored

        :return: True if successful
        """
        if self._is_cached_task:
            return False

        # check if standalone
        diff = self.task.data.script.diff
        if diff and not diff.lstrip().startswith('diff '):
            # standalone, we need to create if
            fd, local_filename = tempfile.mkstemp(suffix='.py')
            os.close(fd)
            with open(local_filename, 'wt') as f:
                f.write(diff)
            self._local_temp_file = local_filename
        else:
            local_filename = self.task.data.script.entry_point

        cwd = os.path.join(os.getcwd(), self.task.data.script.working_dir or '')
        # try to check based on current root repo + entrypoint
        if Task.current_task() and not (Path(cwd)/local_filename).is_file():
            working_dir = Task.current_task().data.script.working_dir or ''
            working_dir = working_dir.strip('.')
            levels = 0
            if working_dir:
                levels = 1 + sum(1 for c in working_dir if c == '/')
            cwd = os.path.abspath(os.path.join(os.getcwd(), os.sep.join(['..'] * levels))) if levels else os.getcwd()
            cwd = os.path.join(cwd, working_dir)

        python = sys.executable
        env = dict(**os.environ)
        env.pop('CLEARML_PROC_MASTER_ID', None)
        env.pop('TRAINS_PROC_MASTER_ID', None)
        env['CLEARML_TASK_ID'] = env['TRAINS_TASK_ID'] = str(self.task.id)
        env['CLEARML_LOG_TASK_TO_BACKEND'] = '1'
        env['CLEARML_SIMULATE_REMOTE_TASK'] = '1'
        self.task.mark_started()
        self._job_process = subprocess.Popen(args=[python, local_filename], cwd=cwd, env=env)
        return True

    def wait_for_process(self, timeout=None):
        # type: (Optional[int]) -> Optional[int]
        """
        Wait until Job subprocess completed/exited

        :param timeout: Timeout in seconds to wait for the subprocess to finish. Default: None => infinite
        :return Sub-process exit code. 0 is success, None if subprocess is not running or timeout
        """
        if not self._job_process:
            return None
        try:
            exit_code = self._job_process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

        self._job_process = None
        if self._local_temp_file:
            # noinspection PyBroadException
            try:
                Path(self._local_temp_file).unlink()
            except Exception:
                pass
            self._local_temp_file = None

        if exit_code == 0:
            self.task.mark_completed()
        else:
            user_aborted = False
            if self.task.status == Task.TaskStatusEnum.stopped:
                self.task.reload()
                if str(self.task.data.status_reason).lower().startswith('user aborted'):
                    user_aborted = True

            if not user_aborted:
                self.task.mark_failed(force=True)

        return exit_code

    def status(self, force=False):
        # type: (bool) -> str
        """
        Return the Job Task current status. Options are: "created", "queued", "in_progress", "stopped", "published",
        "publishing", "closed", "failed", "completed", "unknown".

        :param force: Force status update, otherwise, only refresh state every 1 sec

        :return: Task status Task.TaskStatusEnum in string.
        """
        if self._job_process:
            # refresh the task state, we need to do it manually
            self.wait_for_process(timeout=0)

        return super(LocalClearmlJob, self).status(force=force)


class RunningJob(BaseJob):
    """
    Wrapper to an already running Task
    """

    def __init__(self, existing_task):  # noqa
        # type: (Union[Task, str]) -> None
        super(RunningJob, self).__init__()
        self.task = existing_task if isinstance(existing_task, Task) else Task.get_task(task_id=existing_task)
        self.task_started = bool(self.task.status != Task.TaskStatusEnum.created)

    def force_set_is_cached(self, cached):
        # type: (bool) -> ()
        self._is_cached_task = bool(cached)


class TrainsJob(ClearmlJob):
    """
    Deprecated, use ClearmlJob
    """

    def __init__(self, **kwargs):
        super(TrainsJob, self).__init__(**kwargs)
        warnings.warn(
            "Use clearml.automation.ClearmlJob",
            DeprecationWarning,
        )


# noinspection PyMethodMayBeStatic, PyUnusedLocal
class _JobStub(object):
    """
    This is a Job Stub, use only for debugging
    """

    def __init__(
            self,
            base_task_id,  # type: str
            parameter_override=None,  # type: Optional[Mapping[str, str]]
            task_overrides=None,  # type: Optional[Mapping[str, str]]
            tags=None,  # type: Optional[Sequence[str]]
            **kwargs  # type: Any
     ):
        # type: (...) -> ()
        self.task = None
        self.base_task_id = base_task_id
        self.parameter_override = parameter_override
        self.task_overrides = task_overrides
        self.tags = tags
        self.iteration = -1
        self.task_started = None

    def launch(self, queue_name=None):
        # type: (str) -> ()
        self.iteration = 0
        self.task_started = time()
        print('launching', self.parameter_override, 'in', queue_name)

    def abort(self):
        # type: () -> ()
        self.task_started = -1

    def elapsed(self):
        # type: () -> float
        """
        Return the time in seconds since job started. Return -1 if job is still pending

        :return: Seconds from start.
        """
        if self.task_started is None:
            return -1
        return time() - self.task_started

    def iterations(self):
        # type: () -> int
        """
        Return the last iteration value of the current job. -1 if job has not started yet
        :return: Task last iteration.
        """
        if self.task_started is None:
            return -1
        return self.iteration

    def get_metric(self, title, series):
        # type: (str, str) -> (float, float, float)
        """
        Retrieve a specific scalar metric from the running Task.

        :param str title: Graph title (metric)
        :param str series: Series on the specific graph (variant)
        :return: min value, max value, last value
        """
        return 0, 1.0, 0.123

    def task_id(self):
        # type: () -> str
        return 'stub'

    def status(self):
        # type: () -> str
        return 'in_progress'

    def wait(self, timeout=None, pool_period=30.):
        # type: (Optional[float], float) -> bool
        """
        Wait for the task to be processed (i.e., aborted/completed/failed)

        :param timeout: maximum time (minutes) to wait for Task to finish
        :param pool_period: check task status every pool_period seconds
        :return: True, if the Task finished.
        """
        return True

    def get_console_output(self, number_of_reports=1):
        # type: (int) -> Sequence[str]
        """
        Return a list of console outputs reported by the Task.
        Returned console outputs are retrieved from the most updated console outputs.


        :param int number_of_reports: number of reports to return, default 1, the last (most updated) console output
        :return: List of strings each entry corresponds to one report.
        """
        return []

    def is_running(self):
        # type: () -> bool
        return self.task_started is not None and self.task_started > 0

    def is_stopped(self):
        # type: () -> bool
        return self.task_started is not None and self.task_started < 0

    def is_pending(self):
        # type: () -> bool
        return self.task_started is None

    def started(self):
        # type: () -> bool
        return not self.is_pending()
