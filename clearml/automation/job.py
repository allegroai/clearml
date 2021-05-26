import hashlib
import warnings
from datetime import datetime
from logging import getLogger
from time import time, sleep
from typing import Optional, Mapping, Sequence, Any

from ..storage.util import hash_dict
from ..task import Task
from ..backend_api.services import tasks as tasks_service


logger = getLogger('clearml.automation.job')


class ClearmlJob(object):
    _job_hash_description = 'job_hash={}'

    def __init__(
            self,
            base_task_id,  # type: str
            parameter_override=None,  # type: Optional[Mapping[str, str]]
            task_overrides=None,  # type: Optional[Mapping[str, str]]
            tags=None,  # type: Optional[Sequence[str]]
            parent=None,  # type: Optional[str]
            disable_clone_task=False,  # type: bool
            allow_caching=False,  # type: bool
            **kwargs  # type: Any
    ):
        # type: (...) -> ()
        """
        Create a new Task based in a base_task_id with a different set of parameters

        :param str base_task_id: base task id to clone from
        :param dict parameter_override: dictionary of parameters and values to set fo the cloned task
        :param dict task_overrides:  Task object specific overrides
        :param list tags: additional tags to add to the newly cloned task
        :param str parent: Set newly created Task parent task field, default: base_tak_id.
        :param dict kwargs: additional Task creation parameters
        :param bool disable_clone_task: if False (default) clone base task id.
            If True, use the base_task_id directly (base-task must be in draft-mode / created),
        :param bool allow_caching: If True check if we have a previously executed Task with the same specification
            If we do, use it and set internal is_cached flag. Default False (always create new Task).
        """
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
        if allow_caching and not disable_clone_task or not self.task:
            # look for a cached copy of the Task
            # get parameters + task_overrides + as dict and hash it.
            task_hash = self._create_task_hash(
                base_temp_task, section_overrides=sections, params_override=task_params)
            task = self._get_cached_task(task_hash)
            # if we found a task, just use
            if task:
                self._is_cached_task = True
                self.task = task
                self.task_started = True
                self._worker = None
                return

        # check again if we need to clone the Task
        if not disable_clone_task:
            self.task = Task.clone(base_task_id, parent=parent or base_task_id, **kwargs)

        if tags:
            self.task.set_tags(list(set(self.task.get_tags()) | set(tags)))

        if task_params:
            self.task.set_parameters(task_params)

        if task_overrides and sections:
            # store back Task parameters into backend
            # noinspection PyProtectedMember
            self.task._edit(**sections)

        self._set_task_cache_hash(self.task, task_hash)
        self.task_started = False
        self._worker = None

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

        :return False if Task is not in "created" status (i.e. cannot be enqueued)
        """
        if self._is_cached_task:
            return False
        try:
            Task.enqueue(task=self.task, queue_name=queue_name)
            return True
        except Exception as ex:
            logger.warning(ex)
        return False

    def abort(self):
        # type: () -> ()
        """
        Abort currently running job (can be called multiple times)
        """
        if not self.task or self._is_cached_task:
            return
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

    def status(self):
        # type: () -> str
        """
        Return the Job Task current status, see Task.TaskStatusEnum

        :return: Task status Task.TaskStatusEnum in string.
        """
        return self.task.status

    def wait(self, timeout=None, pool_period=30.):
        # type: (Optional[float], float) -> bool
        """
        Wait until the task is fully executed (i.e., aborted/completed/failed)

        :param timeout: maximum time (minutes) to wait for Task to finish
        :param pool_period: check task status every pool_period seconds
        :return: True, if Task finished.
        """
        tic = time()
        while timeout is None or time() - tic < timeout * 60.:
            if self.is_stopped():
                return True
            sleep(pool_period)

        return self.is_stopped()

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
        # type: () -> str
        """
        Return the current worker id executing this Job. If job is pending, returns None

        :return: ID of the worker executing / executed the job, or None if job is still pending.
        """
        if self.is_pending():
            return self._worker

        if self._worker is None:
            # the last console outputs will update the worker
            self.get_console_output(number_of_reports=1)
            # if we still do not have it, store empty string
            if not self._worker:
                self._worker = ''

        return self._worker

    def is_running(self):
        # type: () -> bool
        """
        Return True, if job is currently running (pending is considered False)

        :return: True, if the task is currently in progress.
        """
        return self.task.status == Task.TaskStatusEnum.in_progress

    def is_stopped(self):
        # type: () -> bool
        """
        Return True, if job finished executing (for any reason)

        :return: True the task is currently one of these states, stopped / completed / failed / published.
        """
        return self.task.status in (
            Task.TaskStatusEnum.stopped, Task.TaskStatusEnum.completed,
            Task.TaskStatusEnum.failed, Task.TaskStatusEnum.published)

    def is_failed(self):
        # type: () -> bool
        """
        Return True, if job is has executed and failed

        :return: True the task is currently in failed state
        """
        return self.task.status in (Task.TaskStatusEnum.failed, )

    def is_completed(self):
        # type: () -> bool
        """
        Return True, if job is has executed and completed successfully

        :return: True the task is currently in completed or published state
        """
        return self.task.status in (Task.TaskStatusEnum.completed, Task.TaskStatusEnum.published)

    def is_aborted(self):
        # type: () -> bool
        """
        Return True, if job is has executed and aborted

        :return: True the task is currently in aborted state
        """
        return self.task.status in (Task.TaskStatusEnum.stopped, )

    def is_pending(self):
        # type: () -> bool
        """
        Return True, if job is waiting for execution

        :return: True the task is currently is currently queued.
        """
        return self.task.status in (Task.TaskStatusEnum.queued, Task.TaskStatusEnum.created)

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
    def _create_task_hash(cls, task, section_overrides=None, params_override=None):
        # type: (Task, Optional[dict], Optional[dict]) -> Optional[str]
        """
        Create Hash (str) representing the state of the Task
        :param task: A Task to hash
        :param section_overrides: optional dict (keys are Task's section names) with task overrides.
        :param params_override: Alternative to the entire Task's hyper parameters section
        (notice this should not be a nested dict but a flat key/value)
        :return: str crc32 of the Task configuration
        """
        if not task:
            return None
        if section_overrides and section_overrides.get('script'):
            script = section_overrides['script']
            if not isinstance(script, dict):
                script = script.to_dict()
        else:
            script = task.data.script.to_dict() if task.data.script else {}

        # if we have a repository, we must make sure we have a specific version_num to ensure consistency
        if script.get('repository') and not script.get('version_num') and not script.get('tag'):
            return None

        # we need to ignore `requirements` section because ir might be changing from run to run
        script.pop("requirements", None)

        hyper_params = task.get_parameters() if params_override is None else params_override
        configs = task.get_configuration_objects()
        return hash_dict(dict(script=script, hyper_params=hyper_params, configs=configs), hash_func='crc32')

    @classmethod
    def _get_cached_task(cls, task_hash):
        # type: (str) -> Optional[Task]
        """

        :param task_hash:
        :return: A task matching the requested task hash
        """
        if not task_hash:
            return None
        # noinspection PyProtectedMember
        potential_tasks = Task._query_tasks(
            status=['completed', 'stopped', 'published'],
            system_tags=['-{}'.format(Task.archived_tag)],
            _all_=dict(fields=['comment'], pattern=cls._job_hash_description.format(task_hash)),
            only_fields=['id'],
        )
        for obj in potential_tasks:
            task = Task.get_task(task_id=obj.id)
            if task_hash == cls._create_task_hash(task):
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
        hash_comment = cls._job_hash_description.format(task_hash) + '\n'
        task.set_comment(task.comment + '\n' + hash_comment if task.comment else hash_comment)


class TrainsJob(ClearmlJob):

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

    def worker(self):
        # type: () -> ()
        return None

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
