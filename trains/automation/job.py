import hashlib
from datetime import datetime
from logging import getLogger
from time import time, sleep
from typing import Optional, Mapping, Sequence, Any

from ..task import Task
from ..backend_api.services import tasks as tasks_service


logger = getLogger('trains.automation.job')


class TrainsJob(object):
    def __init__(
            self,
            base_task_id,  # type: str
            parameter_override=None,  # type: Optional[Mapping[str, str]]
            task_overrides=None,  # type: Optional[Mapping[str, str]]
            tags=None,  # type: Optional[Sequence[str]]
            parent=None,  # type: Optional[str]
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
        """
        self.task = Task.clone(base_task_id, parent=parent or base_task_id, **kwargs)
        if tags:
            self.task.set_tags(list(set(self.task.get_tags()) | set(tags)))
        if parameter_override:
            params = self.task.get_parameters_as_dict()
            params.update(parameter_override)
            self.task.set_parameters_as_dict(params)
        if task_overrides:
            # todo: make sure it works
            # noinspection PyProtectedMember
            self.task._edit(**task_overrides)
        self.task_started = False
        self._worker = None

    def get_metric(self, title, series):
        # type: (str, str) -> (float, float, float)
        """
        Retrieve a specific scalar metric from the running Task.

        :param str title: Graph title (metric)
        :param str series: Series on the specific graph (variant)
        :return tuple: min value, max value, last value
        """
        title = hashlib.md5(str(title).encode('utf-8')).hexdigest()
        series = hashlib.md5(str(series).encode('utf-8')).hexdigest()
        metric = 'last_metrics.{}.{}.'.format(title, series)
        values = ['min_value', 'max_value', 'value']
        metrics = [metric + v for v in values]

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

    def launch(self, queue_name=None):
        # type: (str) -> ()
        """
        Send Job for execution on the requested execution queue

        :param str queue_name:
        """
        try:
            Task.enqueue(task=self.task, queue_name=queue_name)
        except Exception as ex:
            logger.warning(ex)

    def abort(self):
        # type: () -> ()
        """
        Abort currently running job (can be called multiple times)
        """
        try:
            self.task.stopped()
        except Exception as ex:
            logger.warning(ex)

    def elapsed(self):
        # type: () -> float
        """
        Return the time in seconds since job started. Return -1 if job is still pending

        :return float: seconds from start
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

        :return int: Task last iteration
        """
        if not self.task_started and self.task.status != Task.TaskStatusEnum.in_progress:
            return -1
        self.task_started = True
        return self.task.get_last_iteration()

    def task_id(self):
        # type: () -> str
        """
        Return the Task id.

        :return str: Task id
        """
        return self.task.id

    def status(self):
        # type: () -> str
        """
        Return the Job Task current status, see Task.TaskStatusEnum

        :return str: Task status Task.TaskStatusEnum in string
        """
        return self.task.status

    def wait(self, timeout=None, pool_period=30.):
        # type: (Optional[float], float) -> bool
        """
        Wait until the task is fully executed (i.e. aborted/completed/failed)

        :param timeout: maximum time (minutes) to wait for Task to finish
        :param pool_period: check task status every pool_period seconds
        :return bool: Return True is Task finished.
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
        :return list: List of strings each entry corresponds to one report.
        """
        return self.task.get_reported_console_output(number_of_reports=number_of_reports)

    def worker(self):
        # type: () -> str
        """
        Return the current worker id executing this Job. If job is pending, returns None

        :return str: Worker ID (str) executing / executed the job, or None if job is still pending.
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
        Return True if job is currently running (pending is considered False)

        :return bool: True iff the task is currently in progress
        """
        return self.task.status == Task.TaskStatusEnum.in_progress

    def is_stopped(self):
        # type: () -> bool
        """
        Return True if job is has executed and is not any more

        :return bool: True the task is currently one of these states, stopped / completed / failed
        """
        return self.task.status in (
            Task.TaskStatusEnum.stopped, Task.TaskStatusEnum.completed,
            Task.TaskStatusEnum.failed, Task.TaskStatusEnum.published)

    def is_pending(self):
        # type: () -> bool
        """
        Return True if job is waiting for execution

        :return bool: True the task is currently is currently queued
        """
        return self.task.status in (Task.TaskStatusEnum.queued, Task.TaskStatusEnum.created)

    def started(self):
        # type: () -> bool
        """
        Return True if job already started, or ended (or False if created/pending)

        :return bool: False if the task is currently in draft mode or pending
        """
        if not self.task_started and self.task.status in (
                Task.TaskStatusEnum.in_progress, Task.TaskStatusEnum.created):
            return False

        self.task_started = True
        return True


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

        :return float: seconds from start
        """
        if self.task_started is None:
            return -1
        return time() - self.task_started

    def iterations(self):
        # type: () -> int
        """
        Return the last iteration value of the current job. -1 if job has not started yet
        :return int: Task last iteration
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
        :return list: min value, max value, last value
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
        Wait for the task to be processed (i.e. aborted/completed/failed)

        :param timeout: maximum time (minutes) to wait for Task to finish
        :param pool_period: check task status every pool_period seconds
        :return bool: Return True is Task finished.
        """
        return True

    def get_console_output(self, number_of_reports=1):
        # type: (int) -> Sequence[str]
        """
        Return a list of console outputs reported by the Task.
        Returned console outputs are retrieved from the most updated console outputs.


        :param int number_of_reports: number of reports to return, default 1, the last (most updated) console output
        :return list: List of strings each entry corresponds to one report.
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
