import hashlib
import json
from copy import copy
from datetime import datetime
from itertools import product
from logging import getLogger
from threading import Thread, Event
from time import time
from typing import List, Set, Union, Any, Sequence, Optional, Mapping, Callable

from .job import TrainsJob
from .parameters import Parameter
from ..logger import Logger
from ..backend_api.services import workers as workers_service, tasks as tasks_services
from ..task import Task

logger = getLogger('trains.automation.optimization')


try:
    import pandas as pd
    Task.add_requirements('pandas')
except ImportError:
    pd = None
    logger.warning('Pandas is not installed, summary table reporting will be skipped.')


class Objective(object):
    """
    Optimization ``Objective`` class to maximize / minimize over all experiments. This class will sample a specific
    scalar from all experiments, and maximize / minimize over single scalar (i.e., title and series combination).

    ``SearchStrategy`` and ``HyperParameterOptimizer`` use ``Objective`` in the strategy search algorithm.
    """

    def __init__(self, title, series, order='max', extremum=False):
        # type: (str, str, str, bool) -> ()
        """
        Construct ``Objective`` object that will return the scalar value for a specific task ID.

        :param str title: The scalar graph title to sample from.
        :param str series: The scalar series title to sample from.
        :param str order: The setting for maximizing or minimizing the objective scalar value.

            The values are:

            - ``max``
            - ``min``

        :param bool extremum: Return the global minimum / maximum reported metric value

            The values are:

            - ``True`` - Return the global minimum / maximum reported metric value.
            - ``False`` - Return the last value reported for a specific Task. (Default)

        """
        self.title = title
        self.series = series
        assert order in ('min', 'max',)
        # normalize value so we always look for the highest objective value
        self.sign = -1 if (isinstance(order, str) and order.lower().strip() == 'min') else +1
        self._metric = None
        self.extremum = extremum

    def get_objective(self, task_id):
        # type: (Union[str, Task, TrainsJob]) -> Optional[float]
        """
        Return a specific task scalar value based on the objective settings (title/series).

        :param str task_id: The Task id to retrieve scalar from (or ``TrainsJob`` object).

        :return: The scalar value.
        """
        # create self._metric
        self._get_last_metrics_encode_field()

        if isinstance(task_id, Task):
            task_id = task_id.id
        elif isinstance(task_id, TrainsJob):
            task_id = task_id.task_id()

        # noinspection PyBroadException, Py
        try:
            # noinspection PyProtectedMember
            task = Task._query_tasks(
                task_ids=[task_id], only_fields=['last_metrics.{}.{}'.format(self._metric[0], self._metric[1])])[0]
        except Exception:
            return None

        metrics = task.last_metrics
        # noinspection PyBroadException
        try:
            values = metrics[self._metric[0]][self._metric[1]]
            if not self.extremum:
                return values['value']

            return values['min_value'] if self.sign < 0 else values['max_value']
        except Exception:
            return None

    def get_current_raw_objective(self, task):
        # type: (Union[TrainsJob, Task]) -> (int, float)
        """
        Return the current raw value (without sign normalization) of the objective.

        :param str task: The Task or Job to retrieve scalar from (or ``TrainsJob`` object).

        :return: Tuple(iteration, value) if, and only if, the metric exists. None if the metric does not exist.

        """

        if not isinstance(task, Task):
            if hasattr(task, 'task'):
                task = task.task
            if not isinstance(task, Task):
                task = Task.get_task(task_id=str(task))
                if not task:
                    raise ValueError("Task object could not be found")

        # todo: replace with more efficient code
        scalars = task.get_reported_scalars()

        # noinspection PyBroadException
        try:
            return scalars[self.title][self.series]['x'][-1], scalars[self.title][self.series]['y'][-1]
        except Exception:
            return None

    def get_objective_sign(self):
        # type: () -> float
        """
        Return the sign of the objective.

        - ``+1`` - If maximizing
        - ``-1`` - If minimizing

        :return: Objective function sign.
        """
        return self.sign

    def get_objective_metric(self):
        # type: () -> (str, str)
        """
        Return the metric title, series pair of the objective.

        :return: (title, series)
        """
        return self.title, self.series

    def get_normalized_objective(self, task_id):
        # type: (Union[str, Task, TrainsJob]) -> Optional[float]
        """
        Return a normalized task scalar value based on the objective settings (title/series).
        I.e. objective is always to maximize the returned value

        :param str task_id: The Task id to retrieve scalar from.

        :return: Normalized scalar value.
        """
        objective = self.get_objective(task_id=task_id)
        if objective is None:
            return None
        # normalize value so we always look for the highest objective value
        return self.sign * objective

    def _get_last_metrics_encode_field(self):
        # type: () -> str
        """
        Return encoded representation of the title/series metric.

        :return: The objective title/series.
        """
        if not self._metric:
            title = hashlib.md5(str(self.title).encode('utf-8')).hexdigest()
            series = hashlib.md5(str(self.series).encode('utf-8')).hexdigest()
            self._metric = title, series
        return '{}last_metrics.{}.{}.{}'.format(
            '-' if self.sign > 0 else '', self._metric[0], self._metric[1],
            ('min_value' if self.sign < 0 else 'max_value') if self.extremum else 'value')


class Budget(object):
    class Field(object):
        def __init__(self, limit=None):
            # type: (Optional[float]) -> ()
            self.limit = limit
            self.current = {}

        def update(self, uid, value):
            # type: (Union[str, int], float) -> ()
            if value is not None:
                try:
                    self.current[uid] = float(value)
                except (TypeError, ValueError):
                    pass

        @property
        def used(self):
            # type: () -> (Optional[float])
            if self.limit is None or not self.current:
                return None
            return sum(self.current.values())/float(self.limit)

    def __init__(self, jobs_limit, iterations_limit, compute_time_limit):
        # type: (Optional[int], Optional[int], Optional[float]) -> ()
        self.jobs = self.Field(jobs_limit)
        self.iterations = self.Field(iterations_limit)
        self.compute_time = self.Field(compute_time_limit)

    def to_dict(self):
        # type: () -> (Mapping[str, Mapping[str, float]])

        # returned dict is Mapping[Union['jobs', 'iterations', 'compute_time'], Mapping[Union['limit', 'used'], float]]
        current_budget = {}
        jobs = self.jobs.used
        current_budget['jobs'] = {'limit': self.jobs.limit, 'used': jobs if jobs else 0}
        iterations = self.iterations.used
        current_budget['iterations'] = {'limit': self.iterations.limit, 'used': iterations if iterations else 0}
        compute_time = self.compute_time.used
        current_budget['compute_time'] = {'limit': self.compute_time.limit, 'used': compute_time if compute_time else 0}
        return current_budget


class SearchStrategy(object):
    """
    The base search strategy class. Inherit this class to implement your custom strategy.
    """
    _tag = 'optimization'
    _job_class = TrainsJob  # type: TrainsJob

    def __init__(
            self,
            base_task_id,  # type: str
            hyper_parameters,  # type: Sequence[Parameter]
            objective_metric,  # type: Objective
            execution_queue,  # type: str
            num_concurrent_workers,  # type: int
            pool_period_min=2.,  # type: float
            time_limit_per_job=None,  # type: Optional[float]
            compute_time_limit=None,  # type: Optional[float]
            min_iteration_per_job=None,  # type: Optional[int]
            max_iteration_per_job=None,  # type: Optional[int]
            total_max_jobs=None,  # type: Optional[int]
            **_  # type: Any
    ):
        # type: (...) -> ()
        """
        Initialize a search strategy optimizer.

        :param str base_task_id: The Task ID (str)
        :param list hyper_parameters: The list of parameter objects to optimize over.
        :param Objective objective_metric: The Objective metric to maximize / minimize.
        :param str execution_queue: The execution queue to use for launching Tasks (experiments).
        :param int num_concurrent_workers: The maximum number of concurrent running machines.
        :param float pool_period_min: The time between two consecutive pools (minutes).
        :param float time_limit_per_job: The maximum execution time per single job in minutes. When time limit is
            exceeded, the job is aborted. (Optional)
        :param float compute_time_limit: The maximum compute time in minutes. When time limit is exceeded,
            all jobs aborted. (Optional)
        :param int min_iteration_per_job: The minimum iterations (of the Objective metric) per single job (Optional)
        :param int max_iteration_per_job: The maximum iterations (of the Objective metric) per single job.
            When maximum iterations is exceeded, the job is aborted.  (Optional)
        :param int total_max_jobs: The total maximum jobs for the optimization process. The default value is ``None``,
            for unlimited.
        """
        super(SearchStrategy, self).__init__()
        self._base_task_id = base_task_id
        self._hyper_parameters = hyper_parameters
        self._objective_metric = objective_metric
        self._execution_queue = execution_queue
        self._num_concurrent_workers = num_concurrent_workers
        self.pool_period_minutes = pool_period_min
        self.time_limit_per_job = time_limit_per_job
        self.compute_time_limit = compute_time_limit
        self.max_iteration_per_job = max_iteration_per_job
        self.min_iteration_per_job = min_iteration_per_job
        self.total_max_jobs = total_max_jobs
        self._stop_event = Event()
        self._current_jobs = []
        self._pending_jobs = []
        self._num_jobs = 0
        self._job_parent_id = None
        self._created_jobs_ids = {}
        self._naming_function = None
        self._job_project = {}
        self.budget = Budget(
            jobs_limit=self.total_max_jobs,
            compute_time_limit=self.compute_time_limit if self.compute_time_limit else None,
            iterations_limit=self.total_max_jobs * self.max_iteration_per_job if
            self.max_iteration_per_job and self.total_max_jobs else None
        )
        self._validate_base_task()
        self._optimizer_task = None

    def start(self):
        # type: () -> ()
        """
        Start the Optimizer controller function loop(). If the calling process is stopped, the controller will stop
        as well.

        .. important::
            This function returns only after the optimization is completed or :meth:`stop` was called.

        """
        counter = 0
        while True:
            logger.debug('optimization loop #{}'.format(counter))
            if not self.process_step():
                break
            if self._stop_event.wait(timeout=self.pool_period_minutes * 60.):
                break
            counter += 1

    def stop(self):
        # type: () -> ()
        """
        Stop the current running optimization loop. Called from a different thread than the :meth:`start`.
        """
        self._stop_event.set()

    def process_step(self):
        # type: () -> bool
        """
        Abstract helper function. Implementation is not required. Default use in start default implementation
        Main optimization loop, called from the daemon thread created by :meth:`start`.

        - Call monitor job on every ``TrainsJob`` in jobs:

          - Check the performance or elapsed time, and then decide whether to kill the jobs.

        - Call create_job:

          - Check if spare job slots exist, and if they do call create a new job based on previous tested experiments.

        :return: True, if continue the optimization. False, if immediately stop.

        """
        updated_jobs = []
        for job in self._current_jobs:
            if self.monitor_job(job):
                updated_jobs.append(job)

        self._current_jobs = updated_jobs

        pending_jobs = []
        for job in self._pending_jobs:
            if job.is_pending():
                pending_jobs.append(job)
            else:
                self.budget.jobs.update(job.task_id(), 1)

        self._pending_jobs = pending_jobs

        free_workers = self._num_concurrent_workers - len(self._current_jobs)

        # do not create more jobs if we hit the limit
        if self.total_max_jobs and self._num_jobs >= self.total_max_jobs:
            return bool(self._current_jobs)

        # see how many free slots we have and create job
        for i in range(max(0, free_workers)):
            new_job = self.create_job()
            if not new_job:
                break
            self._num_jobs += 1
            new_job.launch(self._execution_queue)
            self._current_jobs.append(new_job)
            self._pending_jobs.append(new_job)

        return bool(self._current_jobs)

    def create_job(self):
        # type: () -> Optional[TrainsJob]
        """
        Abstract helper function. Implementation is not required. Default use in process_step default implementation
        Create a new job if needed. return the newly created job. If no job needs to be created, return ``None``.

        :return: A Newly created TrainsJob object, or None if no TrainsJob created.
        """
        return None

    def monitor_job(self, job):
        # type: (TrainsJob) -> bool
        """
        Helper function, Implementation is not required. Default use in process_step default implementation.
        Check if the job needs to be aborted or already completed.

        If returns ``False``, the job was aborted / completed, and should be taken off the current job list

        If there is a budget limitation, this call should update
        ``self.budget.compute_time.update`` / ``self.budget.iterations.update``

        :param TrainsJob job: A ``TrainsJob`` object to monitor.

        :return: False, if the job is no longer relevant.
        """

        abort_job = self.update_budget_per_job(job)

        if abort_job:
            job.abort()
            return False

        return not job.is_stopped()

    def update_budget_per_job(self, job):
        abort_job = False
        if self.time_limit_per_job:
            elapsed = job.elapsed() / 60.
            if elapsed > 0:
                self.budget.compute_time.update(job.task_id(), elapsed)
                if elapsed > self.time_limit_per_job:
                    abort_job = True

        if self.compute_time_limit:
            if not self.time_limit_per_job:
                elapsed = job.elapsed() / 60.
                if elapsed > 0:
                    self.budget.compute_time.update(job.task_id(), elapsed)

        if self.max_iteration_per_job:
            iterations = self._get_job_iterations(job)
            if iterations > 0:
                self.budget.iterations.update(job.task_id(), iterations)
                if iterations > self.max_iteration_per_job:
                    abort_job = True

        return abort_job

    def get_running_jobs(self):
        # type: () -> Sequence[TrainsJob]
        """
        Return the current running TrainsJobs.

        :return: List of TrainsJob objects.
        """
        return self._current_jobs

    def get_created_jobs_ids(self):
        # type: () -> Mapping[str, dict]
        """
        Return a Task IDs dict created by this optimizer until now, including completed and running jobs.
        The values of the returned dict are the parameters used in the specific job

        :return: dict of task IDs (str) as keys, and their parameters dict as values.
        """
        return {job_id: job_val[1] for job_id, job_val in self._created_jobs_ids.items()}

    def get_created_jobs_tasks(self):
        # type: () -> Mapping[str, dict]
        """
        Return a Task IDs dict created by this optimizer until now.
        The values of the returned dict are the TrainsJob.

        :return: dict of task IDs (str) as keys, and their TrainsJob as values.
        """
        return {job_id: job_val[0] for job_id, job_val in self._created_jobs_ids.items()}

    def get_top_experiments(self, top_k):
        # type: (int) -> Sequence[Task]
        """
        Return a list of Tasks of the top performing experiments, based on the controller ``Objective`` object.

        :param int top_k: The number of Tasks (experiments) to return.

        :return: A list of Task objects, ordered by performance, where index 0 is the best performing Task.
        """
        # noinspection PyProtectedMember
        top_tasks = self._get_child_tasks(
            parent_task_id=self._job_parent_id or self._base_task_id,
            order_by=self._objective_metric._get_last_metrics_encode_field(),
            additional_filters={'page_size': int(top_k), 'page': 0})
        return top_tasks

    def get_objective_metric(self):
        # type: () -> (str, str)
        """
        Return the metric title, series pair of the objective.

        :return: (title, series)
        """
        return self._objective_metric.get_objective_metric()

    def helper_create_job(
            self,
            base_task_id,  # type: str
            parameter_override=None,  # type: Optional[Mapping[str, str]]
            task_overrides=None,  # type: Optional[Mapping[str, str]]
            tags=None,  # type: Optional[Sequence[str]]
            parent=None,  # type: Optional[str]
            **kwargs  # type: Any
    ):
        # type: (...) -> TrainsJob
        """
        Create a Job using the specified arguments, ``TrainsJob`` for details.

        :return: A newly created Job instance.
        """
        if parameter_override:
            param_str = ['{}={}'.format(k, parameter_override[k]) for k in sorted(parameter_override.keys())]
            if self._naming_function:
                name = self._naming_function(self._base_task_name, parameter_override)
            elif self._naming_function is False:
                name = None
            else:
                name = '{}: {}'.format(self._base_task_name, ' '.join(param_str))
            comment = '\n'.join(param_str)
        else:
            name = None
            comment = None
        tags = (tags or []) + [self._tag, 'opt' + (': {}'.format(self._job_parent_id) if self._job_parent_id else '')]
        new_job = self._job_class(
            base_task_id=base_task_id, parameter_override=parameter_override,
            task_overrides=task_overrides, tags=tags, parent=parent or self._job_parent_id,
            name=name, comment=comment, project=self._get_task_project(parent or self._job_parent_id), **kwargs)
        self._created_jobs_ids[new_job.task_id()] = (new_job, parameter_override)
        logger.info('Creating new Task: {}'.format(parameter_override))
        return new_job

    def set_job_class(self, job_class):
        # type: (TrainsJob) -> ()
        """
        Set the class to use for the :meth:`helper_create_job` function.

        :param TrainsJob job_class: The Job Class type.
        """
        self._job_class = job_class

    def set_job_default_parent(self, job_parent_task_id):
        # type: (str) -> ()
        """
        Set the default parent for all Jobs created by the :meth:`helper_create_job` method.

        :param str job_parent_task_id: The parent Task ID.
        """
        self._job_parent_id = job_parent_task_id

    def set_job_naming_scheme(self, naming_function):
        # type: (Optional[Callable[[str, dict], str]]) -> ()
        """
        Set the function used to name a newly created job.

        :param callable naming_function:

            .. code-block:: py

               naming_functor(base_task_name, argument_dict) -> str

        """
        self._naming_function = naming_function

    def set_optimizer_task(self, task):
        # type: (Task) -> ()
        """
        Set the optimizer task object to be used to store/generate reports on the optimization process.
        Usually this is the current task of this process.

        :param Task task: The optimizer's current Task.
        """
        self._optimizer_task = task

    def _validate_base_task(self):
        # type: () -> ()
        """
        Check the base task exists and contains the requested Objective metric and hyper parameters.
        """
        # check if the task exists
        try:
            task = Task.get_task(task_id=self._base_task_id)
            self._base_task_name = task.name
        except ValueError:
            raise ValueError("Could not find base task id {}".format(self._base_task_id))
        # check if the hyper-parameters exist:
        task_parameters = task.get_parameters(backwards_compatibility=False)
        missing_params = [h.name for h in self._hyper_parameters if h.name not in task_parameters]
        if missing_params:
            logger.warning('Could not find requested hyper-parameters {} on base task {}'.format(
                missing_params, self._base_task_id))
        # check if the objective metric exists (i.e. no typos etc)
        if self._objective_metric.get_objective(self._base_task_id) is None:
            logger.warning('Could not find requested metric {} report on base task {}'.format(
                self._objective_metric.get_objective_metric(), self._base_task_id))

    def _get_task_project(self, parent_task_id):
        # type: (str) -> (Optional[str])
        if not parent_task_id:
            return
        if parent_task_id not in self._job_project:
            task = Task.get_task(task_id=parent_task_id)
            self._job_project[parent_task_id] = task.project

        return self._job_project.get(parent_task_id)

    def _get_job_iterations(self, job):
        # type: (Union[TrainsJob, Task]) -> int
        iteration_value = self._objective_metric.get_current_raw_objective(job)
        return iteration_value[0] if iteration_value else -1

    @classmethod
    def _get_child_tasks(
            cls,
            parent_task_id,  # type: str
            status=None,  # type: Optional[Task.TaskStatusEnum]
            order_by=None,  # type: Optional[str]
            additional_filters=None  # type: Optional[dict]
    ):
        # type: (...) -> (Sequence[Task])
        """
        Helper function. Return a list of tasks tagged automl, with specific ``status``, ordered by ``sort_field``.

        :param str parent_task_id: The base Task ID (parent).
        :param status: The current status of requested tasks (for example, ``in_progress`` and ``completed``).
        :param str order_by: The field name to sort results.

            Examples:

            .. code-block:: py

                "-last_metrics.title.series.min"
                "last_metrics.title.series.max"
                "last_metrics.title.series.last"
                "execution.parameters.name"
                "updated"

        :param dict additional_filters: The additional task filters.
        :return: A list of Task objects
        """
        task_filter = {'parent': parent_task_id,
                       # 'tags': [cls._tag],
                       'system_tags': ['-archived']}
        task_filter.update(additional_filters or {})

        if status:
            task_filter['status'] = status

        if order_by and (order_by.startswith('last_metrics') or order_by.startswith('-last_metrics')):
            parts = order_by.split('.')
            if parts[-1] in ('min', 'max', 'last'):
                title = hashlib.md5(str(parts[1]).encode('utf-8')).hexdigest()
                series = hashlib.md5(str(parts[2]).encode('utf-8')).hexdigest()
                minmax = 'min_value' if 'min' in parts[3] else ('max_value' if 'max' in parts[3] else 'value')
                order_by = '{}last_metrics.'.join(
                    ('-' if order_by and order_by[0] == '-' else '', title, series, minmax))

        if order_by:
            task_filter['order_by'] = [order_by]

        return Task.get_tasks(task_filter=task_filter)


class GridSearch(SearchStrategy):
    """
    Grid search strategy controller. Full grid sampling of every hyper-parameter combination.
    """

    def __init__(
            self,
            base_task_id,  # type: str
            hyper_parameters,  # type: Sequence[Parameter]
            objective_metric,  # type: Objective
            execution_queue,  # type: str
            num_concurrent_workers,  # type: int
            pool_period_min=2.,  # type: float
            time_limit_per_job=None,  # type: Optional[float]
            compute_time_limit=None,  # type: Optional[float]
            max_iteration_per_job=None,  # type: Optional[int]
            total_max_jobs=None,  # type: Optional[int]
            **_  # type: Any
    ):
        # type: (...) -> ()
        """
        Initialize a grid search optimizer

        :param str base_task_id: The Task ID.
        :param list hyper_parameters: The list of parameter objects to optimize over.
        :param Objective objective_metric: The Objective metric to maximize / minimize.
        :param str execution_queue: The execution queue to use for launching Tasks (experiments).
        :param int num_concurrent_workers: The maximum number of concurrent running machines.
        :param float pool_period_min: The time between two consecutive pools (minutes).
        :param float time_limit_per_job: The maximum execution time per single job in minutes. When the time limit is
            exceeded job is aborted. (Optional)
        :param float compute_time_limit: The maximum compute time in minutes. When time limit is exceeded,
            all jobs aborted. (Optional)
        :param int max_iteration_per_job: The maximum iterations (of the Objective metric)
            per single job, When exceeded, the job is aborted.
        :param int total_max_jobs: The total maximum jobs for the optimization process. The default is ``None``, for
            unlimited.
        """
        super(GridSearch, self).__init__(
            base_task_id=base_task_id, hyper_parameters=hyper_parameters, objective_metric=objective_metric,
            execution_queue=execution_queue, num_concurrent_workers=num_concurrent_workers,
            pool_period_min=pool_period_min, time_limit_per_job=time_limit_per_job,
            compute_time_limit=compute_time_limit, max_iteration_per_job=max_iteration_per_job,
            total_max_jobs=total_max_jobs, **_)
        self._param_iterator = None

    def create_job(self):
        # type: () -> Optional[TrainsJob]
        """
        Create a new job if needed. Return the newly created job. If no job needs to be created, return ``None``.

        :return: A newly created TrainsJob object, or None if no TrainsJob is created.
        """
        try:
            parameters = self._next_configuration()
        except StopIteration:
            return None

        return self.helper_create_job(base_task_id=self._base_task_id, parameter_override=parameters)

    def _next_configuration(self):
        # type: () -> Mapping[str, str]
        def param_iterator_fn():
            hyper_params_values = [p.to_list() for p in self._hyper_parameters]
            for state in product(*hyper_params_values):
                yield dict(kv for d in state for kv in d.items())

        if not self._param_iterator:
            self._param_iterator = param_iterator_fn()
        return next(self._param_iterator)


class RandomSearch(SearchStrategy):
    """
    Random search strategy controller. Random uniform sampling of hyper-parameters.
    """

    # Number of already chosen random samples before assuming we covered the entire hyper-parameter space
    _hp_space_cover_samples = 42

    def __init__(
            self,
            base_task_id,  # type: str
            hyper_parameters,  # type: Sequence[Parameter]
            objective_metric,  # type: Objective
            execution_queue,  # type: str
            num_concurrent_workers,  # type: int
            pool_period_min=2.,  # type: float
            time_limit_per_job=None,  # type: Optional[float]
            compute_time_limit=None,  # type: Optional[float]
            max_iteration_per_job=None,  # type: Optional[int]
            total_max_jobs=None,  # type: Optional[int]
            **_  # type: Any
    ):
        # type: (...) -> ()
        """
        Initialize a random search optimizer.

        :param str base_task_id: The Task ID.
        :param list hyper_parameters: The list of Parameter objects to optimize over.
        :param Objective objective_metric: The Objective metric to maximize / minimize.
        :param str execution_queue: The execution queue to use for launching Tasks (experiments).
        :param int num_concurrent_workers: The maximum umber of concurrent running machines.
        :param float pool_period_min: The time between two consecutive pools (minutes).
        :param float time_limit_per_job: The maximum execution time per single job in minutes,
            when time limit is exceeded job is aborted. (Optional)
        :param float compute_time_limit: The maximum compute time in minutes. When time limit is exceeded,
            all jobs aborted. (Optional)
        :param int max_iteration_per_job: The maximum iterations (of the Objective metric)
            per single job. When exceeded, the job is aborted.
        :param int total_max_jobs: The total maximum jobs for the optimization process. The default is ``None``, for
            unlimited.
        """
        super(RandomSearch, self).__init__(
            base_task_id=base_task_id, hyper_parameters=hyper_parameters, objective_metric=objective_metric,
            execution_queue=execution_queue, num_concurrent_workers=num_concurrent_workers,
            pool_period_min=pool_period_min, time_limit_per_job=time_limit_per_job,
            compute_time_limit=compute_time_limit, max_iteration_per_job=max_iteration_per_job,
            total_max_jobs=total_max_jobs, **_)
        self._hyper_parameters_collection = set()

    def create_job(self):
        # type: () -> Optional[TrainsJob]
        """
        Create a new job if needed. Return the newly created job. If no job needs to be created, return ``None``.

        :return: A newly created TrainsJob object, or None if no TrainsJob created
        """
        parameters = None

        # maximum tries to ge a random set that is not already in the collection
        for i in range(self._hp_space_cover_samples):
            parameters = {}
            for p in self._hyper_parameters:
                parameters.update(p.get_value())
            # hash the parameters dictionary
            param_hash = hash(json.dumps(parameters, sort_keys=True))
            # if this is a new set of parameters, use it.
            if param_hash not in self._hyper_parameters_collection:
                self._hyper_parameters_collection.add(param_hash)
                break
            # try again
            parameters = None

        # if we failed to find a random set of parameters, assume we selected all of them
        if not parameters:
            return None

        return self.helper_create_job(base_task_id=self._base_task_id, parameter_override=parameters)


class HyperParameterOptimizer(object):
    """
    Hyper-parameter search controller. Clones the base experiment, changes arguments and tries to maximize/minimize
    the defined objective.
    """
    _tag = 'optimization'

    def __init__(
            self,
            base_task_id,  # type: str
            hyper_parameters,  # type: Sequence[Parameter]
            objective_metric_title,  # type: str
            objective_metric_series,  # type: str
            objective_metric_sign='min',  # type: str
            optimizer_class=RandomSearch,  # type: type(SearchStrategy)
            max_number_of_concurrent_tasks=10,  # type: int
            execution_queue='default',  # type: str
            optimization_time_limit=None,  # type: Optional[float]
            compute_time_limit=None,  # type: Optional[float]
            auto_connect_task=True,  # type: Union[bool, Task]
            always_create_task=False,  # type: bool
            **optimizer_kwargs  # type: Any
    ):
        # type: (...) -> ()
        """
        Create a new hyper-parameter controller. The newly created object will launch and monitor the new experiments.

        :param str base_task_id: The Task ID to be used as template experiment to optimize.
        :param list hyper_parameters: The list of Parameter objects to optimize over.
        :param str objective_metric_title: The Objective metric title to maximize / minimize (for example,
            ``validation``).
        :param str objective_metric_series: The Objective metric series to maximize / minimize (for example, ``loss``).
        :param str objective_metric_sign: The objective to maximize / minimize.

            The values are:

            - ``min`` - Minimize the last reported value for the specified title/series scalar.
            - ``max`` - Maximize the last reported value for the specified title/series scalar.
            - ``min_global`` - Minimize the min value of *all* reported values for the specific title/series scalar.
            - ``max_global`` - Maximize the max value of *all* reported values for the specific title/series scalar.

        :param class.SearchStrategy optimizer_class: The SearchStrategy optimizer to use for the hyper-parameter search
        :param int max_number_of_concurrent_tasks: The maximum number of concurrent Tasks (experiments) running at the
            same time.
        :param str execution_queue: The execution queue to use for launching Tasks (experiments).
        :param float optimization_time_limit: The maximum time (minutes) for the entire optimization process. The
            default is ``None``, indicating no time limit.
        :param float compute_time_limit: The maximum compute time in minutes. When time limit is exceeded,
            all jobs aborted. (Optional)
        :param bool auto_connect_task: Store optimization arguments and configuration in the Task

            The values are:

            - ``True`` - The optimization argument and configuration will be stored in the Task. All arguments will
              be under the hyper-parameter section ``opt``, and the optimization hyper_parameters space will
              stored in the Task configuration object section.

            - ``False`` - Do not store with Task.
            - ``Task`` - A specific Task object to connect the optimization process with.
        :param bool always_create_task: Always create a new Task

            The values are:

            - ``True`` - No current Task initialized. Create a new task named ``optimization`` in the ``base_task_id``
              project.

            - ``False`` - Use the :py:meth:`task.Task.current_task` (if exists) to report statistics.

        :param ** optimizer_kwargs: Arguments passed directly to the optimizer constructor.

            Example:

            .. code-block:: py

                :linenos:
                :caption: Example

                from trains import Task
                from trains.automation import UniformParameterRange, DiscreteParameterRange
                from trains.automation import GridSearch, RandomSearch, HyperParameterOptimizer

                task = Task.init('examples', 'HyperParameterOptimizer example')
                an_optimizer = HyperParameterOptimizer(
                    base_task_id='fa30fa45d95d4927b87c323b5b04dc44',
                    hyper_parameters=[
                        UniformParameterRange('lr', min_value=0.01, max_value=0.3, step_size=0.05),
                        DiscreteParameterRange('network', values=['ResNet18', 'ResNet50', 'ResNet101']),
                    ],
                    objective_metric_title='title',
                    objective_metric_series='series',
                    objective_metric_sign='min',
                    max_number_of_concurrent_tasks=5,
                    optimizer_class=RandomSearch,
                    execution_queue='workers', time_limit_per_job=120, pool_period_min=0.2)

                # This will automatically create and print the optimizer new task id
                # for later use. if a Task was already created, it will use it.
                an_optimizer.set_time_limit(in_minutes=10.)
                an_optimizer.start()
                # we can create a pooling loop if we like
                while not an_optimizer.reached_time_limit():
                    top_exp = an_optimizer.get_top_experiments(top_k=3)
                    print(top_exp)
                # wait until optimization completed or timed-out
                an_optimizer.wait()
                # make sure we stop all jobs
                an_optimizer.stop()
        """

        # create a new Task, if we do not have one already
        self._task = auto_connect_task if isinstance(auto_connect_task, Task) else Task.current_task()
        if not self._task and always_create_task:
            base_task = Task.get_task(task_id=base_task_id)
            self._task = Task.init(
                project_name=base_task.get_project_name(),
                task_name='Optimizing: {}'.format(base_task.name),
                task_type=Task.TaskTypes.optimizer,
            )

        opts = dict(
            base_task_id=base_task_id,
            objective_metric_title=objective_metric_title,
            objective_metric_series=objective_metric_series,
            objective_metric_sign=objective_metric_sign,
            max_number_of_concurrent_tasks=max_number_of_concurrent_tasks,
            execution_queue=execution_queue,
            optimization_time_limit=optimization_time_limit,
            compute_time_limit=compute_time_limit,
            optimizer_kwargs=optimizer_kwargs)
        # make sure all the created tasks are our children, as we are creating them
        if self._task:
            self._task.add_tags([self._tag])
            if auto_connect_task:
                optimizer_class, hyper_parameters, opts = self._connect_args(
                    optimizer_class=optimizer_class, hyper_param_configuration=hyper_parameters, **opts)

        self.base_task_id = opts['base_task_id']
        self.hyper_parameters = hyper_parameters
        self.max_number_of_concurrent_tasks = opts['max_number_of_concurrent_tasks']
        self.execution_queue = opts['execution_queue']
        self.objective_metric = Objective(
            title=opts['objective_metric_title'], series=opts['objective_metric_series'],
            order='min' if opts['objective_metric_sign'] in ('min', 'min_global') else 'max',
            extremum=opts['objective_metric_sign'].endswith('_global'))
        # if optimizer_class is an instance, use it as is.
        if type(optimizer_class) != type:
            self.optimizer = optimizer_class
        else:
            self.optimizer = optimizer_class(
                base_task_id=opts['base_task_id'], hyper_parameters=hyper_parameters,
                objective_metric=self.objective_metric, execution_queue=opts['execution_queue'],
                num_concurrent_workers=opts['max_number_of_concurrent_tasks'],
                compute_time_limit=opts['compute_time_limit'], **opts.get('optimizer_kwargs', {}))
        self.optimizer.set_optimizer_task(self._task)
        self.optimization_timeout = None
        self.optimization_start_time = None
        self._thread = None
        self._stop_event = None
        self._report_period_min = 5.
        self._thread_reporter = None
        self._experiment_completed_cb = None
        if self._task:
            self.optimizer.set_job_default_parent(self._task.id)
        self.set_time_limit(in_minutes=opts['optimization_time_limit'])

    def get_num_active_experiments(self):
        # type: () -> int
        """
        Return the number of current active experiments.

        :return: The number of active experiments.
        """
        if not self.optimizer:
            return 0
        return len(self.optimizer.get_running_jobs())

    def get_active_experiments(self):
        # type: () -> Sequence[Task]
        """
        Return a list of Tasks of the current active experiments.

        :return: A list of Task objects, representing the current active experiments.
        """
        if not self.optimizer:
            return []
        return [j.task for j in self.optimizer.get_running_jobs()]

    def start(self, job_complete_callback=None):
        # type: (Optional[Callable[[str, float, int, dict, str], None]]) -> bool
        """
        Start the HyperParameterOptimizer controller. If the calling process is stopped, then the controller stops
        as well.

        :param Callable job_complete_callback: Callback function, called when a job is completed.

            .. code-block:: py

                def job_complete_callback(
                    job_id,                 # type: str
                    objective_value,        # type: float
                    objective_iteration,    # type: int
                    job_parameters,         # type: dict
                    top_performance_job_id  # type: str
                ):
                    pass

        :return: True, if the controller started. False, if the controller did not start.

        """
        if not self.optimizer:
            return False

        if self._thread:
            return True

        self.optimization_start_time = time()
        self._experiment_completed_cb = job_complete_callback
        self._stop_event = Event()
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()
        self._thread_reporter = Thread(target=self._report_daemon)
        self._thread_reporter.daemon = True
        self._thread_reporter.start()
        return True

    def stop(self, timeout=None, wait_for_reporter=True):
        # type: (Optional[float], Optional[bool]) -> ()
        """
        Stop the HyperParameterOptimizer controller and the optimization thread.

        :param float timeout: Wait timeout for the optimization thread to exit (minutes).
            The default is ``None``, indicating do not wait terminate immediately.
        :param wait_for_reporter: Wait for reporter to flush data.
        """
        if not self._thread or not self._stop_event or not self.optimizer:
            if self._thread_reporter and wait_for_reporter:
                self._thread_reporter.join()
            return

        _thread = self._thread
        self._stop_event.set()
        self.optimizer.stop()

        # wait for optimizer thread
        if timeout is not None:
            _thread.join(timeout=timeout * 60.)

        # stop all running tasks:
        for j in self.optimizer.get_running_jobs():
            j.abort()

        # clear thread
        self._thread = None
        if wait_for_reporter:
            # wait for reporter to flush
            self._thread_reporter.join()

    def is_active(self):
        # type: () -> bool
        """
        Is the optimization procedure active (still running)

        The values are:

        - ``True`` - The optimization procedure is active (still running).
        - ``False`` - The optimization procedure is not active (not still running).

        .. note::
            If the daemon thread has not yet started, ``is_active`` returns ``True``.

        :return: A boolean indicating whether the optimization procedure is active (still running) or stopped.
        """
        return self._stop_event is None or self._thread is not None

    def is_running(self):
        # type: () -> bool
        """
        Is the optimization controller is running

        The values are:

        - ``True`` - The optimization procedure is running.
        - ``False`` - The optimization procedure is running.

        :return: A boolean indicating whether the optimization procedure is active (still running) or stopped.
        """
        return self._thread is not None

    def wait(self, timeout=None):
        # type: (Optional[float]) -> bool
        """
        Wait for the optimizer to finish.

        .. note::
            This method does not stop the optimizer. Call :meth:`stop` to terminate the optimizer.

        :param float timeout: The timeout to wait for the optimization to complete (minutes).
            If ``None``, then wait until we reached the timeout, or optimization completed.

        :return: True, if the optimization finished. False, if the optimization timed out.

        """
        if not self.is_running():
            return True

        if timeout is not None:
            timeout *= 60.
        else:
            timeout = max(0, self.optimization_timeout - self.optimization_start_time) \
                if self.optimization_timeout else None

        _thread = self._thread

        _thread.join(timeout=timeout)
        if _thread.is_alive():
            return False

        return True

    def set_time_limit(self, in_minutes=None, specific_time=None):
        # type: (Optional[float], Optional[datetime]) -> ()
        """
        Set a time limit for the HyperParameterOptimizer controller. If we reached the time limit, stop the optimization
        process. If ``specific_time`` is provided, use it; otherwise, use the ``in_minutes``.

        :param float in_minutes: The maximum processing time from current time (minutes).
        :param datetime specific_time: The specific date/time limit.
        """
        if specific_time:
            self.optimization_timeout = specific_time.timestamp()
        else:
            self.optimization_timeout = (float(in_minutes) * 60.) + time() if in_minutes else None

    def get_time_limit(self):
        # type: () -> datetime
        """
        Return the controller optimization time limit.

        :return: The absolute datetime limit of the controller optimization process.
        """
        return datetime.fromtimestamp(self.optimization_timeout)

    def elapsed(self):
        # type: () -> float
        """
        Return minutes elapsed from controller stating time stamp.

        :return: The minutes from controller start time. A negative value means the process has not started yet.
        """
        if self.optimization_start_time is None:
            return -1.0
        return (time() - self.optimization_start_time) / 60.

    def reached_time_limit(self):
        # type: () -> bool
        """
        Did the optimizer reach the time limit

        The values are:

        - ``True`` - The time limit passed.
        - ``False`` - The time limit did not pass.

        This method returns immediately, it does not wait for the optimizer.

        :return: True, if optimizer is running and we passed the time limit, otherwise returns False.
        """
        if self.optimization_start_time is None:
            return False
        if not self.is_running():
            return False

        return time() > self.optimization_timeout

    def get_top_experiments(self, top_k):
        # type: (int) -> Sequence[Task]
        """
        Return a list of Tasks of the top performing experiments, based on the controller ``Objective`` object.

        :param int top_k: The number of Tasks (experiments) to return.

        :return: A list of Task objects, ordered by performance, where index 0 is the best performing Task.
        """
        if not self.optimizer:
            return []
        return self.optimizer.get_top_experiments(top_k=top_k)

    def get_optimizer(self):
        # type: () -> SearchStrategy
        """
        Return the currently used optimizer object.

        :return: The SearchStrategy object used.
        """
        return self.optimizer

    def set_default_job_class(self, job_class):
        # type: (TrainsJob) -> ()
        """
        Set the Job class to use when the optimizer spawns new Jobs.

        :param TrainsJob job_class: The Job Class type.
        """
        self.optimizer.set_job_class(job_class)

    def set_report_period(self, report_period_minutes):
        # type: (float) -> ()
        """
        Set reporting period for the accumulated objective report (minutes). This report is sent on the Optimizer Task,
        and collects the Objective metric from all running jobs.

        :param float report_period_minutes: The reporting period (minutes). The default is once every 10 minutes.
        """
        self._report_period_min = float(report_period_minutes)

    def _connect_args(self, optimizer_class=None, hyper_param_configuration=None, **kwargs):
        # type: (SearchStrategy, dict, Any) -> (SearchStrategy, list, dict)
        if not self._task:
            logger.warning('Auto Connect turned on but no Task was found, '
                           'hyper-parameter optimization argument logging disabled')
            return optimizer_class, hyper_param_configuration, kwargs

        configuration_dict = {'parameter_optimization_space': [c.to_dict() for c in hyper_param_configuration]}
        self._task.connect_configuration(configuration_dict)
        # this is the conversion back magic:
        configuration_dict = {'parameter_optimization_space': [
            Parameter.from_dict(c) for c in configuration_dict['parameter_optimization_space']]}

        arguments = {'opt': kwargs}
        if type(optimizer_class) != type:
            logger.warning('Auto Connect optimizer_class disabled, {} is already instantiated'.format(optimizer_class))
            self._task.connect(arguments)
        else:
            arguments['opt']['optimizer_class'] = str(optimizer_class).split('.')[-1][:-2] \
                if not isinstance(optimizer_class, str) else optimizer_class
            self._task.connect(arguments)
            # this is the conversion back magic:
            original_class = optimizer_class
            optimizer_class = arguments['opt'].pop('optimizer_class', None)
            if optimizer_class == 'RandomSearch':
                optimizer_class = RandomSearch
            elif optimizer_class == 'GridSearch':
                optimizer_class = GridSearch
            elif optimizer_class == 'OptimizerBOHB':
                from .hpbandster import OptimizerBOHB
                optimizer_class = OptimizerBOHB
            elif optimizer_class == 'OptimizerOptuna':
                from .optuna import OptimizerOptuna
                optimizer_class = OptimizerOptuna
            else:
                logger.warning("Could not resolve optimizer_class {} reverting to original class {}".format(
                    optimizer_class, original_class))
                optimizer_class = original_class

        return optimizer_class, configuration_dict['parameter_optimization_space'], arguments['opt']

    def _daemon(self):
        # type: () -> ()
        """
        Implement the main pooling thread, calling loop every ``self.pool_period_minutes`` minutes.
        """
        self.optimizer.start()
        self._thread = None

    def _report_daemon(self):
        # type: () -> ()
        title, series = self.objective_metric.get_objective_metric()
        title = '{}/{}'.format(title, series)
        counter = 0
        completed_jobs = dict()
        task_logger = None
        cur_completed_jobs = set()

        while self._thread is not None:
            timeout = self.optimization_timeout - time() if self.optimization_timeout else 0.

            if timeout >= 0:
                timeout = min(self._report_period_min * 60., timeout if timeout else self._report_period_min * 60.)
                # make sure that we have the first report fired before we actually go to sleep, wait for 15 sec.
                if counter <= 0:
                    timeout = 15
                print('Progress report #{} completed, sleeping for {} minutes'.format(counter, timeout / 60.))
                if self._stop_event.wait(timeout=timeout):
                    # wait for one last report
                    timeout = -1

            counter += 1

            # get task to report on.
            if self._task or Task.current_task():
                task_logger = (self._task or Task.current_task()).get_logger()

                # do some reporting

                self._report_remaining_budget(task_logger, counter)

                if self.optimizer.budget.compute_time.used and self.optimizer.budget.compute_time.used >= 1.0:
                    # Reached compute time limit
                    timeout = -1

                self._report_resources(task_logger, counter)
                # collect a summary of all the jobs and their final objective values
                cur_completed_jobs = set(self.optimizer.get_created_jobs_ids().keys()) - \
                    {j.task_id() for j in self.optimizer.get_running_jobs()}
                self._report_completed_status(completed_jobs, cur_completed_jobs, task_logger, title)

                self._report_completed_tasks_best_results(set(completed_jobs.keys()), task_logger, title, counter)
            # if we should leave, stop everything now.
            if timeout < 0:
                # we should leave
                self.stop(wait_for_reporter=False)
                return
        if task_logger and counter:
            counter += 1
            self._report_remaining_budget(task_logger, counter)
            self._report_resources(task_logger, counter)
            self._report_completed_status(completed_jobs, cur_completed_jobs, task_logger, title, force=True)
            self._report_completed_tasks_best_results(set(completed_jobs.keys()), task_logger, title, counter)

    def _report_completed_status(self, completed_jobs, cur_completed_jobs, task_logger, title, force=False):
        best_experiment = float('-inf'), None
        if force or cur_completed_jobs != set(completed_jobs.keys()):
            pairs = []
            labels = []
            created_jobs = copy(self.optimizer.get_created_jobs_ids())
            id_status = {j_id: j_run.status() for j_id, j_run in self.optimizer.get_created_jobs_tasks().items()}
            for i, (job_id, params) in enumerate(created_jobs.items()):
                value = self.objective_metric.get_objective(job_id)
                if job_id in completed_jobs:
                    if value != completed_jobs[job_id][0]:
                        iteration_value = self.objective_metric.get_current_raw_objective(job_id)
                        completed_jobs[job_id] = (
                            value,
                            iteration_value[0] if iteration_value else -1,
                            copy(dict(**params, **{"status": id_status.get(job_id)})))
                    elif completed_jobs.get(job_id):
                        completed_jobs[job_id] = (completed_jobs[job_id][0],
                                                  completed_jobs[job_id][1],
                                                  copy(dict(**params, **{"status": id_status.get(job_id)})))
                    pairs.append((i, completed_jobs[job_id][0]))
                    labels.append(str(completed_jobs[job_id][2])[1:-1])
                else:
                    if value is not None:
                        pairs.append((i, value))
                        labels.append(str(params)[1:-1])
                        iteration_value = self.objective_metric.get_current_raw_objective(job_id)
                        completed_jobs[job_id] = (
                            value,
                            iteration_value[0] if iteration_value else -1,
                            copy(dict(**params, **{"status": id_status.get(job_id)})))
                        # callback new experiment completed
                        if self._experiment_completed_cb:
                            normalized_value = self.objective_metric.get_normalized_objective(job_id)
                            if normalized_value is not None and normalized_value > best_experiment[0]:
                                best_experiment = normalized_value, job_id
                            c = completed_jobs[job_id]
                            self._experiment_completed_cb(job_id, c[0], c[1], c[2], best_experiment[1])

            if pairs:
                print('Updating job performance summary plot/table')

                # update scatter plot
                task_logger.report_scatter2d(
                    title='optimization', series=title,
                    scatter=pairs, iteration=0, labels=labels,
                    mode='markers', xaxis='job #', yaxis='objective')

                # update summary table
                if pd:
                    index = list(completed_jobs.keys())
                    table = {'objective': [completed_jobs[i][0] for i in index],
                             'iteration': [completed_jobs[i][1] for i in index]}
                    columns = set([c for k, v in completed_jobs.items() for c in v[2].keys()])
                    for c in sorted(columns):
                        table.update({c: [completed_jobs[i][2].get(c, '') for i in index]})

                    df = pd.DataFrame(table, index=index)
                    df.sort_values(by='objective', ascending=bool(self.objective_metric.sign < 0), inplace=True)
                    df.index.name = 'task id'
                    task_logger.report_table(
                        "summary", "job", 0, table_plot=df,
                        extra_layout={"title": "objective: {}".format(title)})

    def _report_remaining_budget(self, task_logger, counter):
        # noinspection PyBroadException
        try:
            budget = self.optimizer.budget.to_dict()
        except Exception:
            budget = {}
        # report remaining budget
        for budget_part, value in budget.items():
            task_logger.report_scalar(
                title='remaining budget', series='{} %'.format(budget_part),
                iteration=counter, value=round(100 - value['used'] * 100., ndigits=1))
        if self.optimization_timeout and self.optimization_start_time:
            task_logger.report_scalar(
                title='remaining budget', series='time %',
                iteration=counter,
                value=round(100 - (100. * (time() - self.optimization_start_time) /
                                   (self.optimization_timeout - self.optimization_start_time)), ndigits=1)
            )

    def _report_completed_tasks_best_results(self, completed_jobs, task_logger, title, counter):
        # type: (Set[str], Logger, str, int) -> ()
        if completed_jobs:
            value_func, series_name = (max, "max") if self.objective_metric.get_objective_sign() > 0 else \
                (min, "min")
            latest_completed, obj_values = self._get_latest_completed_task_value(completed_jobs, series_name)
            if latest_completed:
                val = value_func(obj_values)
                task_logger.report_scalar(
                    title=title,
                    series=series_name,
                    iteration=counter,
                    value=val)
                task_logger.report_scalar(
                    title=title,
                    series="last reported",
                    iteration=counter,
                    value=latest_completed)

    def _report_resources(self, task_logger, iteration):
        # type: (Logger, int) -> ()
        self._report_active_workers(task_logger, iteration)
        self._report_tasks_status(task_logger, iteration)

    def _report_active_workers(self, task_logger, iteration):
        # type: (Logger, int) -> ()
        cur_task = self._task or Task.current_task()
        res = cur_task.send(workers_service.GetAllRequest())
        response = res.wait()
        if response.ok():
            all_workers = response
            queue_workers = len(
                [
                    worker.get("id")
                    for worker in all_workers.response_data.get("workers")
                    for q in worker.get("queues")
                    if q.get("name") == self.execution_queue
                ]
            )
            task_logger.report_scalar(title="resources",
                                      series="queue workers",
                                      iteration=iteration,
                                      value=queue_workers)

    def _report_tasks_status(self, task_logger, iteration):
        # type: (Logger, int) -> ()
        tasks_status = {"running tasks": 0, "pending tasks": 0}
        for job in self.optimizer.get_running_jobs():
            if job.is_running():
                tasks_status["running tasks"] += 1
            else:
                tasks_status["pending tasks"] += 1
        for series, val in tasks_status.items():
            task_logger.report_scalar(
                title="resources", series=series,
                iteration=iteration, value=val)

    def _get_latest_completed_task_value(self, cur_completed_jobs, series_name):
        # type: (Set[str], str) -> (float, List[float])
        completed_value = None
        latest_completed = None
        obj_values = []
        cur_task = self._task or Task.current_task()
        for j in cur_completed_jobs:
            res = cur_task.send(tasks_services.GetByIdRequest(task=j))
            response = res.wait()
            if not response.ok() or response.response_data["task"].get("status") != Task.TaskStatusEnum.completed:
                continue
            completed_time = datetime.strptime(response.response_data["task"]["completed"].partition("+")[0],
                                               "%Y-%m-%dT%H:%M:%S.%f")
            completed_time = completed_time.timestamp()
            completed_values = self._get_last_value(response)
            obj_values.append(completed_values['max_value'] if series_name == "max" else completed_values['min_value'])
            if not latest_completed or completed_time > latest_completed:
                latest_completed = completed_time
                completed_value = completed_values['value']
        return completed_value, obj_values

    def _get_last_value(self, response):
        metrics, title, series, values = TrainsJob.get_metric_req_params(self.objective_metric.title,
                                                                         self.objective_metric.series)
        last_values = response.response_data["task"]['last_metrics'][title][series]
        return last_values
