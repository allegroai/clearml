import hashlib
import json
from copy import copy
from datetime import datetime
from itertools import product
from logging import getLogger
from threading import Thread, Event
from time import time
from typing import Union, Any, Sequence, Optional, Mapping, Callable

from .job import TrainsJob
from .parameters import Parameter
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
    Objective class to maximize/minimize over all experiments
    Class will sample specific scalar from all experiments, and maximize/minimize
    over single scalar (i.e. title and series combination)

    Used by the SearchStrategy/HyperParameterOptimizer in the strategy search algorithm
    """

    def __init__(self, title, series, order='max', extremum=False):
        # type: (str, str, str, bool) -> ()
        """
        Construct objective object that will return the scalar value for a specific task ID

        :param str title: Scalar graph title to sample from
        :param str series: Scalar series title to sample from
        :param str order: Either "max" or "min" , setting for maximizing/minimizing the objective scalar value
        :param bool extremum: Default False, which will bring the last value reported for a specific Task
            If True, return the global minimum / maximum reported metric value
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
        Return a specific task scalar value based on the objective settings (title/series)

        :param str task_id: Task id to retrieve scalar from (or TrainsJob object)
        :return float: scalar value
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
        Return the current raw value (without sign normalization) of the objective

        :param str task: Task or Job to retrieve scalar from (or TrainsJob object)
        :return tuple: (iteration, value) if metric does not exist return None
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
        Return the sign of the objective (i.e. +1 if maximizing, and -1 if minimizing)

        :return float: objective function sign
        """
        return self.sign

    def get_objective_metric(self):
        # type: () -> (str, str)
        """
        Return the metric title, series pair of the objective

        :return (str, str): return (title, series)
        """
        return self.title, self.series

    def get_normalized_objective(self, task_id):
        # type: (Union[str, Task, TrainsJob]) -> Optional[float]
        """
        Return a normalized task scalar value based on the objective settings (title/series)
        I.e. objective is always to maximize the returned value

        :param str task_id: Task id to retrieve scalar from
        :return float: normalized scalar value
        """
        objective = self.get_objective(task_id=task_id)
        if objective is None:
            return None
        # normalize value so we always look for the highest objective value
        return self.sign * objective

    def _get_last_metrics_encode_field(self):
        # type: () -> str
        """
        Return encoded representation of title/series metric

        :return str: string representing the objective title/series
        """
        if not self._metric:
            title = hashlib.md5(str(self.title).encode('utf-8')).hexdigest()
            series = hashlib.md5(str(self.series).encode('utf-8')).hexdigest()
            self._metric = title, series
        return '{}last_metrics.{}.{}.{}'.format(
            '-' if self.sign < 0 else '', self._metric[0], self._metric[1],
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
        if jobs:
            current_budget['jobs'] = {'limit': self.jobs.limit, 'used': jobs}
        iterations = self.iterations.used
        if iterations:
            current_budget['iterations'] = {'limit': self.iterations.limit, 'used': iterations}
        compute_time = self.compute_time.used
        if compute_time:
            current_budget['compute_time'] = {'limit': self.compute_time.limit, 'used': compute_time}
        return current_budget


class SearchStrategy(object):
    """
    Base Search strategy class, inherit to implement your custom strategy
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
            max_iteration_per_job=None,  # type: Optional[int]
            total_max_jobs=None,  # type: Optional[int]
            **_  # type: Any
    ):
        # type: (...) -> ()
        """
        Initialize a search strategy optimizer

        :param str base_task_id: Task ID (str)
        :param list hyper_parameters: list of Parameter objects to optimize over
        :param Objective objective_metric: Objective metric to maximize / minimize
        :param str execution_queue: execution queue to use for launching Tasks (experiments).
        :param int num_concurrent_workers: Limit number of concurrent running machines
        :param float pool_period_min: time in minutes between two consecutive pools
        :param float time_limit_per_job: Optional, maximum execution time per single job in minutes,
            when time limit is exceeded job is aborted
        :param int max_iteration_per_job: Optional, maximum iterations (of the objective metric)
            per single job, when exceeded job is aborted.
        :param int total_max_jobs: total maximum jobs for the optimization process. Default None, unlimited
        """
        super(SearchStrategy, self).__init__()
        self._base_task_id = base_task_id
        self._hyper_parameters = hyper_parameters
        self._objective_metric = objective_metric
        self._execution_queue = execution_queue
        self._num_concurrent_workers = num_concurrent_workers
        self.pool_period_minutes = pool_period_min
        self.time_limit_per_job = time_limit_per_job
        self.max_iteration_per_job = max_iteration_per_job
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
            compute_time_limit=self.total_max_jobs * self.time_limit_per_job if
            self.time_limit_per_job and self.total_max_jobs else None,
            iterations_limit=self.total_max_jobs * self.max_iteration_per_job if
            self.max_iteration_per_job and self.total_max_jobs else None
        )
        self._validate_base_task()

    def start(self):
        # type: () -> ()
        """
        Start the Optimizer controller function loop()
        If the calling process is stopped, the controller will stop as well.

        Notice: This function returns only after optimization is completed! or stop() was called.
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
        Stop the current running optimization loop,
        Called from a different thread than the start()
        """
        self._stop_event.set()

    def process_step(self):
        # type: () -> bool
        """
        Abstract helper function, not a must to implement, default use in start default implementation
        Main optimization loop, called from the daemon thread created by start()
        - Call monitor job on every TrainsJob in jobs:
            - Check the performance or elapsed time, then decide if to kill the jobs
        - Call create_job:
            - Check if we have spare jpb slots
            - If yes: call create a new job based on previous tested experiments

        :return bool: True to continue the optimization and False to immediately stop
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
        Abstract helper function, not a must to implement, default use in process_step default implementation
        Create a new job if needed. return the newly created job.
        If no job needs to be created, return None

        :return TrainsJob: newly created TrainsJob object or None if no TrainsJob created
        """
        return None

    def monitor_job(self, job):
        # type: (TrainsJob) -> bool
        """
        Helper function, not a must to implement, default use in process_step default implementation
        Check if the job needs to be aborted or already completed
        if return False, the job was aborted / completed, and should be taken off the current job list

        If there is a budget limitation,
        this call should update self.budget.compute_time.update() / self.budget.iterations.update()

        :param TrainsJob job: a TrainsJob object to monitor
        :return bool: If False, job is no longer relevant
        """
        abort_job = False

        if self.time_limit_per_job:
            elapsed = job.elapsed() / 60.
            if elapsed > 0:
                self.budget.compute_time.update(job.task_id(), elapsed)
                if elapsed > self.time_limit_per_job:
                    abort_job = True

        if self.max_iteration_per_job:
            iterations = self._get_job_iterations(job)
            if iterations > 0:
                self.budget.iterations.update(job.task_id(), iterations)
                if iterations > self.max_iteration_per_job:
                    abort_job = True

        if abort_job:
            job.abort()
            return False

        return not job.is_stopped()

    def get_running_jobs(self):
        # type: () -> Sequence[TrainsJob]
        """
        Return the current running TrainsJobs

        :return list: list of TrainsJob objects
        """
        return self._current_jobs

    def get_created_jobs_ids(self):
        # type: () -> Mapping[str, dict]
        """
        Return a task ids dict created ny this optimizer until now, including completed and running jobs.
        The values of the returned dict are the parameters used in the specific job

        :return dict: dict of task ids (str) as keys, and their parameters dict as value
        """
        return self._created_jobs_ids

    def get_top_experiments(self, top_k):
        # type: (int) -> Sequence[Task]
        """
        Return a list of Tasks of the top performing experiments, based on the controller Objective object

        :param int top_k: Number of Tasks (experiments) to return
        :return list: List of Task objects, ordered by performance, where index 0 is the best performing Task.
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
        Return the metric title, series pair of the objective

        :return (str, str): return (title, series)
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
        Create a Job using the specified arguments, TrainsJob for details

        :return TrainsJob: Returns a newly created Job instance
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
        self._created_jobs_ids[new_job.task_id()] = parameter_override
        logger.info('Creating new Task: {}'.format(parameter_override))
        return new_job

    def set_job_class(self, job_class):
        # type: (TrainsJob) -> ()
        """
        Set the class to use for the helper_create_job function

        :param TrainsJob job_class: Job Class type
        """
        self._job_class = job_class

    def set_job_default_parent(self, job_parent_task_id):
        # type: (str) -> ()
        """
        Set the default parent for all Jobs created by the helper_create_job method
        :param str job_parent_task_id: Parent task id
        """
        self._job_parent_id = job_parent_task_id

    def set_job_naming_scheme(self, naming_function):
        # type: (Optional[Callable[[str, dict], str]]) -> ()
        """
        Set the function used to name a newly created job

        :param callable naming_function: naming_functor(base_task_name, argument_dict) -> str
        """
        self._naming_function = naming_function

    def _validate_base_task(self):
        # type: () -> ()
        """
        Check the base task exists and contains the requested objective metric and hyper parameters
        """
        # check if the task exists
        try:
            task = Task.get_task(task_id=self._base_task_id)
            self._base_task_name = task.name
        except ValueError:
            raise ValueError("Could not find base task id {}".format(self._base_task_id))
        # check if the hyper-parameters exist:
        task_parameters = task.get_parameters_as_dict()
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
        Helper function, return a list of tasks tagged automl with specific status ordered by sort_field

        :param str parent_task_id: Base Task ID (parent)
        :param status: Current status of requested tasks (in_progress, completed etc)
        :param str order_by: Field name to sort results.
            Examples:
                "-last_metrics.title.series.min"
                "last_metrics.title.series.max"
                "last_metrics.title.series.last"
                "execution.parameters.name"
                "updated"
        :param dict additional_filters: Additional task filters
        :return list(Task): List of Task objects
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
    Grid search strategy controller.
    Full grid sampling of every hyper-parameter combination
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
            max_iteration_per_job=None,  # type: Optional[int]
            total_max_jobs=None,  # type: Optional[int]
            **_  # type: Any
    ):
        # type: (...) -> ()
        """
        Initialize a grid search optimizer

        :param str base_task_id: Task ID (str)
        :param list hyper_parameters: list of Parameter objects to optimize over
        :param Objective objective_metric: Objective metric to maximize / minimize
        :param str execution_queue: execution queue to use for launching Tasks (experiments).
        :param int num_concurrent_workers: Limit number of concurrent running machines
        :param float pool_period_min: time in minutes between two consecutive pools
        :param float time_limit_per_job: Optional, maximum execution time per single job in minutes,
            when time limit is exceeded job is aborted
        :param int max_iteration_per_job: maximum iterations (of the objective metric)
            per single job, when exceeded job is aborted.
        :param int total_max_jobs: total maximum jobs for the optimization process. Default None, unlimited
        """
        super(GridSearch, self).__init__(
            base_task_id=base_task_id, hyper_parameters=hyper_parameters, objective_metric=objective_metric,
            execution_queue=execution_queue, num_concurrent_workers=num_concurrent_workers,
            pool_period_min=pool_period_min, time_limit_per_job=time_limit_per_job,
            max_iteration_per_job=max_iteration_per_job, total_max_jobs=total_max_jobs, **_)
        self._param_iterator = None

    def create_job(self):
        # type: () -> Optional[TrainsJob]
        """
        Create a new job if needed. return the newly created job.
        If no job needs to be created, return None

        :return TrainsJob: newly created TrainsJob object or None if no TrainsJob created
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
    Random search strategy controller.
    Random uniform sampling of hyper-parameters
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
            max_iteration_per_job=None,  # type: Optional[int]
            total_max_jobs=None,  # type: Optional[int]
            **_  # type: Any
    ):
        # type: (...) -> ()
        """
        Initialize a random search optimizer

        :param str base_task_id: Task ID (str)
        :param list hyper_parameters: list of Parameter objects to optimize over
        :param Objective objective_metric: Objective metric to maximize / minimize
        :param str execution_queue: execution queue to use for launching Tasks (experiments).
        :param int num_concurrent_workers: Limit number of concurrent running machines
        :param float pool_period_min: time in minutes between two consecutive pools
        :param float time_limit_per_job: Optional, maximum execution time per single job in minutes,
            when time limit is exceeded job is aborted
        :param int max_iteration_per_job: maximum iterations (of the objective metric)
            per single job, when exceeded job is aborted.
        :param int total_max_jobs: total maximum jobs for the optimization process. Default None, unlimited
        """
        super(RandomSearch, self).__init__(
            base_task_id=base_task_id, hyper_parameters=hyper_parameters, objective_metric=objective_metric,
            execution_queue=execution_queue, num_concurrent_workers=num_concurrent_workers,
            pool_period_min=pool_period_min, time_limit_per_job=time_limit_per_job,
            max_iteration_per_job=max_iteration_per_job, total_max_jobs=total_max_jobs, **_)
        self._hyper_parameters_collection = set()

    def create_job(self):
        # type: () -> Optional[TrainsJob]
        """
        Create a new job if needed. return the newly created job.
        If no job needs to be created, return None

        :return TrainsJob: newly created TrainsJob object or None if no TrainsJob created
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
    Hyper-parameter search controller. Cloning base experiment,
    changing arguments and trying to maximize/minimize the defined objective
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
            auto_connect_task=True,  # type: bool
            always_create_task=False,  # type: bool
            **optimizer_kwargs  # type: Any
    ):
        # type: (...) -> ()
        """
        Create a new hyper-parameter controller. The newly created object will launch and monitor the new experiments.

        :param str base_task_id: Task ID to be used as template experiment to optimize.
        :param list hyper_parameters: list of Parameter objects to optimize over
        :param str objective_metric_title: Objective metric title to maximize / minimize (example: 'validation')
        :param str objective_metric_series: Objective metric series to maximize / minimize (example: 'loss')
        :param str objective_metric_sign: Objective to maximize / minimize.
            Valid options: ['min', 'max', 'min_global', 'max_global']
            'min'/'max': Minimize/Maximize the last reported value for the specified title/series scalar
            'min_global'/'max_global': Minimize/Maximize the min/max value
                of *all* reported values for the specific title/series scalar
        :param class.SearchStrategy optimizer_class: SearchStrategy optimizer to use for the hyper-parameter search
        :param int max_number_of_concurrent_tasks: Maximum number of
            concurrent Tasks (experiment) running at the same time.
        :param str execution_queue: execution queue to use for launching Tasks (experiments).
        :param float optimization_time_limit: Maximum time (minutes) for the entire optimization process.
            Default is None, no time limit,
        :param bool auto_connect_task: If True optimization argument and configuration will be stored on the Task
            All arguments will be under the hyper-parameter section as 'opt/<arg>'
            and the hyper_parameters will stored in the task connect_configuration (see artifacts/hyper-parameter)
        :param bool always_create_task: If True there ts no current Task initialized,
            we create a new task names 'optimization' in the base_task_id project.
            otherwise we use the Task.current_task (if exists) to report statistics
        :param ** optimizer_kwargs: arguments passed directly to the optimizer constructor

            Example:

            .. code-block:: python
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
        self._task = Task.current_task()
        if not self._task and always_create_task:
            base_task = Task.get_task(task_id=self.base_task_id)
            self._task = Task.init(
                project_name=base_task.get_project_name(),
                task_name='Optimizing: {}'.format(base_task.name),
            )  # TODO: add task_type=controller

        opts = dict(
            base_task_id=base_task_id,
            objective_metric_title=objective_metric_title,
            objective_metric_series=objective_metric_series,
            objective_metric_sign=objective_metric_sign,
            max_number_of_concurrent_tasks=max_number_of_concurrent_tasks,
            execution_queue=execution_queue,
            optimization_time_limit=optimization_time_limit,
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
                num_concurrent_workers=opts['max_number_of_concurrent_tasks'], **opts.get('optimizer_kwargs', {}))
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
        Return the number of current active experiments

        :return int: number of active experiments
        """
        if not self.optimizer:
            return 0
        return len(self.optimizer.get_running_jobs())

    def get_active_experiments(self):
        # type: () -> Sequence[Task]
        """
        Return a list of Tasks of the current active experiments

        :return list: List of Task objects, representing the current active experiments
        """
        if not self.optimizer:
            return []
        return [j.task for j in self.optimizer.get_running_jobs()]

    def start(self, job_complete_callback=None):
        # type: (Optional[Callable[[str, float, int, dict, str], None]]) -> bool
        """
        Start the HyperParameterOptimizer controller.
        If the calling process is stopped, the controller will stop as well.

        :param Callable job_complete_callback: callback function, called when a job is completed.
            def job_complete_callback(
                job_id,                 # type: str
                objective_value,        # type: float
                objective_iteration,    # type: int
                job_parameters,         # type: dict
                top_performance_job_id  # type: str
            ):
                pass
        :return bool: If True the controller started
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

    def stop(self, timeout=None):
        # type: (Optional[float]) -> ()
        """
        Stop the HyperParameterOptimizer controller and  optimization thread,

        :param float timeout: Wait timeout in minutes for the optimization thread to exit.
            Default None, do not wait terminate immediately.
        """
        if not self._thread or not self._stop_event or not self.optimizer:
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
        # wait for reporter to flush
        self._thread_reporter.join()

    def is_active(self):
        # type: () -> bool
        """
        Return True if the optimization procedure is still running
        Note, if the daemon thread has not yet started, it will still return True

        :return bool: If False the optimization procedure stopped
        """
        return self._stop_event is None or self._thread is not None

    def is_running(self):
        # type: () -> bool
        """
        Return True if the optimization controller is running

        :return bool: If True if optimization procedure is active
        """
        return self._thread is not None

    def wait(self, timeout=None):
        # type: (Optional[float]) -> bool
        """
        Wait for the optimizer to finish.
        It will not stop the optimizer in any case. Call stop() to terminate the optimizer.

        :param float timeout: Timeout in minutes to wait for the optimization to complete
            If None, wait until we reached the timeout, or optimization completed.
        :return bool: True if optimization finished, False if timeout.
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
        Set a time limit for the HyperParameterOptimizer controller,
        i.e. if we reached the time limit, stop the optimization process

        :param float in_minutes: Set maximum processing time in minutes from current time
        :param datetime specific_time: Set specific date/time limit
        """
        if specific_time:
            self.optimization_timeout = specific_time.timestamp()
        else:
            self.optimization_timeout = (in_minutes * 60.) + time() if in_minutes else None

    def get_time_limit(self):
        # type: () -> datetime
        """
        Return the controller optimization time limit.

        :return datetime: Absolute datetime limit of the controller optimization process
        """
        return datetime.fromtimestamp(self.optimization_timeout)

    def elapsed(self):
        # type: () -> float
        """
        Return minutes elapsed from controller stating time stamp

        :return float: minutes from controller start time, negative value means the process has not started yet.
        """
        if self.optimization_start_time is None:
            return -1.0
        return (time() - self.optimization_start_time) / 60.

    def reached_time_limit(self):
        # type: () -> bool
        """
        Return True if we passed the time limit. Function returns immediately, it does not wait for the optimizer.

        :return bool: Return True, if optimizer is running and we passed the time limit, otherwise returns False.
        """
        if self.optimization_start_time is None:
            return False
        if not self.is_running():
            return False

        return time() > self.optimization_timeout

    def get_top_experiments(self, top_k):
        # type: (int) -> Sequence[Task]
        """
        Return a list of Tasks of the top performing experiments, based on the controller Objective object

        :param int top_k: Number of Tasks (experiments) to return
        :return list: List of Task objects, ordered by performance, where index 0 is the best performing Task.
        """
        if not self.optimizer:
            return []
        return self.optimizer.get_top_experiments(top_k=top_k)

    def get_optimizer(self):
        # type: () -> SearchStrategy
        """
        Return the currently used optimizer object

        :return SearchStrategy: Used SearchStrategy object
        """
        return self.optimizer

    def set_default_job_class(self, job_class):
        # type: (TrainsJob) -> ()
        """
        Set the Job class to use when the optimizer spawns new Jobs

        :param TrainsJob job_class: Job Class type
        """
        self.optimizer.set_job_class(job_class)

    def set_report_period(self, report_period_minutes):
        # type: (float) -> ()
        """
        Set reporting period in minutes, for the accumulated objective report
        This report is sent on the Optimizer Task, and collects objective metric from all running jobs.

        :param float report_period_minutes: Reporting period in minutes. Default once every 10 minutes.
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
            else:
                logger.warning("Could not resolve optimizer_class {} reverting to original class {}".format(
                    optimizer_class, original_class))
                optimizer_class = original_class

        return optimizer_class, configuration_dict['parameter_optimization_space'], arguments['opt']

    def _daemon(self):
        # type: () -> ()
        """
        implement the main pooling thread, calling loop every self.pool_period_minutes minutes
        """
        self.optimizer.start()
        self._thread = None

    def _report_daemon(self):
        # type: () -> ()
        worker_to_series = {}
        title, series = self.objective_metric.get_objective_metric()
        title = '{}/{}'.format(title, series)
        series = 'machine:'
        counter = 0
        completed_jobs = dict()
        best_experiment = float('-inf'), None

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

                # running objective, per machine
                running_job_ids = set()
                for j in self.optimizer.get_running_jobs():
                    worker = j.worker()
                    running_job_ids.add(j.task_id())
                    if worker not in worker_to_series:
                        worker_to_series[worker] = len(worker_to_series) + 1
                    machine_id = worker_to_series[worker]
                    value = self.objective_metric.get_objective(j)
                    if value is not None:
                        task_logger.report_scalar(
                            title=title, series='{}{}'.format(series, machine_id),
                            iteration=counter, value=value)

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

                # collect a summary of all the jobs and their final objective values
                cur_completed_jobs = set(self.optimizer.get_created_jobs_ids().keys()) - running_job_ids
                if cur_completed_jobs != set(completed_jobs.keys()):
                    pairs = []
                    labels = []
                    created_jobs = copy(self.optimizer.get_created_jobs_ids())
                    for i, (job_id, params) in enumerate(created_jobs.items()):
                        if job_id in completed_jobs:
                            pairs.append((i, completed_jobs[job_id][0]))
                            labels.append(str(completed_jobs[job_id][2])[1:-1])
                        else:
                            value = self.objective_metric.get_objective(job_id)
                            if value is not None:
                                pairs.append((i, value))
                                labels.append(str(params)[1:-1])
                                iteration_value = self.objective_metric.get_current_raw_objective(job_id)
                                completed_jobs[job_id] = (
                                    value, iteration_value[0] if iteration_value else -1, copy(params))
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
                            task_logger.report_table("summary", "job", 0, table_plot=df)

            # if we should leave, stop everything now.
            if timeout < 0:
                # we should leave
                self.stop()
                return
