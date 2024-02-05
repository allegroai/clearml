import hashlib
import json
import six
from copy import copy, deepcopy
from datetime import datetime
from itertools import product
from logging import getLogger
from threading import Thread, Event
from time import time
from typing import List, Set, Union, Any, Sequence, Optional, Mapping, Callable, Tuple
from abc import ABC, abstractmethod

from .job import ClearmlJob, LocalClearmlJob
from .parameters import Parameter
from ..backend_interface.util import get_or_create_project, datetime_from_isoformat
from ..logger import Logger
from ..backend_api.services import workers as workers_service, tasks as tasks_service, events as events_service
from ..task import Task

logger = getLogger('clearml.automation.optimization')


class _ObjectiveInterface(ABC):
    @abstractmethod
    def get_objective(self, task_id):
        pass

    @abstractmethod
    def get_current_raw_objective(self, task):
        pass

    @abstractmethod
    def get_objective_sign(self):
        pass

    @abstractmethod
    def get_objective_metric(self):
        pass

    @abstractmethod
    def get_normalized_objective(self, task_id):
        pass

    @abstractmethod
    def get_top_tasks(self, top_k, optimizer_task_id=None, task_filter=None):
        pass


class Objective(_ObjectiveInterface):
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
        self.sign = -1 if (isinstance(order, str) and order.lower().strip() == 'min') else 1
        self._metric = None
        self.extremum = extremum

    def get_objective(self, task_id):
        # type: (Union[str, Task, ClearmlJob]) -> Optional[float]
        """
        Return a specific task scalar value based on the objective settings (title/series).

        :param str task_id: The Task ID to retrieve scalar from (or ``ClearMLJob`` object).

        :return: The scalar value.
        """
        # create self._metric
        self._get_last_metrics_encode_field()

        if isinstance(task_id, Task):
            task_id = task_id.id
        elif isinstance(task_id, ClearmlJob):
            task_id = task_id.task_id()

        # noinspection PyBroadException, Py
        try:
            # noinspection PyProtectedMember
            task = Task._query_tasks(
                task_ids=[task_id], only_fields=['last_metrics.{}.{}'.format(self._metric[0], self._metric[1])])[0]
        except Exception:
            return None

        metrics = task.last_metrics
        if not metrics:
            return None

        # noinspection PyBroadException
        try:
            values = metrics[self._metric[0]][self._metric[1]]
            if not self.extremum:
                return values['value']

            return values['min_value'] if self.sign < 0 else values['max_value']
        except Exception:
            return None

    def get_current_raw_objective(self, task):
        # type: (Union[ClearmlJob, Task]) -> (int, float)
        """
        Return the current raw value (without sign normalization) of the objective.

        :param str task: The Task or Job to retrieve scalar from (or ``ClearmlJob`` object).

        :return: Tuple(iteration, value) if, and only if, the metric exists. None if the metric does not exist.

        """
        if isinstance(task, Task):
            task_id = task.id
        elif isinstance(task, ClearmlJob):
            task_id = task.task_id()
        else:
            task_id = task

        if not task_id:
            raise ValueError("Task ID not provided")

        # send request
        # noinspection PyBroadException
        try:
            # noinspection PyProtectedMember
            res = Task._get_default_session().send(
                events_service.ScalarMetricsIterHistogramRequest(
                    task=task_id, key='iter', samples=None),
            )
        except Exception:
            res = None

        if not res:
            return None
        response = res.wait()
        if not response.ok() or not response.response_data:
            return None

        scalars = response.response_data
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
        # type: (Union[str, Task, ClearmlJob]) -> Optional[float]
        """
        Return a normalized task scalar value based on the objective settings (title/series).
        I.e. objective is always to maximize the returned value

        :param str task_id: The Task ID to retrieve scalar from.

        :return: Normalized scalar value.
        """
        objective = self.get_objective(task_id=task_id)
        if objective is None:
            return None
        # normalize value so we always look for the highest objective value
        return self.sign * objective

    def get_top_tasks(self, top_k, optimizer_task_id=None, task_filter=None):
        # type: (int, Optional[str], Optional[dict]) -> Sequence[Task]
        """
        Return a list of Tasks of the top performing experiments, based on the title/series objective.

        :param int top_k: The number of Tasks (experiments) to return.
        :param str optimizer_task_id: Parent optimizer Task ID
        :param dict task_filter: Optional task_filtering for the query

        :return: A list of Task objects, ordered by performance, where index 0 is the best performing Task.
        """
        task_filter = deepcopy(task_filter) if task_filter else {}
        task_filter.update({'page_size': int(top_k), 'page': 0})
        if optimizer_task_id:
            task_filter['parent'] = optimizer_task_id
        order_by = self._get_last_metrics_encode_field()
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
            return sum(self.current.values()) / float(self.limit)

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
    _job_class = ClearmlJob  # type: ClearmlJob

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
        self._job_project_id = None
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

        - Call monitor job on every ``ClearmlJob`` in jobs:

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
            if not new_job.launch(self._execution_queue):
                # error enqueuing Job, something wrong here
                continue
            self._num_jobs += 1
            self._current_jobs.append(new_job)
            self._pending_jobs.append(new_job)

        return bool(self._current_jobs)

    def create_job(self):
        # type: () -> Optional[ClearmlJob]
        """
        Abstract helper function. Implementation is not required. Default use in process_step default implementation
        Create a new job if needed. return the newly created job. If no job needs to be created, return ``None``.

        :return: A Newly created ClearmlJob object, or None if no ClearmlJob created.
        """
        return None

    def monitor_job(self, job):
        # type: (ClearmlJob) -> bool
        """
        Helper function, Implementation is not required. Default use in process_step default implementation.
        Check if the job needs to be aborted or already completed.

        If returns ``False``, the job was aborted / completed, and should be taken off the current job list.

        If there is a budget limitation, this call should update
        ``self.budget.compute_time.update`` / ``self.budget.iterations.update``

        :param ClearmlJob job: A ``ClearmlJob`` object to monitor.

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
            if iterations and iterations > 0:
                self.budget.iterations.update(job.task_id(), iterations)
                if iterations > self.max_iteration_per_job:
                    abort_job = True

        return abort_job

    def get_running_jobs(self):
        # type: () -> Sequence[ClearmlJob]
        """
        Return the current running ClearmlJob.

        :return: List of ClearmlJob objects.
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
        The values of the returned dict are the ClearmlJob.

        :return: dict of task IDs (str) as keys, and their ClearmlJob as values.
        """
        return {job_id: job_val[0] for job_id, job_val in self._created_jobs_ids.items()}

    def get_top_experiments(self, top_k):
        # type: (int) -> Sequence[Task]
        """
        Return a list of Tasks of the top performing experiments, based on the controller ``Objective`` object.

        :param int top_k: The number of Tasks (experiments) to return.

        :return: A list of Task objects, ordered by performance, where index 0 is the best performing Task.
        """
        return self._objective_metric.get_top_tasks(top_k=top_k,
                                                    optimizer_task_id=self._job_parent_id or self._base_task_id)

    def get_top_experiments_id_metrics_pair(self, top_k, all_metrics=False, only_completed=False):
        # type: (int, bool, bool) -> Sequence[(str, dict)]
        """
        Return a list of pairs (Task ID, scalar metric dict) of the top performing experiments.
        Order is based on the controller ``Objective`` object.

        :param int top_k: The number of Tasks (experiments) to return.
        :param all_metrics: Default False, only return the objective metric on the metrics dictionary.
            If True, return all scalar metrics of the experiment
        :param only_completed: return only completed Tasks. Default False.

        :return: A list of pairs (Task ID, metric values dict), ordered by performance,
        where index 0 is the best performing Task.
        Example w/ all_metrics=False:

        .. code-block:: py

            [
                ('0593b76dc7234c65a13a301f731958fa',
                    {
                        'accuracy per class/cat': {
                            'metric': 'accuracy per class',
                            'variant': 'cat',
                            'value': 0.119,
                            'min_value': 0.119,
                            'max_value': 0.782
                        },
                    }
                ),
            ]

        Example w/ all_metrics=True:

        .. code-block:: py

            [
                ('0593b76dc7234c65a13a301f731958fa',
                    {
                        'accuracy per class/cat': {
                            'metric': 'accuracy per class',
                            'variant': 'cat',
                            'value': 0.119,
                            'min_value': 0.119,
                            'max_value': 0.782
                        },
                        'accuracy per class/deer': {
                            'metric': 'accuracy per class',
                            'variant': 'deer',
                            'value': 0.219,
                            'min_value': 0.219,
                            'max_value': 0.282
                        },
                    }
                ),
            ]
        """
        additional_filters = dict(page_size=int(top_k), page=0)
        if only_completed:
            additional_filters['status'] = ["completed"]

        # noinspection PyProtectedMember
        top_tasks_ids_metric = self._get_child_tasks_ids(
            parent_task_id=self._job_parent_id or self._base_task_id,
            order_by=self._objective_metric._get_last_metrics_encode_field()[0],
            additional_filters=additional_filters,
            additional_fields=['last_metrics']
        )

        title_series = self._objective_metric.get_objective_metric() if not all_metrics else (None, None)
        titles = [ts[0] for ts in title_series]
        series = [ts[1] for ts in title_series]
        return [(i, {'{}/{}'.format(v['metric'], v['variant']): v
                     for variant in metric.values() for v in variant.values()
                     if all_metrics or (v['metric'] in titles and v['variant'] in series)}
                 ) for i, metric in top_tasks_ids_metric]

    def get_top_experiments_details(
            self,
            top_k,
            all_metrics=False,
            all_hyper_parameters=False,
            only_completed=False
    ):
        # type: (int, bool, bool, bool) -> Sequence[(str, dict)]
        """
        Return a list of dictionaries of the top performing experiments.
        Example: ``[{'task_id': Task-ID, 'metrics': scalar-metric-dict, 'hyper_parameters': Hyper-Parameters},]``

        Order is based on the controller ``Objective`` object.

        :param int top_k: The number of Tasks (experiments) to return.
        :param all_metrics: Default False, only return the objective metric on the metrics dictionary.
            If True, return all scalar metrics of the experiment
        :param all_hyper_parameters: Default False. If True, return all the hyperparameters from all the sections.
        :param only_completed: return only completed Tasks. Default False.

        :return: A list of dictionaries ({task_id: '', hyper_parameters: {}, metrics: {}}), ordered by performance,
            where index 0 is the best performing Task.
            Example w/ all_metrics=False:

            .. code-block:: py

                [
                    {
                        task_id: '0593b76dc7234c65a13a301f731958fa',
                        hyper_parameters: {'General/lr': '0.03', 'General/batch_size': '32'},
                        metrics: {
                            'accuracy per class/cat': {
                                'metric': 'accuracy per class',
                                'variant': 'cat',
                                'value': 0.119,
                                'min_value': 0.119,
                                'max_value': 0.782
                            },
                        }
                    },
                ]

            Example w/ all_metrics=True:

            .. code-block:: py

                [
                    {
                        task_id: '0593b76dc7234c65a13a301f731958fa',
                        hyper_parameters: {'General/lr': '0.03', 'General/batch_size': '32'},
                        metrics: {
                            'accuracy per class/cat': {
                                'metric': 'accuracy per class',
                                'variant': 'cat',
                                'value': 0.119,
                                'min_value': 0.119,
                                'max_value': 0.782
                            },
                            'accuracy per class/deer': {
                                'metric': 'accuracy per class',
                                'variant': 'deer',
                                'value': 0.219,
                                'min_value': 0.219,
                                'max_value': 0.282
                            },
                        }
                    },
                ]
        """
        additional_filters = dict(page_size=int(top_k), page=0)
        if only_completed:
            additional_filters['status'] = ["completed"]

        # noinspection PyProtectedMember
        top_tasks_ids_metric_params = self._get_child_tasks_ids(
            parent_task_id=self._job_parent_id or self._base_task_id,
            order_by=self._objective_metric._get_last_metrics_encode_field()[
                0] if self._objective_metric.len == 1 else None,
            additional_filters=additional_filters,
            additional_fields=['last_metrics', 'hyperparams']
        )
        if self._objective_metric.len != 1:
            top_tasks_ids_metric_params_dict = {}
            for task in top_tasks_ids_metric_params:
                objective = self._objective_metric.get_objective(task[0])
                if objective is None or any(o is None for o in objective):
                    continue
                top_tasks_ids_metric_params_dict[task[0]] = (objective, task)
            # noinspection PyProtectedMember
            sorted_ids = self._objective_metric._sort_jobs_by_domination(top_tasks_ids_metric_params_dict)
            top_tasks_ids_metric_params = [top_tasks_ids_metric_params_dict[s][1] for s in sorted_ids]

        # get hp_parameters:
        hp_params = set(p.name for p in self._hyper_parameters)

        title_series = self._objective_metric.get_objective_metric() if not all_metrics else (None, None)
        titles = [ts[0] for ts in title_series]
        series = [ts[1] for ts in title_series]
        return [
            {
                'task_id': tid,
                'hyper_parameters': {
                    '{}/{}'.format(p.section, p.name): p.value
                    for params in (param_sections or {}).values() for p in (params or {}).values()
                    if all_hyper_parameters or '{}/{}'.format(p.section, p.name) in hp_params
                },
                'metrics': {
                    '{}/{}'.format(v['metric'], v['variant']): v for variant in metric.values()
                    for v in variant.values() if all_metrics or v['metric'] in titles and v['variant'] in series
                }
            } for tid, metric, param_sections in top_tasks_ids_metric_params
        ]

    def get_objective_metric(self):
        # type: () -> Union[Tuple[str, str], List[Tuple[str, str]]]
        """
        Return the metric title, series pair of the objective.

        :return: (title, series)
        """
        objective = self._objective_metric.get_objective_metric()
        return objective[0] if self._objective_metric.len == 1 else objective

    def helper_create_job(
            self,
            base_task_id,  # type: str
            parameter_override=None,  # type: Optional[Mapping[str, str]]
            task_overrides=None,  # type: Optional[Mapping[str, str]]
            tags=None,  # type: Optional[Sequence[str]]
            parent=None,  # type: Optional[str]
            **kwargs  # type: Any
    ):
        # type: (...) -> ClearmlJob
        """
        Create a Job using the specified arguments, ``ClearmlJob`` for details.

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
            name=name, comment=comment,
            project=self._job_project_id or self._get_task_project(parent or self._job_parent_id),
            **kwargs)
        self._created_jobs_ids[new_job.task_id()] = (new_job, parameter_override)
        logger.info('Creating new Task: {}'.format(parameter_override))
        return new_job

    def set_job_class(self, job_class):
        # type: (ClearmlJob) -> ()
        """
        Set the class to use for the :meth:`helper_create_job` function.

        :param ClearmlJob job_class: The Job Class type.
        """
        self._job_class = job_class

    def set_job_default_parent(self, job_parent_task_id, project_name=None):
        # type: (Optional[str], Optional[str]) -> ()
        """
        Set the default parent for all Jobs created by the :meth:`helper_create_job` method.

        :param str job_parent_task_id: The parent Task ID.
        :param str project_name: If specified, create the jobs in the specified project
        """
        self._job_parent_id = job_parent_task_id
        # noinspection PyProtectedMember
        self._job_project_id = get_or_create_project(
            session=Task._get_default_session(), project_name=project_name, description='HPO process spawned Tasks') \
            if project_name else None

    def set_job_naming_scheme(self, naming_function):
        # type: (Optional[Callable[[str, dict], str]]) -> ()
        """
        Set the function used to name a newly created job.

        :param callable naming_function: Callable function for naming a newly created job.
            Use the following format:

            .. code-block:: py

               naming_functor(base_task_name, argument_dict) -> str

        """
        self._naming_function = naming_function

    def set_optimizer_task(self, task):
        # type: (Task) -> ()
        """
        Set the optimizer task object to be used to store/generate reports on the optimization process.
        Usually this is the current task of this process.

        :param Task task: The optimizer`s current Task.
        """
        self._optimizer_task = task

    def _validate_base_task(self):
        # type: () -> ()
        """
        Check the base task exists and contains the requested Objective metric and hyperparameters.
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
        # type: (Union[ClearmlJob, Task]) -> int
        iteration_value = self._objective_metric.get_current_raw_objective(job)
        if iteration_value is not None and any(iv is not None and iv[0] is not None for iv in iteration_value):
            return max(iv[0] for iv in iteration_value if iv is not None)
        return -1

    @classmethod
    def _get_child_tasks_ids(
            cls,
            parent_task_id,  # type: str
            status=None,  # type: Optional[Union[Task.TaskStatusEnum], Sequence[Task.TaskStatusEnum]]
            order_by=None,  # type: Optional[str]
            additional_fields=None,  # type: Optional[Sequence[str]]
            additional_filters=None,  # type: Optional[dict]
    ):
        # type: (...) -> (Union[Sequence[str], Sequence[List]])
        """
        Helper function. Return a list of tasks is tagged automl, with specific ``status``, ordered by ``sort_field``.

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

        :param additional_fields: Optional, list of fields (str) to return next to the Task ID,
            this implies return value is a list of pairs
        :param dict additional_filters: The additional task filters.
        :return: A list of Task IDs (str)
        """
        task_filter = {
            'parent': parent_task_id,
            # 'tags': [cls._tag],
            # since we have auto archive we do not want to filter out archived tasks
            # 'system_tags': ['-archived'],
        }
        task_filter.update(additional_filters or {})

        if status:
            task_filter['status'] = status if isinstance(status, (tuple, list)) else [status]

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

        if additional_fields:
            task_filter['only_fields'] = list(set(list(additional_fields) + ['id']))

        # noinspection PyProtectedMember
        task_objects = Task._query_tasks(**task_filter)
        if not additional_fields:
            return [t.id for t in task_objects]
        return [[t.id] + [getattr(t, f, None) for f in additional_fields] for t in task_objects]

    @classmethod
    def _get_child_tasks(
            cls,
            parent_task_id,  # type: str
            status=None,  # type: Optional[Union[Task.TaskStatusEnum], Sequence[Task.TaskStatusEnum]]
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
        return [
            Task.get_task(task_id=t_id) for t_id in cls._get_child_tasks_ids(
                parent_task_id=parent_task_id,
                status=status,
                order_by=order_by,
                additional_filters=additional_filters)
        ]


class MultiObjective(_ObjectiveInterface):
    def __init__(self, title, series, order, extremum):
        self.title = title
        self.series = series
        self.order = order
        self.extremum = extremum
        self.objectives = []
        for title_, series_, order_, extremum_ in zip(title, series, order, extremum):
            self.objectives.append(Objective(title=title_, series=series_, order=order_, extremum=extremum_))
        self.len = len(self.objectives)

    def get_objective(self, task_id):
        # type: (Union[str, Task, ClearmlJob]) -> Optional[List[float]]
        """
        Return a specific task scalar values based on the objective settings (title/series).

        :param str task_id: The Task ID to retrieve scalar from (or ``ClearMLJob`` object).

        :return: The scalar values.
        """
        objective = [o.get_objective(task_id) for o in self.objectives]
        if any(o is None for o in objective):
            return None
        return objective

    def get_current_raw_objective(self, task):
        # type: (Union[ClearmlJob, Task]) -> Optional[List[Tuple[int, float]]]
        """
        Return the current raw value (without sign normalization) of each objective.

        :param str task: The Task or Job to retrieve scalar from (or ``ClearmlJob`` object).
        :return: List[Optional[Tuple(iteration, value)]]. None if the metric does not exist.
        """
        objective = [o.get_current_raw_objective(task) for o in self.objectives]
        if any(o is None for o in objective):
            return None
        return objective

    def get_objective_sign(self):
        # type: () -> List[float]
        """
        Return the sign of the objectives.

        - ``+1`` - If maximizing
        - ``-1`` - If minimizing

        :return: Objective function signs.
        """
        return [o.get_objective_sign() for o in self.objectives]

    def get_normalized_objective(self, task_id):
        # type: (Union[str, Task, ClearmlJob]) -> Optional[List[float]]
        """
        Return a normalized task scalar values based on the objective settings (title/series).
        I.e. objective is always to maximize the returned value

        :param str task_id: The Task ID to retrieve scalars from.

        :return: Normalized scalar values.
        """
        objective = [o.get_normalized_objective(task_id) for o in self.objectives]
        if any(o is None for o in objective):
            return None
        return objective

    def get_objective_metric(self):
        # type: () -> List[(str, str)]
        """
        Return the metric title, series pairs of the objectives.

        :return: List[(title, series)]
        """
        return [o.get_objective_metric() for o in self.objectives]

    def get_top_tasks(self, top_k, optimizer_task_id=None, task_filter=None):
        # type: (int, Optional[str], Optional[dict]) -> Sequence[Task]
        """
        Return a list of Tasks of the top performing experiments.
        If there is only one objective, the tasks are sorted based on that objective.
        If there are multiple objectives, the tasks are sorted based on successive Pareto fronts.
        A trial is located at the Pareto front if there are no trials that dominate the trial.
        A trial dominates another trial if all its objective metrics are greater or equal than the other
        trial's and there is at least one objective metric that is strictly greater than the other.

        :param int top_k: The number of Tasks (experiments) to return.
        :param str optimizer_task_id: Parent optimizer Task ID
        :param dict task_filter: Optional task_filtering for the query

        :return: A list of Task objects, ordered by performance, where index 0 is the best performing Task.
        """
        if self.len == 1:
            return self.objectives[0].get_top_tasks(
                top_k=top_k,
                optimizer_task_id=optimizer_task_id,
                task_filter=task_filter
            )
        task_filter = deepcopy(task_filter) if task_filter else {}
        if optimizer_task_id:
            task_filter["parent"] = optimizer_task_id
        # noinspection PyProtectedMember
        tasks = Task._query_tasks(**task_filter)
        candidates = {}
        for task in tasks:
            values = self.get_objective(task.id)
            if values is None or any(v is None for v in values):
                continue
            candidates[task.id] = (values, task.id)
        sorted_ids = self._sort_jobs_by_domination(candidates)
        if not sorted_ids:
            return []
        return Task.get_tasks(task_ids=sorted_ids[:top_k])

    def _get_last_metrics_encode_field(self):
        # noinspection PyProtectedMember
        return [o._get_last_metrics_encode_field() for o in self.objectives]

    def _weakly_dominates_normalized(self, lhs, rhs):
        return all(lhs_elem * o.sign >= rhs_elem * o.sign for lhs_elem, rhs_elem, o in zip(lhs, rhs, self.objectives))

    @staticmethod
    def _dominates(lhs, rhs):
        return all(lhs_elem >= rhs_elem for lhs_elem, rhs_elem in zip(lhs, rhs)) and \
            any(lhs_elem > rhs_elem for lhs_elem, rhs_elem in zip(lhs, rhs))

    def _sort_jobs_by_domination(self, jobs):
        job_ids = list(jobs.keys())
        job_ids_sorted = []
        while len(job_ids_sorted) < len(jobs.keys()):
            have_result = False
            for job_id in job_ids:
                if all(self._weakly_dominates_normalized(jobs[job_id][0], jobs[other_job_id][0])
                       for other_job_id in job_ids):
                    have_result = True
                    job_ids_sorted.append(job_id)
            if not have_result:
                job_ids_sorted.extend(job_ids)
            job_ids = [job_id for job_id in job_ids if job_id not in job_ids_sorted]
        return job_ids_sorted


class GridSearch(SearchStrategy):
    """
    Grid search strategy controller. Full grid sampling of every hyperparameter combination.
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
        # type: () -> Optional[ClearmlJob]
        """
        Create a new job if needed. Return the newly created job. If no job needs to be created, return ``None``.

        :return: A newly created ClearmlJob object, or None if no ClearmlJob is created.
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
    Random search strategy controller. Random uniform sampling of hyperparameters.
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
        # type: () -> Optional[ClearmlJob]
        """
        Create a new job if needed. Return the newly created job. If no job needs to be created, return ``None``.

        :return: A newly created ClearmlJob object, or None if no ClearmlJob created
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
    Hyperparameter search controller. Clones the base experiment, changes arguments and tries to maximize/minimize
    the defined objective.
    """
    _tag = 'optimization'

    def __init__(
            self,
            base_task_id,  # type: str
            hyper_parameters,  # type: Sequence[Parameter]
            objective_metric_title,  # type: Union[str, Sequence[str]]
            objective_metric_series,  # type: Union[str, Sequence[str]]
            objective_metric_sign="min",  # type: Union[str, Sequence[str]]
            optimizer_class=RandomSearch,  # type: Union[SearchStrategy, type(SearchStrategy)]
            max_number_of_concurrent_tasks=10,  # type: int
            execution_queue="default",  # type: str
            optimization_time_limit=None,  # type: Optional[float]
            compute_time_limit=None,  # type: Optional[float]
            auto_connect_task=True,  # type: Union[bool, Task]
            always_create_task=False,  # type: bool
            spawn_project=None,  # type: Optional[str]
            save_top_k_tasks_only=None,  # type: Optional[int]
            **optimizer_kwargs  # type: Any
    ):
        # type: (...) -> ()
        """
        Create a new hyperparameter controller. The newly created object will launch and monitor the new experiments.

        :param str base_task_id: The Task ID to be used as template experiment to optimize.
        :param list hyper_parameters: The list of Parameter objects to optimize over.
        :param Union[str, Sequence[str]] objective_metric_title: The Objective metric title(s) to maximize / minimize
            (for example, ``validation``, ``["validation", "loss"]``). If ``objective_metric_title`` is a sequence
            (used to optimize multiple objectives at the same time), then ``objective_metric_series`` and
             ``objective_metric_sign`` have to be sequences of the same length. Each title will be matched
             with the respective series and sign
        :param Union[str, Sequence[str]] objective_metric_series: The Objective metric series to maximize / minimize
            (for example, ``loss_series``, ``["validation_series", "loss_series"]``).
        :param Union[str, Sequence[str]] objective_metric_sign: The objectives to maximize / minimize.
            The values are:

          - ``min`` - Minimize the last reported value for the specified title/series scalar.
          - ``max`` - Maximize the last reported value for the specified title/series scalar.
          - ``min_global`` - Minimize the min value of *all* reported values for the specific title/series scalar.
          - ``max_global`` - Maximize the max value of *all* reported values for the specific title/series scalar.

        :param class.SearchStrategy optimizer_class: The SearchStrategy optimizer to use for the hyperparameter search
        :param int max_number_of_concurrent_tasks: The maximum number of concurrent Tasks (experiments) running at the
            same time.
        :param str execution_queue: The execution queue to use for launching Tasks (experiments).
        :param float optimization_time_limit: The maximum time (minutes) for the entire optimization process. The
            default is ``None``, indicating no time limit.
        :param float compute_time_limit: The maximum compute time in minutes. When time limit is exceeded,
            all jobs aborted. (Optional)
        :param bool auto_connect_task: Store optimization arguments and configuration in the Task.
            The values are:

          - ``True`` - The optimization argument and configuration will be stored in the Task. All arguments will
              be under the hyperparameter section ``opt``, and the optimization hyper_parameters space will be
              stored in the Task configuration object section.
          - ``False`` - Do not store with Task.
          - ``Task`` - A specific Task object to connect the optimization process with.

        :param bool always_create_task: Always create a new Task.
            The values are:

          - ``True`` - No current Task initialized. Create a new task named ``optimization`` in the ``base_task_id``
              project.
          - ``False`` - Use the :py:meth:`task.Task.current_task` (if exists) to report statistics.

        :param str spawn_project: If project name is specified, create all optimization Jobs (Tasks) in the
            specified project instead of the original base_task_id project.

        :param int save_top_k_tasks_only: If specified and above 0, keep only the top_k performing Tasks,
            and archive the rest of the created Tasks. Default: -1 keep everything, nothing will be archived.

        :param ** optimizer_kwargs: Arguments passed directly to the optimizer constructor.

            Example:

            .. code-block:: py

                :linenos:
                :caption: Example

                from clearml import Task
                from clearml.automation import UniformParameterRange, DiscreteParameterRange
                from clearml.automation import GridSearch, RandomSearch, HyperParameterOptimizer

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
        if type(objective_metric_title) is not type(objective_metric_series) or type(
                objective_metric_title) is not type(objective_metric_sign):
            raise TypeError(
                "objective_metric_series, objective_metric_title and objective_metric_sign have to be of the same type"
                " (strings if doing single objective optimization and lists of the same length"
                " if doing multi-objective optimization)"
            )
        if isinstance(objective_metric_title, str):
            objective_metric_series = [objective_metric_series]
            objective_metric_title = [objective_metric_title]
            objective_metric_sign = [objective_metric_sign]
        if len(objective_metric_series) != len(objective_metric_title) or len(objective_metric_series) != len(
                objective_metric_sign
        ):
            raise ValueError(
                "Can not use multiple objective optimization when objective_metric_series, objective_metric_title"
                " or objective_metric_sign do not have the same length"
            )
        # create a new Task, if we do not have one already
        self._task = auto_connect_task if isinstance(auto_connect_task, Task) else Task.current_task()
        self._readonly_task = \
            isinstance(auto_connect_task, Task) and str(self._task.status) not in ('created', 'in_progress')
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
        if self._task and not self._readonly_task:
            self._task.add_tags([self._tag])
            if auto_connect_task:
                optimizer_class, hyper_parameters, opts = self._connect_args(
                    optimizer_class=optimizer_class, hyper_param_configuration=hyper_parameters, **opts)

        self.base_task_id = opts['base_task_id']
        self.hyper_parameters = hyper_parameters
        self.max_number_of_concurrent_tasks = opts['max_number_of_concurrent_tasks']
        self.execution_queue = opts['execution_queue']
        self._objective_metric = MultiObjective(
            title=opts["objective_metric_title"],
            series=opts["objective_metric_series"],
            order=["min" if sign_ in ("min", "min_global") else "max" for sign_ in opts["objective_metric_sign"]],
            extremum=[sign_.endswith("_global") for sign_ in opts["objective_metric_sign"]]
        )
        optuna_error_message = "Multi parameter optimization is only supported via Optuna. Please install Optuna via" + \
                               " `pip install optuna and set the `optimizer_class` to `clearml.automation.optuna.OptimizerOptuna`"
        try:
            if self._objective_metric.len != 1:
                from .optuna import OptimizerOptuna

                if optimizer_class != OptimizerOptuna:
                    raise ValueError(optuna_error_message)
        except Exception:
            raise ValueError(optuna_error_message)
        # if optimizer_class is an instance, use it as is.
        if not isinstance(optimizer_class, type):
            self.optimizer = optimizer_class
        else:
            self.optimizer = optimizer_class(
                base_task_id=opts['base_task_id'], hyper_parameters=hyper_parameters,
                objective_metric=self._objective_metric, execution_queue=opts['execution_queue'],
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
        self._save_top_k_tasks_only = max(0, save_top_k_tasks_only or 0)
        self.optimizer.set_job_default_parent(
            self._task.id if self._task else None, project_name=spawn_project or None)
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

    def start_locally(self, job_complete_callback=None):
        # type: (Optional[Callable[[str, float, int, dict, str], None]]) -> bool
        """
        Start the HyperParameterOptimizer controller completely locally. Both the optimizer task
        and all spawned substasks are run on the local machine using the current environment.
        If the calling process is stopped, then the controller stops as well.

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
        self.optimizer.set_job_class(LocalClearmlJob)
        return self.start(job_complete_callback=job_complete_callback)

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
            The default is ``None``, indicating do not wait to terminate immediately.
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

    def get_top_experiments_details(
            self,
            top_k,
            all_metrics=False,
            all_hyper_parameters=False,
            only_completed=False
    ):
        # type: (int, bool, bool, bool) -> Sequence[(str, dict)]
        """
        Return a list of dictionaries of the top performing experiments.
        Example: ``[{'task_id': Task-ID, 'metrics': scalar-metric-dict, 'hyper_parameters': Hyper-Parameters},]``

        Order is based on the controller ``Objective`` object.

        :param int top_k: The number of Tasks (experiments) to return.
        :param all_metrics: Default False, only return the objective metric on the metrics dictionary.
            If True, return all scalar metrics of the experiment
        :param all_hyper_parameters: Default False. If True, return all the hyperparameters from all the sections.
        :param only_completed: return only completed Tasks. Default False.

        :return: A list of dictionaries ({task_id: '', hyper_parameters: {}, metrics: {}}), ordered by performance,
            where index 0 is the best performing Task.
            Example w/ all_metrics=False:

            .. code-block:: py

                [
                    {
                        task_id: '0593b76dc7234c65a13a301f731958fa',
                        hyper_parameters: {'General/lr': '0.03', 'General/batch_size': '32'},
                        metrics: {
                            'accuracy per class/cat': {
                                'metric': 'accuracy per class',
                                'variant': 'cat',
                                'value': 0.119,
                                'min_value': 0.119,
                                'max_value': 0.782
                            },
                        }
                    },
                ]

            Example w/ all_metrics=True:

            .. code-block:: py

                [
                    {
                        task_id: '0593b76dc7234c65a13a301f731958fa',
                        hyper_parameters: {'General/lr': '0.03', 'General/batch_size': '32'},
                        metrics: {
                            'accuracy per class/cat': {
                                'metric': 'accuracy per class',
                                'variant': 'cat',
                                'value': 0.119,
                                'min_value': 0.119,
                                'max_value': 0.782
                            },
                            'accuracy per class/deer': {
                                'metric': 'accuracy per class',
                                'variant': 'deer',
                                'value': 0.219,
                                'min_value': 0.219,
                                'max_value': 0.282
                            },
                        }
                    },
                ]
        """
        if not self.optimizer:
            return []
        return self.optimizer.get_top_experiments_details(
            top_k=top_k,
            all_metrics=all_metrics,
            all_hyper_parameters=all_hyper_parameters,
            only_completed=only_completed)

    def get_optimizer(self):
        # type: () -> SearchStrategy
        """
        Return the currently used optimizer object.

        :return: The SearchStrategy object used.
        """
        return self.optimizer

    def set_default_job_class(self, job_class):
        # type: (ClearmlJob) -> ()
        """
        Set the Job class to use when the optimizer spawns new Jobs.

        :param ClearmlJob job_class: The Job Class type.
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

    @classmethod
    def get_optimizer_top_experiments(
            cls,
            objective_metric_title,  # type: Union[str, List[str]]
            objective_metric_series,  # type: Union[str, List[str]]
            objective_metric_sign,  # type: Union[str, List[str]]
            optimizer_task_id,  # type: str
            top_k,  # type: int
    ):
        # type: (...) -> Sequence[Task]
        """
        Return a list of Tasks of the top performing experiments
        for a specific HyperParameter Optimization session (i.e. Task ID), based on the title/series objective.

        :param str objective_metric_title: The Objective metric title to maximize / minimize (for example,
            ``validation``).
        :param str objective_metric_series: The Objective metric series to maximize / minimize (for example, ``loss``).
        :param str objective_metric_sign: The objective to maximize / minimize.
            The values are:

          - ``min`` - Minimize the last reported value for the specified title/series scalar.
          - ``max`` - Maximize the last reported value for the specified title/series scalar.
          - ``min_global`` - Minimize the min value of *all* reported values for the specific title/series scalar.
          - ``max_global`` - Maximize the max value of *all* reported values for the specific title/series scalar.
        :param str optimizer_task_id: Parent optimizer Task ID
        :param top_k: The number of Tasks (experiments) to return.
        :return: A list of Task objects, ordered by performance, where index 0 is the best performing Task.
        """
        objective = Objective(
            title=objective_metric_title, series=objective_metric_series, order=objective_metric_sign)
        return objective.get_top_tasks(top_k=top_k, optimizer_task_id=optimizer_task_id)

    @property
    def objective_metric(self):
        if self._objective_metric.len == 1:
            return self._objective_metric.objectives[0]
        return self._objective_metric

    def _connect_args(self, optimizer_class=None, hyper_param_configuration=None, **kwargs):
        # type: (SearchStrategy, dict, Any) -> (SearchStrategy, list, dict)
        if not self._task or self._readonly_task:
            logger.warning('Auto Connect turned on but no Task was found, '
                           'hyper-parameter optimization argument logging disabled')
            return optimizer_class, hyper_param_configuration, kwargs

        configuration_dict = {'parameter_optimization_space': [c.to_dict() for c in hyper_param_configuration]}
        self._task.connect_configuration(configuration_dict)
        # this is the conversion back magic:
        configuration_dict = {'parameter_optimization_space': [
            Parameter.from_dict(c) for c in configuration_dict['parameter_optimization_space']]}

        complex_optimizer_kwargs = None
        if 'optimizer_kwargs' in kwargs:
            # do not store complex optimizer kwargs:
            optimizer_kwargs = kwargs.pop('optimizer_kwargs', {})
            complex_optimizer_kwargs = {
                k: v for k, v in optimizer_kwargs.items()
                if not isinstance(v, six.string_types + six.integer_types +
                                  (six.text_type, float, list, tuple, dict, type(None)))}
            kwargs['optimizer_kwargs'] = {
                k: v for k, v in optimizer_kwargs.items() if k not in complex_optimizer_kwargs}

        # skip non basic types:
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

        if complex_optimizer_kwargs:
            if 'optimizer_kwargs' not in arguments['opt']:
                arguments['opt']['optimizer_kwargs'] = complex_optimizer_kwargs
            else:
                arguments['opt']['optimizer_kwargs'].update(complex_optimizer_kwargs)

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
        title_series = self._objective_metric.get_objective_metric()
        title = ["{}/{}".format(ts[0], ts[1]) for ts in title_series]
        counter = 0
        completed_jobs = dict()
        task_logger = None
        cur_completed_jobs = set()
        cur_task = self._task or Task.current_task()
        if cur_task and self.optimizer:
            # noinspection PyProtectedMember
            child_tasks = self.optimizer._get_child_tasks(
                parent_task_id=cur_task.id, status=['completed', 'stopped'])
            hyper_parameters = [h.name for h in self.hyper_parameters]
            for task in child_tasks:
                params = {k: v for k, v in task.get_parameters().items() if k in hyper_parameters}
                params["status"] = str(task.status)
                # noinspection PyProtectedMember
                iteration_value = task.get_last_iteration()
                objective = self._objective_metric.get_objective(task)
                completed_jobs[task.id] = (
                    objective if objective is not None else (
                            [-1] * self._objective_metric.len
                    ),
                    [iteration_value] * self._objective_metric.len if iteration_value is not None else (
                            [-1] * self._objective_metric.len
                    ),
                    params
                )

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
            cur_task = self._task or Task.current_task()
            if cur_task and not self._readonly_task:
                task_logger = cur_task.get_logger()

                # do some reporting

                self._report_remaining_budget(task_logger, counter)

                if (
                        self.optimizer.budget.compute_time.used
                        and self.optimizer.budget.compute_time.limit
                        and self.optimizer.budget.compute_time.used >= self.optimizer.budget.compute_time.limit
                ):
                    logger.warning(
                        "Optimizer task reached compute time limit (used {:.2f} out of {:.2f})".format(
                            self.optimizer.budget.compute_time.limit, self.optimizer.compute_time.used
                        )
                    )
                    timeout = -1

                self._report_resources(task_logger, counter)
                # collect a summary of all the jobs and their final objective values
                cur_completed_jobs = set(self.optimizer.get_created_jobs_ids().keys()) - {
                    j.task_id() for j in self.optimizer.get_running_jobs()
                }
                self._report_completed_status(completed_jobs, cur_completed_jobs, task_logger, title)
                self._report_completed_tasks_best_results(set(completed_jobs.keys()), task_logger, title, counter)

            self._auto_archive_low_performance_tasks(completed_jobs)

            # if we should leave, stop everything now.
            if timeout < 0:
                # we should leave
                self.stop(wait_for_reporter=False)
                return
        if task_logger and counter and not self._readonly_task:
            counter += 1
            self._report_remaining_budget(task_logger, counter)
            self._report_resources(task_logger, counter)
            self._report_completed_status(completed_jobs, cur_completed_jobs, task_logger, title, force=True)
            self._report_completed_tasks_best_results(set(completed_jobs.keys()), task_logger, title, counter)

        self._auto_archive_low_performance_tasks(completed_jobs)

    def _report_completed_status(self, completed_jobs, cur_completed_jobs, task_logger, title, force=False):
        job_ids_sorted_by_objective = self.__sort_jobs_by_objective(completed_jobs)
        best_experiment = (
            (
                self._objective_metric.get_normalized_objective(job_ids_sorted_by_objective[0]),
                job_ids_sorted_by_objective[0],
            )
            if job_ids_sorted_by_objective
            else ([float("-inf")], None)
        )
        if force or cur_completed_jobs != set(completed_jobs.keys()):
            pairs = []
            labels = []
            created_jobs = copy(self.optimizer.get_created_jobs_ids())
            created_jobs_tasks = self.optimizer.get_created_jobs_tasks()
            id_status = {j_id: j_run.status() for j_id, j_run in created_jobs_tasks.items()}
            for i, (job_id, params) in enumerate(created_jobs.items()):
                value = self._objective_metric.get_objective(job_id)
                if job_id in completed_jobs:
                    if value != completed_jobs[job_id][0]:
                        iteration_value = self._objective_metric.get_current_raw_objective(job_id)
                        iteration = [it_[0] if it_ else -1 for it_ in iteration_value]
                        completed_jobs[job_id] = (
                            value,
                            iteration,
                            copy(dict(status=id_status.get(job_id), **params)))
                    elif completed_jobs.get(job_id):
                        completed_jobs[job_id] = (completed_jobs[job_id][0],
                                                  completed_jobs[job_id][1],
                                                  copy(dict(status=id_status.get(job_id), **params)))
                    pairs.append((i, completed_jobs[job_id][0]))
                    labels.append(str(completed_jobs[job_id][2])[1:-1])
                elif value is not None and all(v is not None for v in value):
                    pairs.append((i, value))
                    labels.append(str(params)[1:-1])
                    iteration_value = self._objective_metric.get_current_raw_objective(job_id)
                    iteration = [it_[0] if it_ else -1 for it_ in iteration_value]
                    completed_jobs[job_id] = (
                        value,
                        iteration,
                        copy(dict(status=id_status.get(job_id), **params)))
                    # callback new experiment completed
                    if self._experiment_completed_cb:
                        normalized_value = self._objective_metric.get_normalized_objective(job_id)
                        if self._objective_metric.len == 1 and normalized_value is not None and \
                                normalized_value[0] > best_experiment[0][0]:
                            best_experiment = normalized_value, job_id
                        elif self._objective_metric.len != 1 and normalized_value is not None and \
                                all(n is not None for n in normalized_value) and (best_experiment[0] == float("-inf") or
                                                                                  MultiObjective._dominates(
                                                                                      normalized_value,
                                                                                      best_experiment[0])):  # noqa
                            best_experiment = normalized_value, job_id
                        c = completed_jobs[job_id]
                        self._experiment_completed_cb(job_id, c[0], c[1], c[2], best_experiment[1])

            if pairs:
                print("Updating job performance summary plot/table")
                if isinstance(title, list):
                    for i, title_ in enumerate(title):
                        # update scatter plot
                        task_logger.report_scatter2d(
                            title="Optimization Objective",
                            series=title_,
                            scatter=[(p[0], p[1][i]) for p in pairs],
                            iteration=0,
                            labels=labels,
                            mode="markers",
                            xaxis="job #",
                            yaxis="objective",
                        )
                else:
                    task_logger.report_scatter2d(
                        title="Optimization Objective",
                        series=title,
                        scatter=pairs,
                        iteration=0,
                        labels=labels,
                        mode="markers",
                        xaxis="job #",
                        yaxis="objective",
                    )

            # update summary table
            job_ids_sorted_by_objective = self.__sort_jobs_by_objective(completed_jobs)
            # sort the columns except for 'objective', 'iteration'
            columns = list(sorted(set([c for k, v in completed_jobs.items() for c in v[2].keys()])))

            concat_iterations = True
            if self._objective_metric.len == 1:
                # add the index column (task id) and the first two columns 'objective', 'iteration' then the rest
                table_values = [['task id', 'objective', 'iteration'] + columns]
                table_values += \
                    [([job, completed_jobs[job][0][0], completed_jobs[job][1][0]] +
                      [completed_jobs[job][2].get(c, '') for c in columns]) for job in job_ids_sorted_by_objective]
            else:
                table_values = ['task id']
                for job in job_ids_sorted_by_objective:
                    if not all(iter_ == completed_jobs[job][1][0] for iter_ in completed_jobs[job][1]):
                        concat_iterations = False
                        break
                if concat_iterations:
                    for objective in self._objective_metric.objectives:
                        table_values.append(objective.title + "/" + objective.series)
                    table_values.append("iteration")
                    table_values = [table_values + columns]
                    for job in job_ids_sorted_by_objective:
                        entry = [job]
                        for val in completed_jobs[job][0]:
                            entry += [val]
                        entry += [completed_jobs[job][1][0]]
                        entry += [completed_jobs[job][2].get(c, '') for c in columns]
                        table_values.append(entry)
                else:
                    for objective in self._objective_metric.objectives:
                        table_values.append(objective.title + "/" + objective.series)
                        table_values.append("iteration " + objective.title + "/" + objective.series)
                    table_values = [table_values + columns]
                    for job in job_ids_sorted_by_objective:
                        entry = [job]
                        for val, iter_ in zip(completed_jobs[job][0], completed_jobs[job][1]):
                            entry += [val, iter_]
                        entry += [completed_jobs[job][2].get(c, '') for c in columns]
                        table_values.append(entry)

            # create links for task id in the table
            task_link_template = self._task.get_output_log_web_page() \
                .replace('/{}/'.format(self._task.project), '/{project}/') \
                .replace('/{}/'.format(self._task.id), '/{task}/')
            # create links for task id in the table
            table_values_with_links = deepcopy(table_values)
            for i in range(1, len(table_values_with_links)):
                task_id = table_values_with_links[i][0]
                project_id = created_jobs_tasks[task_id].task.project \
                    if task_id in created_jobs_tasks else '*'
                table_values_with_links[i][0] = '<a href="{}"> {} </a>'.format(
                    task_link_template.format(project=project_id, task=task_id), task_id)

            task_logger.report_table(
                "summary",
                "job",
                0,
                table_plot=table_values_with_links,
                extra_layout={
                    "title": "objective: {}".format(title if not isinstance(title, list) else ", ".join(title))
                },
            )

            # Build parallel Coordinates: convert to columns, and reorder accordingly
            if len(table_values) > 1:
                table_values_columns = [[row[i] for row in table_values] for i in range(len(table_values[0]))]
                if self._objective_metric.len == 1:
                    table_values_columns = \
                        [[table_values_columns[0][0]] + [c[:6] + '...' for c in table_values_columns[0][1:]]] + \
                        table_values_columns[2:-1] + [[title] + table_values_columns[1][1:]]
                else:
                    if not concat_iterations:
                        new_table_values_columns = []
                        handled = []
                        for i in range(1, 2 * len(self._objective_metric.objectives), 2):
                            handled.append(i)
                            new_table_values_columns.append(table_values_columns[i])
                        prefix = []
                        for i in range(len(table_values_columns)):
                            if i in handled or table_values_columns[i][0] == "status":
                                continue
                            prefix.append(table_values_columns[i])
                        table_values_columns = prefix + new_table_values_columns
                    else:
                        table_values_columns = ([table_values_columns[0]] +
                                                table_values_columns[len(self._objective_metric.objectives) + 1:-1] +
                                                table_values_columns[1:len(self._objective_metric.objectives) + 1]
                                                )
                    for i in range(len(table_values_columns[0]) - 1):
                        table_values_columns[0][i + 1] = table_values_columns[0][i + 1][:6] + "..."
                pcc_dims = []
                for col in table_values_columns:
                    # test if all values are numbers:
                    try:
                        # try to cast all values to float
                        values = [float(v) for v in col[1:]]
                        d = dict(label=col[0], values=values)
                    except (ValueError, TypeError):
                        values = list(range(len(col[1:])))
                        ticks = col[1:]
                        unique_ticks = list(set(ticks))
                        d = dict(label=col[0], values=values, tickvals=values, ticktext=ticks)
                        if len(ticks) != len(unique_ticks):  # Mapping duplicate ticktext
                            ticktext = {key: i for i, key in enumerate(unique_ticks)}
                            d["values"] = [ticktext[tick] for tick in ticks]
                            d["tickvals"] = list(range(len(ticktext)))
                            d["ticktext"] = list(sorted(ticktext, key=ticktext.get))
                    pcc_dims.append(d)
                # report parallel coordinates
                plotly_pcc = dict(
                    data=[
                        dict(
                            type="parcoords",
                            line=dict(
                                colorscale="Viridis",
                                reversescale=(
                                    self._objective_metric.len == 1 and self._objective_metric.objectives[0].sign >= 0,
                                ),
                                color=table_values_columns[-1][1:],
                            ),
                            dimensions=pcc_dims,
                        )
                    ],
                    layout={},
                )
                task_logger.report_plotly(title="Parallel Coordinates", series="", iteration=0, figure=plotly_pcc)

            # upload summary as artifact
            if force:
                task = self._task or Task.current_task()
                if task:
                    task.upload_artifact(name='summary', artifact_object={'table': table_values})

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
        if not completed_jobs:
            return

        objectives = self._objective_metric.objectives
        if not isinstance(title, list):
            title = [title]
        for objective, title_ in zip(objectives, title):
            value_func, series_name = (max, "max") if objective.get_objective_sign() > 0 else (min, "min")
            latest_completed, obj_values = self._get_latest_completed_task_value(
                completed_jobs, series_name, objective.title, objective.series
            )
            if latest_completed:
                val = value_func(obj_values)
                task_logger.report_scalar(title=title_, series=series_name, iteration=counter, value=val)
                task_logger.report_scalar(
                    title=title_, series="last reported", iteration=counter, value=latest_completed
                )

    def _report_resources(self, task_logger, iteration):
        # type: (Logger, int) -> ()
        self._report_active_workers(task_logger, iteration)
        self._report_tasks_status(task_logger, iteration)

    def _report_active_workers(self, task_logger, iteration):
        # type: (Logger, int) -> ()
        res = self.__get_session().send(workers_service.GetAllRequest())
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

    def _get_latest_completed_task_value(self, cur_completed_jobs, series_name, title, series):
        # type: (Set[str], str) -> (float, List[float])
        completed_value = None
        latest_completed = None
        obj_values = []
        cur_task = self._task or Task.current_task()
        for j in cur_completed_jobs:
            res = cur_task.send(tasks_service.GetByIdRequest(task=j))
            response = res.wait()
            if not response.ok() or response.response_data["task"].get("status") != Task.TaskStatusEnum.completed:
                continue
            completed_time = datetime_from_isoformat(response.response_data["task"]["completed"].partition("+")[0])
            completed_time = completed_time.timestamp()
            completed_values = self._get_last_value(response, title, series)
            obj_values.append(completed_values['max_value'] if series_name == "max" else completed_values['min_value'])
            if not latest_completed or completed_time > latest_completed:
                latest_completed = completed_time
                completed_value = completed_values['value']
        return completed_value, obj_values

    def _get_last_value(self, response, title, series):
        metrics, title, series, values = ClearmlJob.get_metric_req_params(title, series)
        last_values = response.response_data["task"]['last_metrics'][title][series]
        return last_values

    def _auto_archive_low_performance_tasks(self, completed_jobs):
        if self._save_top_k_tasks_only <= 0:
            return

        # sort based on performance
        job_ids_sorted_by_objective = self.__sort_jobs_by_objective(completed_jobs)

        # query system_tags only
        res = self.__get_session().send(tasks_service.GetAllRequest(
            id=job_ids_sorted_by_objective, status=['completed', 'stopped'], only_fields=['id', 'system_tags']))
        response = res.wait()
        if not response.ok():
            return

        tasks_system_tags_lookup = {
            task.get("id"): task.get("system_tags") for task in response.response_data.get("tasks")}
        for i, task_id in enumerate(job_ids_sorted_by_objective):
            system_tags = tasks_system_tags_lookup.get(task_id, [])
            if i < self._save_top_k_tasks_only and Task.archived_tag in system_tags:
                print('Restoring from archive Task id={} (#{} objective={})'.format(
                    task_id, i, completed_jobs[task_id][0]))
                # top_k task and is archived, remove archive tag
                system_tags = list(set(system_tags) - {Task.archived_tag})
                res = self.__get_session().send(
                    tasks_service.EditRequest(task=task_id, system_tags=system_tags, force=True))
                res.wait()
            elif i >= self._save_top_k_tasks_only and Task.archived_tag not in system_tags:
                print('Archiving Task id={} (#{} objective={})'.format(
                    task_id, i, completed_jobs[task_id][0]))
                # Not in top_k task and not archived, add archive tag
                system_tags = list(set(system_tags) | {Task.archived_tag})
                res = self.__get_session().send(
                    tasks_service.EditRequest(task=task_id, system_tags=system_tags, force=True))
                res.wait()

    def __get_session(self):
        cur_task = self._task or Task.current_task()
        if cur_task:
            return cur_task.default_session
        # noinspection PyProtectedMember
        return Task._get_default_session()

    def __sort_jobs_by_objective(self, completed_jobs):
        if not completed_jobs:
            return []
        if self._objective_metric.len != 1:
            # noinspection PyProtectedMember
            return self._objective_metric._sort_jobs_by_domination(completed_jobs)
        else:
            return list(
                sorted(
                    completed_jobs.keys(),
                    key=lambda x: completed_jobs[x][0],
                    reverse=bool(self._objective_metric.objectives[0].sign >= 0),
                )
            )
