from time import sleep, time
from typing import Any, Optional, Sequence

from ..optimization import Objective, SearchStrategy
from ..parameters import (
    DiscreteParameterRange, UniformParameterRange, RandomSeed, UniformIntegerParameterRange, Parameter, )
from ...task import Task

try:
    # noinspection PyPackageRequirements
    from hpbandster.core.worker import Worker
    # noinspection PyPackageRequirements
    from hpbandster.optimizers import BOHB
    # noinspection PyPackageRequirements
    import hpbandster.core.nameserver as hpns
    # noinspection PyPackageRequirements, PyPep8Naming
    import ConfigSpace as CS
    # noinspection PyPackageRequirements, PyPep8Naming
    import ConfigSpace.hyperparameters as CSH

    Task.add_requirements('hpbandster')
except ImportError:
    raise ImportError("OptimizerBOHB requires 'hpbandster' package, it was not found\n"
                      "install with: pip install hpbandster")


class _TrainsBandsterWorker(Worker):
    def __init__(
            self,
            *args,  # type: Any
            optimizer,  # type: OptimizerBOHB
            base_task_id,  # type: str
            queue_name,  # type: str
            objective,  # type: Objective
            sleep_interval=0,  # type: float
            budget_iteration_scale=1.,  # type: float
            **kwargs  # type: Any
    ):
        # type: (...) -> _TrainsBandsterWorker
        super(_TrainsBandsterWorker, self).__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.base_task_id = base_task_id
        self.queue_name = queue_name
        self.objective = objective
        self.sleep_interval = sleep_interval
        self.budget_iteration_scale = budget_iteration_scale
        self._current_job = None

    def compute(self, config, budget, **kwargs):
        # type: (dict, float, Any) -> dict
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)
        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train.
                We assume budget is iteration, as time might not be stable from machine to machine.
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        self._current_job = self.optimizer.helper_create_job(self.base_task_id, parameter_override=config)
        # noinspection PyProtectedMember
        self.optimizer._current_jobs.append(self._current_job)
        if not self._current_job.launch(self.queue_name):
            return dict()
        iteration_value = None
        is_pending = True

        while not self._current_job.is_stopped():
            if is_pending and not self._current_job.is_pending():
                is_pending = False
                # noinspection PyProtectedMember
                self.optimizer.budget.jobs.update(
                    self._current_job.task_id(),
                    float(self.optimizer._min_iteration_per_job)/self.optimizer._max_iteration_per_job)

            # noinspection PyProtectedMember
            iteration_value = self.optimizer._objective_metric.get_current_raw_objective(self._current_job)
            if iteration_value:
                # update budget
                self.optimizer.budget.iterations.update(self._current_job.task_id(), iteration_value[0])

                # check if we exceeded this job budget
                if iteration_value[0] >= self.budget_iteration_scale * budget:
                    self._current_job.abort()
                    break

            sleep(self.sleep_interval)

        if iteration_value:
            # noinspection PyProtectedMember
            self.optimizer.budget.jobs.update(
                self._current_job.task_id(),
                float(iteration_value[0]) / self.optimizer._max_iteration_per_job)

        result = {
            # this is the a mandatory field to run hyperband
            # remember: HpBandSter always minimizes!
            'loss': float(self.objective.get_normalized_objective(self._current_job) * -1.),
            # can be used for any user-defined information - also mandatory
            'info': self._current_job.task_id()
        }
        print('TrainsBandsterWorker result {}, iteration {}'.format(result, iteration_value))
        # noinspection PyProtectedMember
        self.optimizer._current_jobs.remove(self._current_job)
        return result


class OptimizerBOHB(SearchStrategy, RandomSeed):
    def __init__(
            self,
            base_task_id,  # type: str
            hyper_parameters,  # type: Sequence[Parameter]
            objective_metric,  # type: Objective
            execution_queue,  # type: str
            num_concurrent_workers,  # type: int
            min_iteration_per_job,  # type: Optional[int]
            max_iteration_per_job,  # type: Optional[int]
            total_max_jobs,  # type: Optional[int]
            pool_period_min=2.,  # type: float
            time_limit_per_job=None,  # type: Optional[float]
            compute_time_limit=None,  # type: Optional[float]
            local_port=9090,  # type: int
            **bohb_kwargs  # type: Any
    ):
        # type: (...) -> None
        """
        Initialize a BOHB search strategy optimizer
        BOHB performs robust and efficient hyperparameter optimization at scale by combining
        the speed of Hyperband searches with the guidance and guarantees of convergence of Bayesian
        Optimization. Instead of sampling new configurations at random,
        BOHB uses kernel density estimators to select promising candidates.

        .. note::

            For reference:
            @InProceedings{falkner-icml-18,
                title =        {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},
                author =       {Falkner, Stefan and Klein, Aaron and Hutter, Frank},
                booktitle =    {Proceedings of the 35th International Conference on Machine Learning},
                pages =        {1436--1445},
                year =         {2018},
            }

        :param str base_task_id: Task ID (str)
        :param list hyper_parameters: list of Parameter objects to optimize over
        :param Objective objective_metric: Objective metric to maximize / minimize
        :param str execution_queue: execution queue to use for launching Tasks (experiments).
        :param int num_concurrent_workers: Limit number of concurrent running Tasks (machines)
        :param int min_iteration_per_job: minimum number of iterations for a job to run.
            'iterations' are the reported iterations for the specified objective,
            not the maximum reported iteration of the Task.
        :param int max_iteration_per_job: number of iteration per job
            'iterations' are the reported iterations for the specified objective,
            not the maximum reported iteration of the Task.
        :param int total_max_jobs: total maximum job for the optimization process.
            Must be provided in order to calculate the total budget for the optimization process.
            The total budget is measured by "iterations" (see above)
            and will be set to `max_iteration_per_job * total_max_jobs`
            This means more than total_max_jobs could be created, as long as the cumulative iterations
            (summed over all created jobs) will not exceed `max_iteration_per_job * total_max_jobs`
        :param float pool_period_min: time in minutes between two consecutive pools
        :param float time_limit_per_job: Optional, maximum execution time per single job in minutes,
            when time limit is exceeded job is aborted
        :param float compute_time_limit: The maximum compute time in minutes. When time limit is exceeded,
            all jobs aborted. (Optional)
        :param int local_port: default port 9090 tcp, this is a must for the BOHB workers to communicate, even locally.
        :param bohb_kwargs: arguments passed directly to the BOHB object
        """
        if not max_iteration_per_job or not min_iteration_per_job or not total_max_jobs:
            raise ValueError(
                "OptimizerBOHB is missing a defined budget.\n"
                "The following arguments must be defined: "
                "max_iteration_per_job, min_iteration_per_job, total_max_jobs.\n"
                "Maximum optimization budget is: max_iteration_per_job * total_max_jobs\n"
            )

        super(OptimizerBOHB, self).__init__(
            base_task_id=base_task_id, hyper_parameters=hyper_parameters, objective_metric=objective_metric,
            execution_queue=execution_queue, num_concurrent_workers=num_concurrent_workers,
            pool_period_min=pool_period_min, time_limit_per_job=time_limit_per_job,
            compute_time_limit=compute_time_limit, max_iteration_per_job=max_iteration_per_job,
            min_iteration_per_job=min_iteration_per_job, total_max_jobs=total_max_jobs)
        self._max_iteration_per_job = max_iteration_per_job
        self._min_iteration_per_job = min_iteration_per_job
        verified_bohb_kwargs = ['eta', 'min_budget', 'max_budget', 'min_points_in_model', 'top_n_percent',
                                'num_samples', 'random_fraction', 'bandwidth_factor', 'min_bandwidth']
        self._bohb_kwargs = dict((k, v) for k, v in bohb_kwargs.items() if k in verified_bohb_kwargs)
        self._param_iterator = None
        self._namespace = None
        self._bohb = None
        self._res = None
        self._nameserver_port = local_port

    def set_optimization_args(
            self,
            eta=3,  # type: float
            min_budget=None,  # type: Optional[float]
            max_budget=None,  # type: Optional[float]
            min_points_in_model=None,  # type: Optional[int]
            top_n_percent=15,  # type: Optional[int]
            num_samples=None,  # type: Optional[int]
            random_fraction=1 / 3.,  # type: Optional[float]
            bandwidth_factor=3,  # type: Optional[float]
            min_bandwidth=1e-3,  # type: Optional[float]
    ):
        # type: (...) -> ()
        """
        Defaults copied from BOHB constructor, see details in BOHB.__init__

        BOHB performs robust and efficient hyperparameter optimization
        at scale by combining the speed of Hyperband searches with the
        guidance and guarantees of convergence of Bayesian
        Optimization. Instead of sampling new configurations at random,
        BOHB uses kernel density estimators to select promising candidates.

        .. note::

            For reference:
            @InProceedings{falkner-icml-18,
              title =        {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},
              author =       {Falkner, Stefan and Klein, Aaron and Hutter, Frank},
              booktitle =    {Proceedings of the 35th International Conference on Machine Learning},
              pages =        {1436--1445},
              year =         {2018},
            }

        :param eta : float (3)
            In each iteration, a complete run of sequential halving is executed. In it,
            after evaluating each configuration on the same subset size, only a fraction of
            1/eta of them 'advances' to the next round.
            Must be greater or equal to 2.
        :param min_budget : float (0.01)
            The smallest budget to consider. Needs to be positive!
        :param max_budget : float (1)
            The largest budget to consider. Needs to be larger than min_budget!
            The budgets will be geometrically distributed
            :math:`a^2 + b^2 = c^2 /sim /eta^k` for :math:`k/in [0, 1, ... , num/_subsets - 1]`.
        :param min_points_in_model: int (None)
            number of observations to start building a KDE. Default 'None' means
            dim+1, the bare minimum.
        :param top_n_percent: int (15)
            percentage ( between 1 and 99, default 15) of the observations that are considered good.
        :param num_samples: int (64)
            number of samples to optimize EI (default 64)
        :param random_fraction: float (1/3.)
            fraction of purely random configurations that are sampled from the
            prior without the model.
        :param bandwidth_factor: float (3.)
            to encourage diversity, the points proposed to optimize EI, are sampled
            from a 'widened' KDE where the bandwidth is multiplied by this factor (default: 3)
        :param min_bandwidth: float (1e-3)
            to keep diversity, even when all (good) samples have the same value for one of the parameters,
            a minimum bandwidth (Default: 1e-3) is used instead of zero.

        """
        if min_budget:
            self._bohb_kwargs['min_budget'] = min_budget
        if max_budget:
            self._bohb_kwargs['max_budget'] = max_budget
        if num_samples:
            self._bohb_kwargs['num_samples'] = num_samples
        self._bohb_kwargs['eta'] = eta
        self._bohb_kwargs['min_points_in_model'] = min_points_in_model
        self._bohb_kwargs['top_n_percent'] = top_n_percent
        self._bohb_kwargs['random_fraction'] = random_fraction
        self._bohb_kwargs['bandwidth_factor'] = bandwidth_factor
        self._bohb_kwargs['min_bandwidth'] = min_bandwidth

    def start(self):
        # type: () -> ()
        """
        Start the Optimizer controller function loop()
        If the calling process is stopped, the controller will stop as well.

        .. important::
            This function returns only after optimization is completed or :meth:`stop` was called.

        """
        # Step 1: Start a NameServer
        fake_run_id = 'OptimizerBOHB_{}'.format(time())
        # default port is 9090, we must have one, this is how BOHB workers communicate (even locally)
        self._namespace = hpns.NameServer(run_id=fake_run_id, host='127.0.0.1', port=self._nameserver_port)
        self._namespace.start()

        # we have to scale the budget to the iterations per job, otherwise numbers might be too high
        budget_iteration_scale = self._max_iteration_per_job

        # Step 2: Start the workers
        workers = []
        for i in range(self._num_concurrent_workers):
            w = _TrainsBandsterWorker(
                optimizer=self,
                sleep_interval=int(self.pool_period_minutes * 60),
                budget_iteration_scale=budget_iteration_scale,
                base_task_id=self._base_task_id,
                objective=self._objective_metric,
                queue_name=self._execution_queue,
                nameserver='127.0.0.1', nameserver_port=self._nameserver_port, run_id=fake_run_id, id=i)
            w.run(background=True)
            workers.append(w)

        # Step 3: Run an optimizer
        self._bohb = BOHB(configspace=self._convert_hyper_parameters_to_cs(),
                          run_id=fake_run_id,
                          # num_samples=self.total_max_jobs, # will be set by self._bohb_kwargs
                          min_budget=float(self._min_iteration_per_job) / float(self._max_iteration_per_job),
                          **self._bohb_kwargs)
        # scale the budget according to the successive halving iterations
        if self.budget.jobs.limit:
            self.budget.jobs.limit *= len(self._bohb.budgets)
        if self.budget.iterations.limit:
            self.budget.iterations.limit *= len(self._bohb.budgets)
        # start optimization
        self._res = self._bohb.run(n_iterations=self.total_max_jobs, min_n_workers=self._num_concurrent_workers)

        # Step 4: if we get here, Shutdown
        self.stop()

    def stop(self):
        # type: () -> ()
        """
        Stop the current running optimization loop,
        Called from a different thread than the :meth:`start`.
        """
        # After the optimizer run, we must shutdown the master and the nameserver.
        self._bohb.shutdown(shutdown_workers=True)
        # no need to specifically shutdown the name server, hopefully pyro will do that
        # self._namespace.shutdown()

        if not self._res:
            return

        # Step 5: Analysis
        id2config = self._res.get_id2config_mapping()
        incumbent = self._res.get_incumbent_id()
        all_runs = self._res.get_all_runs()

        # Step 6: Print Analysis
        print('Best found configuration:', id2config[incumbent]['config'])
        print('A total of {} unique configurations where sampled.'.format(len(id2config.keys())))
        print('A total of {} runs where executed.'.format(len(self._res.get_all_runs())))
        print('Total budget corresponds to {:.1f} full function evaluations.'.format(
            sum([r.budget for r in all_runs]) / self._bohb_kwargs.get('max_budget', 1.0)))
        print('The run took {:.1f} seconds to complete.'.format(
            all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

    def _convert_hyper_parameters_to_cs(self):
        # type: () -> CS.ConfigurationSpace
        cs = CS.ConfigurationSpace(seed=self._seed)
        for p in self._hyper_parameters:
            if isinstance(p, UniformParameterRange):
                hp = CSH.UniformFloatHyperparameter(
                    p.name, lower=p.min_value, upper=p.max_value, log=False, q=p.step_size)
            elif isinstance(p, UniformIntegerParameterRange):
                hp = CSH.UniformIntegerHyperparameter(
                    p.name, lower=p.min_value, upper=p.max_value, log=False, q=p.step_size)
            elif isinstance(p, DiscreteParameterRange):
                hp = CSH.CategoricalHyperparameter(p.name, choices=p.values)
            else:
                raise ValueError("HyperParameter type {} not supported yet with OptimizerBOHB".format(type(p)))
            cs.add_hyperparameter(hp)

        return cs
