import logging

from trains import Task
from trains.automation import DiscreteParameterRange, HyperParameterOptimizer, RandomSearch, \
    UniformIntegerParameterRange

try:
    from trains.automation.hpbandster import OptimizerBOHB
    Our_SearchStrategy = OptimizerBOHB
except ValueError:
    logging.getLogger().warning(
        'Apologies, it seems you do not have \'hpbandster\' installed, '
        'we will be using RandomSearch strategy instead\n'
        'If you like to try ' '{{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},\n'
        'run: pip install hpbandster')
    Our_SearchStrategy = RandomSearch


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


# Connecting TRAINS
task = Task.init(project_name='Hyper-Parameter Optimization',
                 task_name='Automatic Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

# experiment template to optimize in the hyper-parameter optimization
args = {
    'template_task_id': None,
    'run_as_service': False,
}
args = task.connect(args)

# Get the template task experiment that we want to optimize
if not args['template_task_id']:
    args['template_task_id'] = Task.get_task(
        project_name='examples', task_name='Keras HP optimization base').id

# Example use case:
an_optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id=args['template_task_id'],
    # here we define the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('layer_1', min_value=128, max_value=512, step_size=128),
        UniformIntegerParameterRange('layer_2', min_value=128, max_value=512, step_size=128),
        DiscreteParameterRange('batch_size', values=[96, 128, 160]),
        DiscreteParameterRange('epochs', values=[30]),
    ],
    # this is the objective metric we want to maximize/minimize
    objective_metric_title='val_acc',
    objective_metric_series='val_acc',
    # now we decide if we want to maximize it or minimize it (accuracy we maximize)
    objective_metric_sign='max',
    # let us limit the number of concurrent experiments,
    # this in turn will make sure we do dont bombard the scheduler with experiments.
    # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
    max_number_of_concurrent_tasks=2,
    # this is the optimizer class (actually doing the optimization)
    # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
    # more are coming soon...
    optimizer_class=Our_SearchStrategy,
    # Select an execution queue to schedule the experiments for execution
    execution_queue='1xGPU',
    # Optional: Limit the execution time of a single experiment, in minutes.
    # (this is optional, and if using  OptimizerBOHB, it is ignored)
    time_limit_per_job=10.,
    # Check the experiments every 6 seconds is way too often, we should probably set it to 5 min,
    # assuming a single experiment is usually hours...
    pool_period_min=0.1,
    # set the maximum number of jobs to launch for the optimization, default (None) unlimited
    # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
    # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
    total_max_jobs=10,
    # This is only applicable for OptimizerBOHB and ignore by the rest
    # set the minimum number of iterations for an experiment, before early stopping
    min_iteration_per_job=10,
    # Set the maximum number of iterations for an experiment to execute
    # (This is optional, unless using OptimizerBOHB where this is a must)
    max_iteration_per_job=30,
)

# if we are running as a service, just enqueue ourselves into the services queue and let it run the optimization
if args['run_as_service']:
    # if this code is executed by `trains-agent` the function call does nothing.
    # if executed locally, the local process will be terminated, and a remote copy will be executed instead
    task.execute_remotely(queue_name='services', exit_process=True)

# report every 12 seconds, this is way too often, but we are testing here J
an_optimizer.set_report_period(2.2)
# start the optimization process, callback function to be called every time an experiment is completed
# this function returns immediately
an_optimizer.start(job_complete_callback=job_complete_callback)
# set the time limit for the optimization process (2 hours)
an_optimizer.set_time_limit(in_minutes=120.0)
# wait until process is done (notice we are controlling the optimization process in the background)
an_optimizer.wait()
# optimization is completed, print the top performing experiments id
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
# make sure background optimization stopped
an_optimizer.stop()

print('We are done, good bye')
