import sys

from clearml import Task
from clearml.automation import DiscreteParameterRange, HyperParameterOptimizer, UniformIntegerParameterRange


try:
    from clearml.automation.optuna import OptimizerOptuna  # noqa
except ImportError:
    print("Multi-objective HPO is currently only supported via Optuna")
    sys.exit(0)


if __name__ == "__main__":
    task = Task.init(
        project_name="Hyper-Parameter Optimization",
        task_name="Multi-objective HPO",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    # experiment template to optimize in the hyper-parameter optimization
    args = {
        "template_task_id": None,
        "run_as_service": False,
    }
    args = task.connect(args)

    # Get the template task experiment that we want to optimize
    if not args["template_task_id"]:
        args["template_task_id"] = Task.get_task(project_name="examples", task_name="Keras HP optimization base").id

    # Set default queue name for the Training tasks themselves.
    # later can be overridden in the UI
    execution_queue = "1xGPU"

    an_optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=args["template_task_id"],
        # here we define the hyper-parameters to optimize
        # Notice: The parameter name should exactly match what you see in the UI: <section_name>/<parameter>
        # For Example, here we see in the base experiment a section Named: "General"
        # under it a parameter named "batch_size", this becomes "General/batch_size"
        # If you have `argparse` for example, then arguments will appear under the "Args" section,
        # and you should instead pass "Args/batch_size"
        hyper_parameters=[
            UniformIntegerParameterRange("General/layer_1", min_value=128, max_value=512, step_size=128),
            UniformIntegerParameterRange("General/layer_2", min_value=128, max_value=512, step_size=128),
            DiscreteParameterRange("General/batch_size", values=[96, 128, 160]),
            DiscreteParameterRange("General/epochs", values=[30]),
        ],
        # this is the objectives' metric/series we want to maximize/minimize
        objective_metric_title=["evaluate", "evaluate"],
        objective_metric_series=["score", "accuracy"],
        # now we decide if we want to maximize it or minimize them
        # in this case, we want to minimize evaluate/score and maximize evaluate/accuracy
        objective_metric_sign=["min", "max"],
        # let us limit the number of concurrent experiments,
        # this in turn will make sure we do dont bombard the scheduler with experiments.
        # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
        max_number_of_concurrent_tasks=1,
        # optimizer_class has to be OptimizerOptuna
        optimizer_class=OptimizerOptuna,
        # Select an execution queue to schedule the experiments for execution
        execution_queue=execution_queue,
        # If specified all Tasks created by the HPO process will be created under the `spawned_project` project
        spawn_project=None,  # 'HPO spawn project',
        # If specified only the top K performing Tasks will be kept, the others will be automatically archived
        save_top_k_tasks_only=None,  # 5,
        # Optional: Limit the execution time of a single experiment, in minutes.
        # (this is optional, and if using  OptimizerBOHB, it is ignored)
        time_limit_per_job=10.0,
        # Check the experiments every 12 seconds is way too often, we should probably set it to 5 min,
        # assuming a single experiment is usually hours...
        pool_period_min=0.2,
        # set the maximum number of jobs to launch for the optimization, default (None) unlimited
        # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
        # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
        total_max_jobs=10,
        # set the minimum number of iterations for an experiment, before early stopping.
        # Does not apply for simple strategies such as RandomSearch or GridSearch
        min_iteration_per_job=10,
        # Set the maximum number of iterations for an experiment to execute
        # (This is optional, unless using OptimizerBOHB where this is a must)
        max_iteration_per_job=30,
    )

    # if we are running as a service, just enqueue ourselves into the services queue and let it run the optimization
    if args["run_as_service"]:
        # if this code is executed by `clearml-agent` the function call does nothing.
        # if executed locally, the local process will be terminated, and a remote copy will be executed instead
        task.execute_remotely(queue_name="services", exit_process=True)

    # report every 12 seconds, this is way too often, but we are testing here
    an_optimizer.set_report_period(0.2)
    # start the optimization process, callback function to be called every time an experiment is completed
    # this function returns immediately
    an_optimizer.start()
    # You can also use the line below instead to run all the optimizer tasks locally, without using queues or agent
    # an_optimizer.start_locally(job_complete_callback=job_complete_callback)
    # set the time limit for the optimization process (2 hours)
    an_optimizer.set_time_limit(in_minutes=120.0)
    # wait until process is done (notice we are controlling the optimization process in the background)
    an_optimizer.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = an_optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    an_optimizer.stop()

    print("We are done, good bye")
