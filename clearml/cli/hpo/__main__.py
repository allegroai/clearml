import sys
import json
from argparse import ArgumentParser, RawTextHelpFormatter

from clearml.automation import (
    DiscreteParameterRange,
    UniformIntegerParameterRange,
    UniformParameterRange,
    LogUniformParameterRange,
    HyperParameterOptimizer,
    RandomSearch,
    GridSearch,
)
from clearml import Task
from clearml.backend_interface.task.populate import CreateAndPopulate

try:
    from clearml.automation.optuna import OptimizerOptuna  # noqa
except ImportError:
    OptimizerOptuna = None
try:
    from clearml.automation.hpbandster import OptimizerBOHB
except ImportError:
    OptimizerBOHB = None


def setup_parser(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Name of the project in which the experiment will be created. If the project does not exist,"
        " it is created. If project_name is None, the repository name is used",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="The name of Task (experiment). "
        "If task_name is None, the Python experiment script's file name is used.",
    )
    group.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="ID of the task to execute and optimize its parameters. Required unless '--script' is specified",
    )
    group.add_argument(
        "--script",
        type=str,
        default=None,
        help="Script to run the parameter search on. Required unless '--task-id' is specified",
    )
    parser.add_argument("--queue", type=str, default=None, help="Queue to run the '--script' from")
    parser.add_argument(
        "--params-search",
        type=str,
        nargs="*",
        default=[],
        help="List of parameters to search optimal value of. Each parameter must be a JSON having the following format:\n"
        "{\n"
        '  "name": str,  # Name of the paramter you want to optimize\n'
        '  "type":  Union["LogUniformParameterRange", "UniformParameterRange", "UniformIntegerParameterRange", "DiscreteParameterRange"],\n'
        "  # other fields depending on type\n"
        "}\n"
        "The fields corresponding to each parameters are:\n"
        "  - LogUniformParameterRange:\n"
        "    - min_value: float  # The minimum exponent sample to use for uniform random sampling\n"
        "    - max_value: float  # The maximum exponent sample to use for uniform random sampling\n"
        "    - base: Optional[float]  # The base used to raise the sampled exponent. Default: 10\n"
        "    - step_size: Optional[float]  # If not None, set step size (quantization) for value sampling. Default: None\n"
        "    - include_max_value: Optional[bool]  # Whether or not to include the max_value in range. Default: True\n"
        "  - UniformParameterRange:\n"
        "    - min_value: float  # The minimum exponent sample to use for uniform random sampling\n"
        "    - max_value: float  # The maximum exponent sample to use for uniform random sampling\n"
        "    - step_size: Optional[float]  # If not None, set step size (quantization) for value sampling. Default: None\n"
        "    - include_max_value: Optional[bool]  # Whether or not to include the max_value in range. Default: True\n"
        "  - UniformIntegerParameterRange:\n"
        "    - min_value: float  # The minimum exponent sample to use for uniform random sampling\n"
        "    - max_value: float  # The maximum exponent sample to use for uniform random sampling\n"
        "    - step_size: Optional[int]  # Default: 1\n"
        "    - include_max_value: Optional[bool]  # Whether or not to include the max_value in range. Default: True\n"
        "  - DiscreteParameterRange:\n"
        "    - values: List[Any]  # The list of valid parameter values to sample from",
    )
    parser.add_argument(
        "--params-override",
        type=str,
        nargs="*",
        default=[],
        help="List of parameters to override (won't be searched, just override default). "
        "Each parameter must be a JSON having the following format:\n"
        "{\n"
        '  "name": str,  # name of the parameter to override\n'
        '  "value": Any  # value of the paramter being overriden\n'
        "}",
    )
    parser.add_argument(
        "--objective-metric-title",
        type=str,
        required=True,
        help="The Objective metric title to maximize/minimize. Example: 'validation'",
    )
    parser.add_argument(
        "--objective-metric-series",
        type=str,
        required=True,
        help="The Objective metric series to maximize/minimize. Example: 'loss'",
    )
    parser.add_argument(
        "--objective-metric-sign",
        choices=["min", "max", "min_global", "max_global"],
        required=True,
        help="The objective to maximize/minimize. The values are:\n"
        "- min - Minimize the last reported value for the specified title/series scalar\n"
        "- max - Maximize the last reported value for the specified title/series scalar\n"
        "- min_global - Minimize the min value of *all* reported values for the specific title/series scalar\n"
        "- max_global - Maximize the max value of *all* reported values for the specific title/series scalar",
    )
    parser.add_argument(
        "--optimizer-class",
        choices=["OptimizerOptuna", "OptimizerBOHB", "GridSearch", "RandomSearch"],
        default="OptimizerOptuna",
        help="Type of optimization. Possible values are: OptimizerOptuna (default), OptimizerBOHB, GridSearch, RandomSearch",
    )
    parser.add_argument(
        "--optimization-time-limit",
        type=float,
        default=None,
        help="The maximum time (minutes) for the entire optimization process."
        " The default is None, indicating no time limit",
    )
    parser.add_argument(
        "--compute-time-limit",
        type=float,
        default=None,
        help="The maximum compute time in minutes. When time limit is exceeded, all jobs aborted",
    )
    parser.add_argument(
        "--pool-period-min", type=float, default=None, help="The time between two consecutive pools (minutes)"
    )
    parser.add_argument(
        "--total-max-jobs",
        type=int,
        default=None,
        help="The total maximum jobs for the optimization process. The default value is None, for unlimited",
    )
    parser.add_argument(
        "--min-iteration-per-job",
        type=int,
        default=None,
        help="The minimum iterations (of the Objective metric) per single job",
    )
    parser.add_argument(
        "--max-iteration-per-job",
        type=int,
        default=None,
        help="The maximum iterations (of the Objective metric) per single job."
        " When maximum iterations is exceeded, the job is aborted",
    )
    parser.add_argument(
        "--save-top-k-tasks-only",
        type=int,
        default=10,
        help="If above 0, keep only the top_k performing Tasks, and archive the rest of the created Tasks."
        " If -1, keep everything, nothing will be archived. Default: 10",
    )
    parser.add_argument(
        "--time-limit-per-job",
        type=float,
        default=None,
        help="Maximum execution time per single job in minutes. When time limit is exceeded job is aborted."
        " Default: no time limit",
    )


def eval_params_search(params_search, params_override):
    def eval_param(param_):
        param_ = json.loads(param_)
        if "/" not in param_["name"]:
            param_["name"] = "General/{}".format(param_["name"])
        return param_

    type_map = {
        "LogUniformParameterRange": LogUniformParameterRange,
        "UniformParameterRange": UniformParameterRange,
        "UniformIntegerParameterRange": UniformIntegerParameterRange,
        "DiscreteParameterRange": DiscreteParameterRange,
    }
    result = []
    for param in params_search:
        param = eval_param(param)
        if param["type"] not in type_map:
            print("Invalid parameter type '{}'".format(param["type"]))
            exit(1)
        type_str = param.pop("type")
        type_ = type_map[type_str]
        try:
            result.append(type_(**param))
        except Exception as e:
            print("Failed instantiating object of type {} with arguments {}. Error is: {}".format(type_str, param, e))
            exit(1)
    for param in params_override:
        param = eval_param(param)
        result.append(DiscreteParameterRange(param["name"], values=[param["value"]]))
    return result


def eval_optimizer_class(optimizer_class):
    type_map = {
        "OptimizerOptuna": OptimizerOptuna,
        "OptimizerBOHB": OptimizerBOHB,
        "GridSearch": GridSearch,
        "RandomSearch": RandomSearch,
    }
    errors = {
        "OptimizerOptuna": "OptimizerOptuna requires 'optuna' package, it was not found.\n"
        " Install with: pip install optuna",
        "OptimizerBOHB": "OptimizerBOHB requires 'hpbandster' package, it was not found.\n"
        "Install with: pip install hpbandster",
    }
    if optimizer_class not in type_map:
        print("Invalid optimizer class '{}'".format(optimizer_class))
        exit(1)
    if type_map[optimizer_class] is None:
        print(errors[optimizer_class])
        exit(1)
    return type_map[optimizer_class]


def build_opt_kwargs(args):
    kwargs = {}
    optional_arg_names = [
        "time_limit_per_job",
        "optimization_time_limit",
        "compute_time_limit",
        "pool_period_min",
        "total_max_jobs",
        "min_iteration_per_job",
        "max_iteration_per_job",
    ]
    for arg_name in optional_arg_names:
        arg_val = getattr(args, arg_name)
        if arg_val is not None:
            kwargs[arg_name] = arg_val
    return kwargs


def cli():
    title = "ClearML HPO - search for the best parameters for your models"
    print(title)
    parser = ArgumentParser(description=title, formatter_class=RawTextHelpFormatter)
    setup_parser(parser)

    # get the args
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        exit(0)

    Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )
    task_id = args.task_id
    if not task_id:
        if args.queue is None:
            print("No queue supplied to run the script from")
            exit(1)
        create_populate = CreateAndPopulate(script=args.script)
        create_populate.create_task()
        Task.enqueue(create_populate.task, queue_name=args.queue)
        task_id = create_populate.get_id()
    optimizer = HyperParameterOptimizer(
        base_task_id=task_id,
        hyper_parameters=eval_params_search(args.params_search, args.params_override),
        objective_metric_title=args.objective_metric_title,
        objective_metric_series=args.objective_metric_series,
        objective_metric_sign=args.objective_metric_sign,
        optimizer_class=eval_optimizer_class(args.optimizer_class),
        save_top_k_tasks_only=args.save_top_k_tasks_only,
        **build_opt_kwargs(args),
    )
    optimizer.start()
    optimizer.wait()
    print("Optimization completed!")
    top_experiments_cnt = 10
    if args.save_top_k_tasks_only != -1 and top_experiments_cnt > args.save_top_k_tasks_only:
        top_experiments_cnt = args.save_top_k_tasks_only
    print("Top {} experiments are: ".format(top_experiments_cnt))
    top_exp = optimizer.get_top_experiments(top_k=top_experiments_cnt)
    print([t.id for t in top_exp])
    optimizer.stop()


def main():
    try:
        cli()
    except KeyboardInterrupt:
        print("\nUser aborted")
    except Exception as ex:
        print("\nError: {}".format(ex))
        exit(1)


if __name__ == "__main__":
    main()
