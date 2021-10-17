import functools
import inspect
import json
import re
from copy import copy, deepcopy
from datetime import datetime
from logging import getLogger
from threading import Thread, Event, RLock
from time import time
from typing import Sequence, Optional, Mapping, Callable, Any, List, Dict, Union, Tuple

from attr import attrib, attrs

from .job import LocalClearmlJob, RunningJob
from .. import Logger
from ..automation import ClearmlJob
from ..backend_interface.task.populate import CreateFromFunction
from ..backend_interface.util import get_or_create_project, exact_match_regex
from ..debugging.log import LoggerRoot
from ..model import BaseModel
from ..task import Task
from ..utilities.process.mp import leave_process
from ..utilities.proxy_object import LazyEvalWrapper, flatten_dictionary


class PipelineController(object):
    """
    Pipeline controller.
    Pipeline is a DAG of base tasks, each task will be cloned (arguments changed as required) executed and monitored
    The pipeline process (task) itself can be executed manually or by the clearml-agent services queue.
    Notice: The pipeline controller lives as long as the pipeline itself is being executed.
    """
    _tag = 'pipeline'
    _node_tag_prefix = 'pipe:'
    _step_pattern = r"\${[^}]*}"
    _config_section = 'Pipeline'
    _args_section = 'Args'
    _pipeline_step_ref = 'pipeline'
    _runtime_property_hash = '_pipeline_hash'
    _reserved_pipeline_names = (_pipeline_step_ref, )
    _task_project_lookup = {}
    _clearml_job_class = ClearmlJob
    _update_execution_plot_interval = 5.*60
    _monitor_node_interval = 5.*60

    @attrs
    class Node(object):
        name = attrib(type=str)  # pipeline step name
        base_task_id = attrib(type=str, default=None)  # base Task ID to be cloned and launched
        task_factory_func = attrib(type=Callable, default=None)  # alternative to base_task_id, function creating a Task
        queue = attrib(type=str, default=None)  # execution queue name to use
        parents = attrib(type=list, default=[])  # list of parent DAG steps
        timeout = attrib(type=float, default=None)  # execution timeout limit
        parameters = attrib(type=dict, default={})  # Task hyper parameters to change
        configurations = attrib(type=dict, default={})  # Task configuration objects to change
        task_overrides = attrib(type=dict, default={})  # Task overrides to change
        executed = attrib(type=str, default=None)  # The actual executed Task ID (None if not executed yet)
        clone_task = attrib(type=bool, default=True)  # If True cline the base_task_id, then execute the cloned Task
        job = attrib(type=ClearmlJob, default=None)  # ClearMLJob object
        skip_job = attrib(type=bool, default=False)  # if True, this step should be skipped
        continue_on_fail = attrib(type=bool, default=False)  # if True, the pipeline continues even if the step failed
        cache_executed_step = attrib(type=bool, default=False)  # if True this pipeline step should be cached
        return_artifacts = attrib(type=list, default=[])  # List of artifact names returned by the step
        monitor_metrics = attrib(type=list, default=[])  # List of metric title/series to monitor
        monitor_artifacts = attrib(type=list, default=[])  # List of artifact names to monitor
        monitor_models = attrib(type=list, default=[])  # List of models to monitor

        def copy(self):
            # type: () -> PipelineController.Node
            """
            return a copy of the current Node, excluding the `job`, `executed`, fields
            :return: new Node copy
            """
            new_copy = PipelineController.Node(
                name=self.name,
                **dict((k, deepcopy(v)) for k, v in self.__dict__.items()
                       if k not in ('name', 'job', 'executed', 'task_factory_func'))
            )
            new_copy.task_factory_func = self.task_factory_func
            return new_copy

    def __init__(
            self,
            name,  # type: str
            project,  # type: str
            version,  # type: str
            pool_frequency=0.2,  # type: float
            add_pipeline_tags=False,  # type: bool
            target_project=None,  # type: Optional[str]
            auto_version_bump=True,  # type: bool
            abort_on_failure=False,  # type: bool
    ):
        # type: (...) -> None
        """
        Create a new pipeline controller. The newly created object will launch and monitor the new experiments.

        :param name: Provide pipeline name (if main Task exists it overrides its name)
        :param project: Provide project storing the pipeline (if main Task exists  it overrides its project)
        :param version: Must provide pipeline version. This version allows to uniquely identify the pipeline
            template execution. Examples for semantic versions: version='1.0.1' , version='23', version='1.2'
        :param float pool_frequency: The pooling frequency (in minutes) for monitoring experiments / states.
        :param bool add_pipeline_tags: (default: False) if True, add `pipe: <pipeline_task_id>` tag to all
            steps (Tasks) created by this pipeline.
        :param str target_project: If provided, all pipeline steps are cloned into the target project
        :param bool auto_version_bump: If True (default), if the same pipeline version already exists
            (with any difference from the current one), the current pipeline version will be bumped to a new version
            version bump examples: 1.0.0 -> 1.0.1 , 1.2 -> 1.3, 10 -> 11 etc.
        :param bool abort_on_failure: If False (default), failed pipeline steps will not cause the pipeline
            to stop immediately, instead any step that is not connected (or indeirectly connected) to the failed step,
            will still be executed. Nonetheless the pipeline itself will be marked failed, unless the failed step
            was specifically defined with "continue_on_fail=True".
            If True, any failed step will cause the pipeline to immediately abort, stop all running steps,
            and mark the pipeline as failed.
        """
        self._nodes = {}
        self._running_nodes = []
        self._start_time = None
        self._pipeline_time_limit = None
        self._default_execution_queue = None
        self._version = str(version).strip()
        if not self._version or not all(i and i.isnumeric() for i in self._version.split('.')):
            raise ValueError(
                "Pipeline version has to be in a semantic version form, "
                "examples: version='1.0.1', version='1.2', version='23'")
        self._pool_frequency = pool_frequency * 60.
        self._thread = None
        self._pipeline_args = dict()
        self._pipeline_args_desc = dict()
        self._stop_event = None
        self._experiment_created_cb = None
        self._experiment_completed_cb = None
        self._pre_step_callbacks = {}
        self._post_step_callbacks = {}
        self._target_project = target_project or ''
        self._add_pipeline_tags = add_pipeline_tags
        self._task = Task.current_task()
        self._step_ref_pattern = re.compile(self._step_pattern)
        self._reporting_lock = RLock()
        self._pipeline_task_status_failed = None
        self._auto_version_bump = bool(auto_version_bump)
        self._mock_execution = False  # used for nested pipelines (eager execution)
        if not self._task:
            self._task = Task.init(
                project_name=project or 'Pipelines',
                task_name=name or 'Pipeline {}'.format(datetime.now()),
                task_type=Task.TaskTypes.controller,
                auto_resource_monitoring=False,
                reuse_last_task_id=False
            )
            self._task.set_system_tags((self._task.get_system_tags() or []) + [self._tag])
            self._task.set_user_properties(version=self._version)

        self._auto_connect_task = bool(self._task)
        # make sure we add to the main Task the pipeline tag
        if self._task:
            self._task.add_tags([self._tag])

        self._monitored_nodes = {}  # type: Dict[str, dict]
        self._abort_running_steps_on_failure = abort_on_failure

    def set_default_execution_queue(self, default_execution_queue):
        # type: (Optional[str]) -> None
        """
        Set the default execution queue for if pipeline step does not specify an execution queue

        :param default_execution_queue: The execution queue to use if no execution queue is provided
        """
        self._default_execution_queue = str(default_execution_queue) if default_execution_queue else None

    def set_pipeline_execution_time_limit(self, max_execution_minutes):
        # type: (Optional[float]) -> None
        """
        Set maximum execution time (minutes) for the entire pipeline. Pass None or 0 to disable execution time limit.

        :param float max_execution_minutes: The maximum time (minutes) for the entire pipeline process. The
            default is ``None``, indicating no time limit.
        """
        self._pipeline_time_limit = max_execution_minutes * 60. if max_execution_minutes else None

    def add_step(
            self,
            name,  # type: str
            base_task_id=None,  # type: Optional[str]
            parents=None,  # type: Optional[Sequence[str]]
            parameter_override=None,  # type: Optional[Mapping[str, Any]]
            configuration_overrides=None,  # type: Optional[Mapping[str, Union[str, Mapping]]]
            task_overrides=None,  # type: Optional[Mapping[str, Any]]
            execution_queue=None,  # type: Optional[str]
            monitor_metrics=None,  # type: Optional[List[Union[Tuple[str, str], Tuple[(str, str), (str, str)]]]]
            monitor_artifacts=None,  # type: Optional[List[Union[str, Tuple[str, str]]]]
            monitor_models=None,  # type: Optional[List[Union[str, Tuple[str, str]]]]
            time_limit=None,  # type: Optional[float]
            base_task_project=None,  # type: Optional[str]
            base_task_name=None,  # type: Optional[str]
            clone_base_task=True,  # type: bool
            continue_on_fail=False,  # type: bool
            pre_execute_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node, dict], bool]]  # noqa
            post_execute_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node], None]]  # noqa
            cache_executed_step=False,  # type: bool
            base_task_factory=None,  # type: Optional[Callable[[PipelineController.Node], Task]]
    ):
        # type: (...) -> bool
        """
        Add a step to the pipeline execution DAG.
        Each step must have a unique name (this name will later be used to address the step)

        :param name: Unique of the step. For example `stage1`
        :param base_task_id: The Task ID to use for the step. Each time the step is executed,
            the base Task is cloned, then the cloned task will be sent for execution.
        :param parents: Optional list of parent nodes in the DAG.
            The current step in the pipeline will be sent for execution only after all the parent nodes
            have been executed successfully.
        :param parameter_override: Optional parameter overriding dictionary.
            The dict values can reference a previously executed step using the following form '${step_name}'
            Examples:
            - Artifact access
                parameter_override={'Args/input_file': '${stage1.artifacts.mydata.url}' }
            - Model access (last model used)
                parameter_override={'Args/input_file': '${stage1.models.output.-1.url}' }
            - Parameter access
                parameter_override={'Args/input_file': '${stage3.parameters.Args/input_file}' }
            - Task ID
                parameter_override={'Args/input_file': '${stage3.id}' }
        :param configuration_overrides: Optional, override Task configuration objects.
            Expected dictionary of configuration object name and configuration object content.
            Examples:
                {'General': dict(key='value')}
                {'General': 'configuration file content'}
                {'OmegaConf': YAML.dumps(full_hydra_dict)}
        :param task_overrides: Optional task section overriding dictionary.
            The dict values can reference a previously executed step using the following form '${step_name}'
            Examples:
            - get the latest commit from a specific branch
                task_overrides={'script.version_num': '', 'script.branch': 'main'}
            - match git repository branch to a previous step
                task_overrides={'script.branch': '${stage1.script.branch}', 'script.version_num': ''}
            - change container image
                task_overrides={'container.image': '${stage1.container.image}'}
            - match container image to a previous step
                task_overrides={'container.image': '${stage1.container.image}'}
        :param execution_queue: Optional, the queue to use for executing this specific step.
            If not provided, the task will be sent to the default execution queue, as defined on the class
        :param monitor_metrics: Optional, log the step's metrics on the pipeline Task.
            Format is a list of pairs metric (title, series) to log:
                [(step_metric_title, step_metric_series), ]
                Example: [('test', 'accuracy'), ]
            Or a list of tuple pairs, to specify a different target metric for to use on the pipeline Task:
                [((step_metric_title, step_metric_series), (target_metric_title, target_metric_series)), ]
                Example: [[('test', 'accuracy'), ('model', 'accuracy')], ]
        :param monitor_artifacts: Optional, log the step's artifacts on the pipeline Task.
            Provided a list of artifact names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: [('processed_data', 'final_processed_data'), ]
            Alternatively user can also provide a list of artifacts to monitor
            (target artifact name will be the same as original artifact name)
            Example: ['processed_data', ]
        :param monitor_models: Optional, log the step's output models on the pipeline Task.
            Provided a list of model names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: [('model_weights', 'final_model_weights'), ]
            Alternatively user can also provide a list of models to monitor
            (target models name will be the same as original model)
            Example: ['model_weights', ]
            To select the latest (lexicographic) model use "model_*", or the last created model with just "*"
            Example:  ['model_weights_*', ]
        :param time_limit: Default None, no time limit.
            Step execution time limit, if exceeded the Task is aborted and the pipeline is stopped and marked failed.
        :param base_task_project: If base_task_id is not given,
            use the base_task_project and base_task_name combination to retrieve the base_task_id to use for the step.
        :param base_task_name: If base_task_id is not given,
            use the base_task_project and base_task_name combination to retrieve the base_task_id to use for the step.
        :param clone_base_task: If True (default) the pipeline will clone the base task, and modify/enqueue
            the cloned Task. If False, the base-task is used directly, notice it has to be in draft-mode (created).
        :param continue_on_fail: (default False). If True, failed step will not cause the pipeline to stop
            (or marked as failed). Notice, that steps that are connected (or indirectly connected)
            to the failed step will be skipped.
        :param pre_execute_callback: Callback function, called when the step (Task) is created
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            `parameters` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. `${step1.parameters.Args/param}` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param post_execute_callback: Callback function, called when a step (Task) is completed
            and it other jobs are executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass

        :param cache_executed_step: If True, before launching the new step,
            after updating with the latest configuration, check if an exact Task with the same parameter/code
            was already executed. If it was found, use it instead of launching a new Task.
            Default: False, a new cloned copy of base_task is always used.
            Notice: If the git repo reference does not have a specific commit ID, the Task will never be used.
            If `clone_base_task` is False there is no cloning, hence the base_task is used.
        :param base_task_factory: Optional, instead of providing a pre-existing Task,
            provide a Callable function to create the Task (returns Task object)

        :return: True if successful
        """

        # always store callback functions (even when running remotely)
        if pre_execute_callback:
            self._pre_step_callbacks[name] = pre_execute_callback
        if post_execute_callback:
            self._post_step_callbacks[name] = post_execute_callback

        # when running remotely do nothing, we will deserialize ourselves when we start
        # if we are not cloning a Task, we assume this step is created from code, not from the configuration
        if not base_task_factory and clone_base_task and self._has_stored_configuration():
            return True

        self._verify_node_name(name)

        if not base_task_factory and not base_task_id:
            if not base_task_project or not base_task_name:
                raise ValueError('Either base_task_id or base_task_project/base_task_name must be provided')
            base_task = Task.get_task(
                project_name=base_task_project,
                task_name=base_task_name,
                allow_archived=True,
                task_filter=dict(
                    status=[str(Task.TaskStatusEnum.created), str(Task.TaskStatusEnum.queued),
                            str(Task.TaskStatusEnum.in_progress), str(Task.TaskStatusEnum.published),
                            str(Task.TaskStatusEnum.stopped), str(Task.TaskStatusEnum.completed),
                            str(Task.TaskStatusEnum.closed)],
                )
            )
            if not base_task:
                raise ValueError('Could not find base_task_project={} base_task_name={}'.format(
                    base_task_project, base_task_name))
            if Task.archived_tag in base_task.get_system_tags():
                LoggerRoot.get_base_logger().warning(
                    'Found base_task_project={} base_task_name={} but it is archived'.format(
                        base_task_project, base_task_name))
            base_task_id = base_task.id

        if configuration_overrides is not None:
            # verify we have a dict or a string on all values
            if not isinstance(configuration_overrides, dict) or \
                    not all(isinstance(v, (str, dict)) for v in configuration_overrides.values()):
                raise ValueError("configuration_overrides must be a dictionary, with all values "
                                 "either dicts or strings, got \'{}\' instead".format(configuration_overrides))

        if task_overrides:
            task_overrides = flatten_dictionary(task_overrides, sep='.')

        self._nodes[name] = self.Node(
            name=name, base_task_id=base_task_id, parents=parents or [],
            queue=execution_queue, timeout=time_limit,
            parameters=parameter_override or {},
            configurations=configuration_overrides,
            clone_task=clone_base_task,
            task_overrides=task_overrides,
            cache_executed_step=cache_executed_step,
            continue_on_fail=continue_on_fail,
            task_factory_func=base_task_factory,
            monitor_metrics=monitor_metrics or [],
            monitor_artifacts=monitor_artifacts or [],
            monitor_models=monitor_models or [],
        )

        if self._task and not self._task.running_locally():
            self.update_execution_plot()

        return True

    def add_function_step(
            self,
            name,  # type: str
            function,  # type: Callable
            function_kwargs=None,  # type: Optional[Dict[str, Any]]
            function_return=None,  # type: Optional[List[str]]
            project_name=None,  # type: Optional[str]
            task_name=None,  # type: Optional[str]
            task_type=None,  # type: Optional[str]
            packages=None,  # type: Optional[Union[str, Sequence[str]]]
            repo=None,  # type: Optional[str]
            repo_branch=None,  # type: Optional[str]
            repo_commit=None,  # type: Optional[str]
            helper_functions=None,  # type: Optional[Sequence[Callable]]
            docker=None,  # type: Optional[str]
            docker_args=None,  # type: Optional[str]
            docker_bash_setup_script=None,  # type: Optional[str]
            parents=None,  # type: Optional[Sequence[str]],
            execution_queue=None,  # type: Optional[str]
            monitor_metrics=None,  # type: Optional[List[Union[Tuple[str, str], Tuple[(str, str), (str, str)]]]]
            monitor_artifacts=None,  # type: Optional[List[Union[str, Tuple[str, str]]]]
            monitor_models=None,  # type: Optional[List[Union[str, Tuple[str, str]]]]
            time_limit=None,  # type: Optional[float]
            continue_on_fail=False,  # type: bool
            pre_execute_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node, dict], bool]]  # noqa
            post_execute_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node], None]]  # noqa
            cache_executed_step=False,  # type: bool
    ):
        # type: (...) -> bool
        """
        Create a Task from a function, including wrapping the function input arguments
        into the hyper-parameter section as kwargs, and storing function results as named artifacts

        Example:
            def mock_func(a=6, b=9):
                c = a*b
                print(a, b, c)
                return c, c**2

            create_task_from_function(mock_func, function_return=['mul', 'square'])

        Example arguments from other Tasks (artifact):
            def mock_func(matrix_np):
                c = matrix_np*matrix_np
                print(matrix_np, c)
                return c

            create_task_from_function(
                mock_func,
                function_input_artifacts={'matrix_np': 'aabb1122.previous_matrix'},
                function_return=['square_matrix']
            )

        :param name: Unique of the step. For example `stage1`
        :param function: A global function to convert into a standalone Task
        :param function_kwargs: Optional, provide subset of function arguments and default values to expose.
            If not provided automatically take all function arguments & defaults
            Optional, pass input arguments to the function from other Tasks's output artifact.
            Example argument named `numpy_matrix` from Task ID `aabbcc` artifact name `answer`:
            {'numpy_matrix': 'aabbcc.answer'}
        :param function_return: Provide a list of names for all the results.
            If not provided no results will be stored as artifacts.
        :param project_name: Set the project name for the task. Required if base_task_id is None.
        :param task_name: Set the name of the remote task. Required if base_task_id is None.
        :param task_type: Optional, The task type to be created. Supported values: 'training', 'testing', 'inference',
            'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc', 'custom'
        :param packages: Manually specify a list of required packages or a local requirements.txt file.
            Example: ["tqdm>=2.1", "scikit-learn"] or "./requirements.txt"
            If not provided, packages are automatically added based on the imports used in the function.
        :param repo: Optional, specify a repository to attach to the function, when remotely executing.
            Allow users to execute the function inside the specified repository, enabling to load modules/script
            from a repository Notice the execution work directory will be the repository root folder.
            Supports both git repo url link, and local repository path.
            Example remote url: 'https://github.com/user/repo.git'
            Example local repo copy: './repo' -> will automatically store the remote
            repo url and commit ID based on the locally cloned copy
        :param repo_branch: Optional, specify the remote repository branch (Ignored, if local repo path is used)
        :param repo_commit: Optional, specify the repository commit id (Ignored, if local repo path is used)
        :param helper_functions: Optional, a list of helper functions to make available
            for the standalone function Task.
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environment
        :param parents: Optional list of parent nodes in the DAG.
            The current step in the pipeline will be sent for execution only after all the parent nodes
            have been executed successfully.
        :param execution_queue: Optional, the queue to use for executing this specific step.
            If not provided, the task will be sent to the default execution queue, as defined on the class
        :param monitor_metrics: Optional, log the step's metrics on the pipeline Task.
            Format is a list of pairs metric (title, series) to log:
                [(step_metric_title, step_metric_series), ]
                Example: [('test', 'accuracy'), ]
            Or a list of tuple pairs, to specify a different target metric for to use on the pipeline Task:
                [((step_metric_title, step_metric_series), (target_metric_title, target_metric_series)), ]
                Example: [[('test', 'accuracy'), ('model', 'accuracy')], ]
        :param monitor_artifacts: Optional, log the step's artifacts on the pipeline Task.
            Provided a list of artifact names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: [('processed_data', 'final_processed_data'), ]
            Alternatively user can also provide a list of artifacts to monitor
            (target artifact name will be the same as original artifact name)
            Example: ['processed_data', ]
        :param monitor_models: Optional, log the step's output models on the pipeline Task.
            Provided a list of model names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: [('model_weights', 'final_model_weights'), ]
            Alternatively user can also provide a list of models to monitor
            (target models name will be the same as original model)
            Example: ['model_weights', ]
            To select the latest (lexicographic) model use "model_*", or the last created model with just "*"
            Example:  ['model_weights_*', ]
        :param time_limit: Default None, no time limit.
            Step execution time limit, if exceeded the Task is aborted and the pipeline is stopped and marked failed.
        :param continue_on_fail: (default False). If True, failed step will not cause the pipeline to stop
            (or marked as failed). Notice, that steps that are connected (or indirectly connected)
            to the failed step will be skipped.
        :param pre_execute_callback: Callback function, called when the step (Task) is created
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            `parameters` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. `${step1.parameters.Args/param}` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param post_execute_callback: Callback function, called when a step (Task) is completed
            and it other jobs are executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass

        :param cache_executed_step: If True, before launching the new step,
            after updating with the latest configuration, check if an exact Task with the same parameter/code
            was already executed. If it was found, use it instead of launching a new Task.
            Default: False, a new cloned copy of base_task is always used.
            Notice: If the git repo reference does not have a specific commit ID, the Task will never be used.

        :return: True if successful
        """
        # always store callback functions (even when running remotely)
        if pre_execute_callback:
            self._pre_step_callbacks[name] = pre_execute_callback
        if post_execute_callback:
            self._post_step_callbacks[name] = post_execute_callback

        self._verify_node_name(name)

        function_kwargs = function_kwargs or {}
        function_input_artifacts = {}
        # go over function_kwargs, split it into string and input artifacts
        for k, v in function_kwargs.items():
            if v and self._step_ref_pattern.match(str(v)):
                # check for step artifacts
                step, _, artifact = v[2:-1].partition('.')
                if step in self._nodes and artifact in self._nodes[step].return_artifacts:
                    function_input_artifacts[k] = "${{{}.id}}.{}".format(step, artifact)
                    continue
                # verify the reference
                self.__verify_step_reference(node=self.Node(name=name), step_ref_string=v)

        function_kwargs = {k: v for k, v in function_kwargs.items() if k not in function_input_artifacts}
        parameters = {"{}/{}".format(CreateFromFunction.kwargs_section, k): v for k, v in function_kwargs.items()}
        if function_input_artifacts:
            parameters.update(
                {"{}/{}".format(CreateFromFunction.input_artifact_section, k): str(v)
                 for k, v in function_input_artifacts.items()}
            )

        if self._mock_execution:
            project_name = project_name or self._target_project or self._task.get_project_name()

            task_definition = self._create_task_from_function(
                docker, docker_args, docker_bash_setup_script, function,
                function_input_artifacts, function_kwargs,
                function_return, packages, project_name, task_name,
                task_type, repo, repo_branch, repo_commit, helper_functions)

        elif self._task.running_locally():
            project_name = project_name or self._target_project or self._task.get_project_name()

            task_definition = self._create_task_from_function(
                docker, docker_args, docker_bash_setup_script, function,
                function_input_artifacts, function_kwargs,
                function_return, packages, project_name, task_name,
                task_type, repo, repo_branch, repo_commit, helper_functions)
            # update configuration with the task definitions
            # noinspection PyProtectedMember
            self._task._set_configuration(
                name=name, config_type='json',
                config_text=json.dumps(task_definition, indent=1)
            )
        else:
            # load task definition from configuration
            # noinspection PyProtectedMember
            task_definition = json.loads(self._task._get_configuration_text(name=name))

        def _create_task(_):
            a_task = Task.create(
                project_name=project_name,
                task_name=task_definition.get('name'),
                task_type=task_definition.get('type'),
            )
            # replace reference
            a_task.update_task(task_definition)
            return a_task

        self._nodes[name] = self.Node(
            name=name, base_task_id=None, parents=parents or [],
            queue=execution_queue, timeout=time_limit,
            parameters=parameters,
            clone_task=False,
            cache_executed_step=cache_executed_step,
            task_factory_func=_create_task,
            continue_on_fail=continue_on_fail,
            return_artifacts=function_return,
            monitor_artifacts=monitor_artifacts,
            monitor_metrics=monitor_metrics,
            monitor_models=monitor_models,
        )

        if self._task and not self._task.running_locally() and not self._mock_execution:
            self.update_execution_plot()

        return True

    def start(
            self,
            queue='services',
            step_task_created_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node, dict], bool]]  # noqa
            step_task_completed_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node], None]]  # noqa
            wait=True,
    ):
        # type: (...) -> bool
        """
        Start the current pipeline remotely (on the selected services queue)
        The current process will be stopped if exit_process is True.

        :param queue: queue name to launch the pipeline on
        :param Callable step_task_created_callback: Callback function, called when a step (Task) is created
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            `parameters` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. `${step1.parameters.Args/param}` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param Callable step_task_completed_callback: Callback function, called when a step (Task) is completed
            and it other jobs are executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass
        :param wait: If True (default), start the pipeline controller, return only
        after the pipeline is done (completed/aborted/failed)

        :return: True, if the controller started. False, if the controller did not start.

        """
        if not self._task:
            raise ValueError(
                "Could not find main Task, "
                "PipelineController must be created with `always_create_task=True`")

        # serialize state only if we are running locally
        if Task.running_locally() or not self._task.is_main_task():
            self._verify()
            self._serialize_pipeline_task()
            self.update_execution_plot()

        # stop current Task and execute remotely or no-op
        self._task.execute_remotely(queue_name=queue, exit_process=True, clone=False)

        if not Task.running_locally() and self._task.is_main_task():
            self._start(
                step_task_created_callback=step_task_created_callback,
                step_task_completed_callback=step_task_completed_callback,
                wait=wait
            )
            leave_process(0)

        return True

    def start_locally(self, run_pipeline_steps_locally=False):
        # type: (bool) -> None
        """
        Start the current pipeline locally, meaning the pipeline logic is running on the current machine,
        instead of on the `services` queue.

        Using run_pipeline_steps_locally=True you can run all the pipeline steps locally as sub-processes.
        Notice: when running pipeline steps locally, it assumes local code execution
        (i.e. it is running the local code as is, regardless of the git commit/diff on the pipeline steps Task)

        :param run_pipeline_steps_locally: (default False) If True, run the
        pipeline steps themselves locally as a subprocess (use for debugging the pipeline locally,
        notice the pipeline code is expected to be available on the local machine)
        """
        if not self._task:
            raise ValueError(
                "Could not find main Task, "
                "PipelineController must be created with `always_create_task=True`")

        if run_pipeline_steps_locally:
            self._clearml_job_class = LocalClearmlJob
            self._default_execution_queue = self._default_execution_queue or 'mock'

        # serialize state only if we are running locally
        self._verify()
        self._serialize_pipeline_task()
        self.update_execution_plot()

        self._start(wait=True)

    def create_draft(self):
        # type: () -> None
        """
        Optional, manually create & serialize the Pipeline Task.
        After calling Pipeline.create(), users can edit the pipeline in the UI and enqueue it for execution.

        Notice: this function should be used to programmatically create pipeline for later usage.
        To automatically create and launch pipelines, call the `start()` method.
        """
        self._verify()
        self._serialize_pipeline_task()
        self._task.close()
        self._task.reset()

    @classmethod
    def get_logger(cls):
        # type: () -> Logger
        """
        Return a logger connected to the Pipeline Task.
        The logger can be used by any function/tasks executed by the pipeline, in order to report
        directly to the pipeline Task itself. It can also be called from the main pipeline control Task.

        Raise ValueError if main Pipeline task could not be located.

        :return: Logger object for reporting metrics (scalars, plots, debug samples etc.)
        """
        return cls._get_pipeline_task().get_logger()

    @classmethod
    def upload_artifact(
        cls,
        name,  # type: str
        artifact_object,  # type: Any
        metadata=None,  # type: Optional[Mapping]
        delete_after_upload=False,  # type: bool
        auto_pickle=True,  # type: bool
        preview=None,  # type: Any
        wait_on_upload=False,  # type: bool
    ):
        # type: (...) -> bool
        """
        Upload (add) an artifact to the main Pipeline Task object.

        The artifact can be uploaded by any function/tasks executed by the pipeline, in order to report
        directly to the pipeline Task itself. It can also be called from the main pipeline control Task.

        Raise ValueError if main Pipeline task could not be located.

        The currently supported upload artifact types include:
        - string / Path - A path to artifact file. If a wildcard or a folder is specified, then ClearML
          creates and uploads a ZIP file.
        - dict - ClearML stores a dictionary as ``.json`` file and uploads it.
        - pandas.DataFrame - ClearML stores a pandas.DataFrame as ``.csv.gz`` (compressed CSV) file and uploads it.
        - numpy.ndarray - ClearML stores a numpy.ndarray as ``.npz`` file and uploads it.
        - PIL.Image - ClearML stores a PIL.Image as ``.png`` file and uploads it.
        - Any - If called with auto_pickle=True, the object will be pickled and uploaded.

        :param str name: The artifact name.

            .. warning::
               If an artifact with the same name was previously uploaded, then it is overwritten.

        :param object artifact_object:  The artifact object.
        :param dict metadata: A dictionary of key-value pairs for any metadata. This dictionary appears with the
            experiment in the **ClearML Web-App (UI)**, **ARTIFACTS** tab.
        :param bool delete_after_upload: After the upload, delete the local copy of the artifact

            - ``True`` - Delete the local copy of the artifact.
            - ``False`` - Do not delete. (default)

        :param bool auto_pickle: If True (default) and the artifact_object is not one of the following types:
            pathlib2.Path, dict, pandas.DataFrame, numpy.ndarray, PIL.Image, url (string), local_file (string)
            the artifact_object will be pickled and uploaded as pickle file artifact (with file extension .pkl)

        :param Any preview: The artifact preview

        :param bool wait_on_upload: Whether or not the upload should be synchronous, forcing the upload to complete
            before continuing.

        :return: The status of the upload.

        - ``True`` - Upload succeeded.
        - ``False`` - Upload failed.

        :raise: If the artifact object type is not supported, raise a ``ValueError``.
        """
        task = cls._get_pipeline_task()
        return task.upload_artifact(
            name=name, artifact_object=artifact_object, metadata=metadata, delete_after_upload=delete_after_upload,
            auto_pickle=auto_pickle, preview=preview, wait_on_upload=wait_on_upload)

    def stop(self, timeout=None, mark_failed=False, mark_aborted=False):
        # type: (Optional[float], bool, bool) -> ()
        """
        Stop the pipeline controller and the optimization thread.
        If mark_failed and mark_aborted are False (default) mark the pipeline as completed,
        unless one of the steps failed, then mark the pipeline as failed

        :param timeout: Wait timeout for the optimization thread to exit (minutes).
            The default is ``None``, indicating do not wait terminate immediately.
        :param mark_failed: If True, mark the pipeline task as failed. (default False)
        :param mark_aborted: If False, mark the pipeline task as aborted. (default False)
        """
        self._stop_event.set()

        self.wait(timeout=timeout)
        if not self._task:
            return

        self._task.close()
        if mark_failed:
            self._task.mark_failed(status_reason='Pipeline aborted and failed', force=True)
        elif mark_aborted:
            self._task.mark_aborted(status_reason='Pipeline aborted', force=True)
        elif self._pipeline_task_status_failed:
            print('Setting pipeline controller Task as failed (due to failed steps) !')
            self._task.mark_failed(status_reason='Pipeline step failed', force=True)

    def wait(self, timeout=None):
        # type: (Optional[float]) -> bool
        """
        Wait for the pipeline to finish.

        .. note::
            This method does not stop the pipeline. Call :meth:`stop` to terminate the pipeline.

        :param float timeout: The timeout to wait for the pipeline to complete (minutes).
            If ``None``, then wait until we reached the timeout, or pipeline completed.

        :return: True, if the pipeline finished. False, if the pipeline timed out.

        """
        if not self.is_running():
            return True

        if timeout is not None:
            timeout *= 60.

        _thread = self._thread

        _thread.join(timeout=timeout)
        if _thread.is_alive():
            return False

        return True

    def is_running(self):
        # type: () -> bool
        """
        return True if the pipeline controller is running.

        :return: A boolean indicating whether the pipeline controller is active (still running) or stopped.
        """
        return self._thread is not None and self._thread.is_alive()

    def is_successful(self):
        # type: () -> bool
        """
        return True if the pipeline controller is fully executed and none of the steps / Tasks failed

        :return: A boolean indicating whether all steps did not fail
        """
        return self._thread and not self.is_running() and not self._pipeline_task_status_failed

    def elapsed(self):
        # type: () -> float
        """
        Return minutes elapsed from controller stating time stamp.

        :return: The minutes from controller start time. A negative value means the process has not started yet.
        """
        if self._start_time is None:
            return -1.0
        return (time() - self._start_time) / 60.

    def get_pipeline_dag(self):
        # type: () -> Mapping[str, PipelineController.Node]
        """
        Return the pipeline execution graph, each node in the DAG is PipelineController.Node object.
        Graph itself is a dictionary of Nodes (key based on the Node name),
        each node holds links to its parent Nodes (identified by their unique names)

        :return: execution tree, as a nested dictionary. Example:

        .. code-block:: py

            {
                'stage1' : Node() {
                    name: 'stage1'
                    job: ClearmlJob
                    ...
                },
            }

        """
        return self._nodes

    def get_processed_nodes(self):
        # type: () -> Sequence[PipelineController.Node]
        """
        Return the a list of the processed pipeline nodes, each entry in the list is PipelineController.Node object.

        :return: executed (excluding currently executing) nodes list
        """
        return {k: n for k, n in self._nodes.items() if n.executed}

    def get_running_nodes(self):
        # type: () -> Sequence[PipelineController.Node]
        """
        Return the a list of the currently running pipeline nodes,
        each entry in the list is PipelineController.Node object.

        :return: Currently running nodes list
        """
        return {k: n for k, n in self._nodes.items() if k in self._running_nodes}

    def update_execution_plot(self):
        # type: () -> ()
        """
        Update sankey diagram of the current pipeline
        """
        with self._reporting_lock:
            self._update_execution_plot()
        # also trigger node monitor scanning
        self._scan_monitored_nodes()

    def add_parameter(self, name, default=None, description=None):
        # type: (str, Optional[Any], Optional[str]) -> None
        """
        Add a parameter to the pipeline Task.
        The parameter can be used as input parameter for any step in the pipeline.
        Notice all parameters will appear under the PipelineController Task's Hyper-parameters -> Pipeline section
        Example: pipeline.add_parameter(name='dataset', description='dataset ID to process the pipeline')
        Then in one of the steps we can refer to the value of the parameter with '${pipeline.dataset}'

        :param name: String name of the parameter.
        :param default: Default value to be put as the default value (can be later changed in the UI)
        :param description: String description of the parameter and its usage in the pipeline
        """
        self._pipeline_args[str(name)] = str(default or '')
        if description:
            self._pipeline_args_desc[str(name)] = str(description)

    def get_parameters(self):
        # type: () -> dict
        """
        Return the pipeline parameters dictionary
        :return: Dictionary str -> str
        """
        return self._pipeline_args

    def _create_task_from_function(
            self, docker, docker_args, docker_bash_setup_script,
            function, function_input_artifacts, function_kwargs, function_return,
            packages, project_name, task_name, task_type, repo, branch, commit, helper_functions
    ):
        task_definition = CreateFromFunction.create_task_from_function(
            a_function=function,
            function_kwargs=function_kwargs or None,
            function_input_artifacts=function_input_artifacts,
            function_return=function_return,
            project_name=project_name,
            task_name=task_name,
            task_type=task_type,
            repo=repo,
            branch=branch,
            commit=commit,
            packages=packages,
            docker=docker,
            docker_args=docker_args,
            docker_bash_setup_script=docker_bash_setup_script,
            output_uri=None,
            helper_functions=helper_functions,
            dry_run=True,
        )
        return task_definition

    def _start(
            self,
            step_task_created_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node, dict], bool]]  # noqa
            step_task_completed_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node], None]]  # noqa
            wait=True,
    ):
        # type: (...) -> bool
        """
        Start the pipeline controller.
        If the calling process is stopped, then the controller stops as well.

        :param Callable step_task_created_callback: Callback function, called when a step (Task) is created
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            `parameters` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. `${step1.parameters.Args/param}` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param Callable step_task_completed_callback: Callback function, called when a step (Task) is completed
            and it other jobs are executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass
        :param wait: If True (default), start the pipeline controller, return only
        after the pipeline is done (completed/aborted/failed)

        :return: True, if the controller started. False, if the controller did not start.

        """
        if self._thread:
            return True

        self._prepare_pipeline(step_task_completed_callback, step_task_created_callback)
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()

        if wait:
            self.wait()
            self.stop()

        return True

    def _prepare_pipeline(
            self,
            step_task_created_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node, dict], bool]]  # noqa
            step_task_completed_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node], None]]  # noqa
    ):
        # type (...) -> None

        params, pipeline_dag = self._serialize_pipeline_task()
        # deserialize back pipeline state
        if not params['_continue_pipeline_']:
            for k in pipeline_dag:
                pipeline_dag[k]['executed'] = None
        self._default_execution_queue = params['_default_queue_']
        self._add_pipeline_tags = params['_add_pipeline_tags_']
        self._target_project = params['_target_project_'] or ''
        self._deserialize(pipeline_dag)
        # if we continue the pipeline, make sure that we re-execute failed tasks
        if params['_continue_pipeline_']:
            for node in list(self._nodes.values()):
                if node.executed is False:
                    node.executed = None
        if not self._verify():
            raise ValueError("Failed verifying pipeline execution graph, "
                             "it has either inaccessible nodes, or contains cycles")
        self.update_execution_plot()
        self._start_time = time()
        self._stop_event = Event()
        self._experiment_created_cb = step_task_created_callback
        self._experiment_completed_cb = step_task_completed_callback

    def _serialize_pipeline_task(self):
        # type: () -> (dict, dict)
        """
        Serialize current pipeline state into the main Task

        :return: params, pipeline_dag
        """
        params = {
            '_default_queue_': self._default_execution_queue,
            '_add_pipeline_tags_': self._add_pipeline_tags,
            '_target_project_': self._target_project,
        }
        pipeline_dag = self._serialize()

        # serialize pipeline state
        if self._task and self._auto_connect_task:
            if self._task.running_locally():
                # noinspection PyProtectedMember
                self._task._set_configuration(
                    name=self._config_section, config_type='dictionary',
                    config_text=json.dumps(pipeline_dag, indent=2))
                params.update(self._pipeline_args)
                # noinspection PyProtectedMember
                self._task._set_parameters(
                    {'{}/{}'.format(self._args_section, k): str(v) for k, v in params.items()},
                    __parameters_descriptions=self._pipeline_args_desc,
                    __update=True,
                )
                params['_continue_pipeline_'] = False

                # make sure we have a unique version number (auto bump version if needed)
                # only needed when manually (from code) creating pipelines
                self._verify_pipeline_version()

                # noinspection PyProtectedMember
                pipeline_hash = self._get_task_hash()

                # noinspection PyProtectedMember
                self._task._set_runtime_properties({
                    self._runtime_property_hash: "{}:{}".format(pipeline_hash, self._version),
                })
            else:
                self._task.connect_configuration(pipeline_dag, name=self._config_section)
                self._task.connect(self._pipeline_args, name=self._args_section)
                self._task.connect(params, name=self._args_section)
                # noinspection PyProtectedMember
                if self._task._get_runtime_properties().get(self._runtime_property_hash):
                    params['_continue_pipeline_'] = True
                else:
                    # noinspection PyProtectedMember
                    pipeline_hash = ClearmlJob._create_task_hash(self._task)
                    # noinspection PyProtectedMember
                    self._task._set_runtime_properties({
                        self._runtime_property_hash: "{}:{}".format(pipeline_hash, self._version),
                    })
                    params['_continue_pipeline_'] = False

        return params, pipeline_dag

    def _verify_pipeline_version(self):
        # if no version bump needed, just set the property
        if not self._auto_version_bump:
            self._task.set_user_properties(version=self._version)
            return

        # check if pipeline version exists, if it does increase version
        pipeline_hash = self._get_task_hash()
        # noinspection PyProtectedMember
        existing_tasks = Task._query_tasks(
            project=[self._task.project], task_name=exact_match_regex(self._task.name),
            type=[str(self._task.task_type)],
            system_tags=['-{}'.format(Task.archived_tag), self._tag],
            _all_=dict(fields=['runtime.{}'.format(self._runtime_property_hash)],
                       pattern=":{}".format(self._version)),
            only_fields=['id', 'runtime'],
        )
        if existing_tasks:
            # check if hash match the current version.
            matched = True
            for t in existing_tasks:
                h, _, v = t.runtime.get(self._runtime_property_hash, '').partition(':')
                if v == self._version:
                    matched = bool(h == pipeline_hash)
                    break
            # if hash did not match, look for the highest version
            if not matched:
                # noinspection PyProtectedMember
                existing_tasks = Task._query_tasks(
                    project=[self._task.project], task_name=exact_match_regex(self._task.name),
                    type=[str(self._task.task_type)],
                    system_tags=['-{}'.format(Task.archived_tag), self._tag],
                    only_fields=['id', 'hyperparams', 'runtime'],
                )
                found_match_version = False
                existing_versions = set([self._version])  # noqa
                for t in existing_tasks:
                    if not t.hyperparams:
                        continue
                    v = t.hyperparams.get('properties', {}).get('version')
                    if v:
                        existing_versions.add(v.value)
                    if t.runtime:
                        h, _, _ = t.runtime.get(self._runtime_property_hash, '').partition(':')
                        if h == pipeline_hash:
                            self._version = v.value
                            found_match_version = True
                            break

                # match to the version we found:
                if found_match_version:
                    getLogger('clearml.automation.controller').info(
                        'Existing Pipeline found, matching version to: {}'.format(self._version))
                else:
                    # if we did not find a matched pipeline version, get the max one and bump the version by 1
                    while True:
                        v = self._version.split('.')
                        self._version = '.'.join(v[:-1] + [str(int(v[-1]) + 1)])
                        if self._version not in existing_versions:
                            break

                    getLogger('clearml.automation.controller').info(
                        'Existing Pipeline version found, bump new version to: {}'.format(self._version))

            self._task.set_user_properties(version=self._version)

    def _get_task_hash(self):
        params_override = dict(**(self._task.get_parameters() or {}))
        params_override.pop('properties/version', None)
        # noinspection PyProtectedMember
        pipeline_hash = ClearmlJob._create_task_hash(self._task, params_override=params_override)
        return pipeline_hash

    def _serialize(self):
        # type: () -> dict
        """
        Store the definition of the pipeline DAG into a dictionary.
        This dictionary will be used to store the DAG as a configuration on the Task
        :return:
        """
        dag = {name: dict((k, v) for k, v in node.__dict__.items()
                          if k not in ('job', 'name', 'task_factory_func'))
               for name, node in list(self._nodes.items())}

        return dag

    def _deserialize(self, dag_dict):
        # type: (dict) -> ()
        """
        Restore the DAG from a dictionary.
        This will be used to create the DAG from the dict stored on the Task, when running remotely.
        :return:
        """

        # if we do not clone the Task, only merge the parts we can override.
        for name in list(self._nodes.keys()):
            if not self._nodes[name].clone_task and name in dag_dict and not dag_dict[name].get('clone_task'):
                for k in ('queue', 'parents', 'timeout', 'parameters', 'configurations', 'task_overrides'):
                    setattr(self._nodes[name], k, dag_dict[name].get(k) or type(getattr(self._nodes[name], k))())

        # if we do clone the Task deserialize everything, except the function creating
        self._nodes = {
            k: self.Node(name=k, **v)
            if k not in self._nodes or (v.get('base_task_id') and v.get('clone_task'))
            else self._nodes[k]
            for k, v in dag_dict.items()}

    def _has_stored_configuration(self):
        """
        Return True if we are running remotely and we have stored configuration on the Task
        """
        if self._auto_connect_task and self._task and not self._task.running_locally() and self._task.is_main_task():
            stored_config = self._task.get_configuration_object(self._config_section)
            return bool(stored_config)

        return False

    def _verify(self):
        # type: () -> bool
        """
        Verify the DAG, (i.e. no cycles and no missing parents)
        On error raise ValueError with verification details

        :return: return True iff DAG has no errors
        """
        # verify nodes
        for node in list(self._nodes.values()):
            # raise value error if not verified
            self._verify_node(node)

        # check the dag itself
        if not self._verify_dag():
            return False

        return True

    def _verify_node(self, node):
        # type: (PipelineController.Node) -> bool
        """
        Raise ValueError on verification errors

        :return: Return True iff the specific node is verified
        """
        if not node.base_task_id and not node.task_factory_func:
            raise ValueError("Node '{}', base_task_id is empty".format(node.name))

        if not self._default_execution_queue and not node.queue:
            raise ValueError("Node '{}' missing execution queue, "
                             "no default queue defined and no specific node queue defined".format(node.name))

        task = node.task_factory_func or Task.get_task(task_id=node.base_task_id)
        if not task:
            raise ValueError("Node '{}', base_task_id={} is invalid".format(node.name, node.base_task_id))

        pattern = self._step_ref_pattern

        # verify original node parents
        if node.parents and not all(isinstance(p, str) and p in self._nodes for p in node.parents):
            raise ValueError("Node '{}', parents={} is invalid".format(node.name, node.parents))

        parents = set()
        for k, v in node.parameters.items():
            if isinstance(v, str):
                for g in pattern.findall(v):
                    ref_step = self.__verify_step_reference(node, g)
                    if ref_step:
                        parents.add(ref_step)
            # verify we have a section name
            if '/' not in k:
                raise ValueError(
                    "Section name is missing in parameter \"{}\", "
                    "parameters should be in the form of "
                    "\"`section-name`/parameter\", example: \"Args/param\"".format(v))

        if parents and parents != set(node.parents or []):
            parents = parents - set(node.parents or [])
            getLogger('clearml.automation.controller').info(
                'Node "{}" missing parent reference, adding: {}'.format(node.name, parents))
            node.parents = (node.parents or []) + list(parents)

        # verify and fix monitoring sections:
        def _verify_monitors(monitors, monitor_type, nested_pairs=False):
            if not monitors:
                return monitors

            if nested_pairs:
                if not all(isinstance(x, (list, tuple)) and x for x in monitors):
                    raise ValueError("{} should be a list of tuples, found: {}".format(monitor_type, monitors))
                # convert single pair into a pair of pairs:
                conformed_monitors = [
                    pair if isinstance(pair[0], (list, tuple)) else (pair, pair) for pair in monitors
                ]
                # verify pair of pairs
                if not all(isinstance(x[0][0], str) and isinstance(x[0][1], str) and
                           isinstance(x[1][0], str) and isinstance(x[1][1], str) for x in conformed_monitors):
                    raise ValueError("{} should be a list of tuples, found: {}".format(monitor_type, monitors))
            else:
                # verify a list of tuples
                if not all(isinstance(x, (list, tuple, str)) and x for x in monitors):
                    raise ValueError(
                        "{} should be a list of tuples, found: {}".format(monitor_type, monitors))
                # convert single str into a pair of pairs:
                conformed_monitors = [
                    pair if isinstance(pair, (list, tuple)) else (pair, pair) for pair in monitors
                ]
                # verify pair of pairs
                if not all(isinstance(x[0], str) and isinstance(x[1], str) for x in conformed_monitors):
                    raise ValueError(
                        "{} should be a list of tuples, found: {}".format(monitor_type, monitors))

            return conformed_monitors

        # verify and fix monitoring sections:
        node.monitor_metrics = _verify_monitors(node.monitor_metrics, 'monitor_metrics', nested_pairs=True)
        node.monitor_artifacts = _verify_monitors(node.monitor_artifacts, 'monitor_artifacts')
        node.monitor_models = _verify_monitors(node.monitor_models, 'monitor_models')

        return True

    def _verify_dag(self):
        # type: () -> bool
        """
        :return: True iff the pipeline dag is fully accessible and contains no cycles
        """
        visited = set()
        prev_visited = None
        while prev_visited != visited:
            prev_visited = copy(visited)
            for k, node in list(self._nodes.items()):
                if k in visited:
                    continue
                if any(p == node.name for p in node.parents or []):
                    # node cannot have itself as parent
                    return False
                if not all(p in visited for p in node.parents or []):
                    continue
                visited.add(k)
        # return False if we did not cover all the nodes
        return not bool(set(self._nodes.keys()) - visited)

    def _launch_node(self, node):
        # type: (PipelineController.Node) -> ()
        """
        Launch a single node (create and enqueue a ClearmlJob)

        :param node: Node to launch
        :return: Return True if a new job was launched
        """
        if node.job or node.executed:
            return False

        updated_hyper_parameters = {}
        for k, v in node.parameters.items():
            updated_hyper_parameters[k] = self._parse_step_ref(v)

        task_overrides = self._parse_task_overrides(node.task_overrides) if node.task_overrides else None

        extra_args = dict()
        if self._target_project:
            extra_args['project'] = get_or_create_project(
                session=self._task.session if self._task else Task.default_session,
                project_name=self._target_project)

        skip_node = None
        if self._pre_step_callbacks.get(node.name):
            skip_node = self._pre_step_callbacks[node.name](self, node, updated_hyper_parameters)

        if skip_node is False:
            node.skip_job = True
            return True

        task_id = node.base_task_id
        disable_clone_task = not node.clone_task
        task_factory_func_task = None
        if node.task_factory_func:
            # create Task
            task_factory_func_task = node.task_factory_func(node)
            task_id = task_factory_func_task.id
            disable_clone_task = True

        try:
            node.job = self._clearml_job_class(
                base_task_id=task_id,
                parameter_override=updated_hyper_parameters,
                configuration_overrides=node.configurations,
                tags=['{} {}'.format(self._node_tag_prefix, self._task.id)]
                if self._add_pipeline_tags and self._task else None,
                parent=self._task.id if self._task else None,
                disable_clone_task=disable_clone_task,
                task_overrides=task_overrides,
                allow_caching=node.cache_executed_step,
                **extra_args
            )
        except Exception:
            self._pipeline_task_status_failed = True
            raise

        if self._experiment_created_cb:
            skip_node = self._experiment_created_cb(self, node, updated_hyper_parameters)

        if skip_node is False:
            # skipping node
            getLogger('clearml.automation.controller').warning(
                'Skipping node {} on callback request'.format(node))
            # delete the job we just created
            node.job.delete()
            node.skip_job = True
        elif node.job.is_cached_task():
            node.executed = node.job.task_id()
            if task_factory_func_task:
                task_factory_func_task.delete(raise_on_error=False)
            self._running_nodes.append(node.name)
        else:
            self._running_nodes.append(node.name)
            return node.job.launch(queue_name=node.queue or self._default_execution_queue)

        return True

    def _update_execution_plot(self):
        # type: () -> ()
        """
        Update sankey diagram of the current pipeline
        """
        if not self._task:
            return

        sankey_node = dict(
            label=[],
            color=[],
            hovertemplate='%{label}<extra></extra>',
            # customdata=[],
            # hovertemplate='%{label}<br />Hyper-Parameters:<br />%{customdata}<extra></extra>',
        )
        sankey_link = dict(
            source=[],
            target=[],
            value=[],
            # hovertemplate='%{target.label}<extra></extra>',
            hovertemplate='<extra></extra>',
        )
        visited = []
        node_params = []
        nodes = list(self._nodes.values())
        while nodes:
            next_nodes = []
            for node in nodes:
                if not all(p in visited for p in node.parents or []):
                    next_nodes.append(node)
                    continue
                visited.append(node.name)
                idx = len(visited) - 1
                parents = [visited.index(p) for p in node.parents or []]
                node_params.append(
                    (node.job.task_parameter_override
                     if node.job and node.job.task_parameter_override
                     else node.parameters) or {})
                # sankey_node['label'].append(node.name)
                # sankey_node['customdata'].append(
                #     '<br />'.join('{}: {}'.format(k, v) for k, v in (node.parameters or {}).items()))
                sankey_node['label'].append(
                    '{}<br />'.format(node.name) +
                    '<br />'.join('{}: {}'.format(k, v if len(str(v)) < 24 else (str(v)[:24]+' ...'))
                                  for k, v in (node.parameters or {}).items()))

                sankey_node['color'].append(self._get_node_color(node))

                for p in parents:
                    sankey_link['source'].append(p)
                    sankey_link['target'].append(idx)
                    sankey_link['value'].append(1)

            nodes = next_nodes

        # make sure we have no independent (unconnected) nodes
        single_nodes = []
        for i in [n for n in range(len(visited)) if n not in sankey_link['source'] and n not in sankey_link['target']]:
            single_nodes.append(i)

        # create the sankey graph
        dag_flow = dict(
            link=sankey_link,
            node=sankey_node,
            textfont=dict(color='rgba(0,0,0,0)', size=1),
            type='sankey',
            orientation='h'
        )

        table_values = self._build_table_report(node_params, visited)

        # hack, show single node sankey
        if single_nodes:
            singles_flow = dict(
                x=list(range(len(single_nodes))), y=[1] * len(single_nodes),
                text=[v for i, v in enumerate(sankey_node['label']) if i in single_nodes],
                mode='markers',
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    color=[v for i, v in enumerate(sankey_node['color']) if i in single_nodes],
                    size=[40] * len(single_nodes),
                ),
                showlegend=False,
                type='scatter',
            )
            # only single nodes
            if len(single_nodes) == len(sankey_node['label']):
                fig = dict(data=[singles_flow], layout={
                    'hovermode': 'closest', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
            else:
                dag_flow['domain'] = {'x': [0.0, 1.0], 'y': [0.2, 1.0]}
                fig = dict(data=[dag_flow, singles_flow],
                           layout={'autosize': True,
                                   'hovermode': 'closest',
                                   'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'visible': False},
                                   'yaxis': {'anchor': 'x', 'domain': [0.0, 0.15], 'visible': False}
                                   })
        else:
            # create the sankey plot
            fig = dict(data=[dag_flow], layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

        # report DAG
        self._task.get_logger().report_plotly(
            title='Pipeline', series='Execution Flow', iteration=0, figure=fig)
        # report detailed table
        self._task.get_logger().report_table(
            title='Pipeline Details', series='Execution Details', iteration=0, table_plot=table_values)

    def _build_table_report(self, node_params, visited):
        # type: (List, List) -> List[List]
        """
        Create the detailed table report on all the jobs in the pipeline

        :param node_params: list of node parameters
        :param visited: list of nodes
        :return: Table as List of List of strings (cell)
        """
        task_link_template = self._task.get_output_log_web_page() \
            .replace('/{}/'.format(self._task.project), '/{project}/') \
            .replace('/{}/'.format(self._task.id), '/{task}/')

        table_values = [["Pipeline Step", "Task ID", "Task Name", "Status", "Parameters"]]

        for name, param in zip(visited, node_params):
            param_str = str(param) if param else ''
            if len(param_str) > 3:
                # remove {} from string
                param_str = param_str[1:-1]

            step_name = name
            if self._nodes[name].base_task_id:
                step_name += '\n[<a href="{}"> {} </a>]'.format(
                    task_link_template.format(project='*', task=self._nodes[name].base_task_id), 'base task')

            table_values.append(
                [step_name,
                 self.__create_task_link(self._nodes[name], task_link_template),
                 self._nodes[name].job.task.name if self._nodes[name].job else '',
                 self.__get_node_status(self._nodes[name]),
                 param_str]
            )

        return table_values

    @staticmethod
    def _get_node_color(node):
        # type (self.Mode) -> str
        """
        Return the node color based on the node/job state
        :param node: A node in the pipeline
        :return: string representing the color of the node (e.g. "red", "green", etc)
        """
        if not node:
            return ""

        if node.executed is not None:
            if node.job and node.job.is_failed():
                return "red"  # failed job
            elif node.job and node.job.is_cached_task():
                return "darkslateblue"
            elif not node.job or node.job.is_completed():
                return "blue"  # completed job
            else:
                return "royalblue"  # aborted job
        elif node.job:
            if node.job.is_pending():
                return "#bdf5bd"  # lightgreen, pending in queue
            elif node.job.is_completed():
                return "blue"  # completed job
            elif node.job.is_failed():
                return "red"  # failed job
            elif node.job.is_stopped():
                return "royalblue"  # aborted job
            else:
                return "green"  # running job
        elif node.skip_job:
            return "gray"  # skipped job
        else:
            return "lightsteelblue"  # pending job

    def _force_task_configuration_update(self):
        pipeline_dag = self._serialize()
        if self._task:
            # noinspection PyProtectedMember
            self._task._set_configuration(
                name=self._config_section, config_type='dictionary',
                config_text=json.dumps(pipeline_dag, indent=2))

    def _daemon(self):
        # type: () -> ()
        """
        The main pipeline execution loop. This loop is executed on its own dedicated thread.
        :return:
        """
        pooling_counter = 0
        launched_nodes = set()
        last_monitor_report = last_plot_report = time()
        while self._stop_event:
            # stop request
            if self._stop_event.wait(self._pool_frequency if pooling_counter else 0.01):
                break

            pooling_counter += 1

            # check the pipeline time limit
            if self._pipeline_time_limit and (time() - self._start_time) > self._pipeline_time_limit:
                break

            # check the state of all current jobs
            # if no a job ended, continue
            completed_jobs = []
            force_execution_plot_update = False
            nodes_failed_stop_pipeline = []
            for j in self._running_nodes:
                node = self._nodes[j]
                if not node.job:
                    continue
                if node.job.is_stopped():
                    completed_jobs.append(j)
                    node_failed = node.job.is_failed()
                    node.executed = node.job.task_id() if not node_failed else False
                    if j in launched_nodes:
                        launched_nodes.remove(j)
                    # check if we need to stop all running steps
                    if node_failed and self._abort_running_steps_on_failure and not node.continue_on_fail:
                        nodes_failed_stop_pipeline.append(node.name)
                elif node.timeout:
                    started = node.job.task.data.started
                    if (datetime.now().astimezone(started.tzinfo) - started).total_seconds() > node.timeout:
                        node.job.abort()
                        completed_jobs.append(j)
                        node.executed = node.job.task_id()
                elif j in launched_nodes and node.job.is_running():
                    # make sure update the execution graph when the job started running
                    # (otherwise it will still be marked queued)
                    launched_nodes.remove(j)
                    force_execution_plot_update = True

            # update running jobs
            self._running_nodes = [j for j in self._running_nodes if j not in completed_jobs]

            # nothing changed, we can sleep
            if not completed_jobs and self._running_nodes:
                # force updating the pipeline state (plot) at least every 5 min.
                if force_execution_plot_update or time()-last_plot_report > self._update_execution_plot_interval:
                    last_plot_report = time()
                    last_monitor_report = time()
                    self.update_execution_plot()
                elif time()-last_monitor_report > self._monitor_node_interval:
                    last_monitor_report = time()
                    self._scan_monitored_nodes()
                continue

            # callback on completed jobs
            if self._experiment_completed_cb or self._post_step_callbacks:
                for job in completed_jobs:
                    job_node = self._nodes.get(job)
                    if not job_node:
                        continue
                    if self._experiment_completed_cb:
                        self._experiment_completed_cb(self, job_node)
                    if self._post_step_callbacks.get(job_node.name):
                        self._post_step_callbacks[job_node.name](self, job_node)

            # check if we need to stop the pipeline, and abort all running steps
            if nodes_failed_stop_pipeline:
                print('Aborting pipeline and stopping all running steps, node {} failed'.format(
                    nodes_failed_stop_pipeline))
                break

            # Pull the next jobs in the pipeline, based on the completed list
            next_nodes = []
            for node in list(self._nodes.values()):
                # check if already processed or needs to be skipped
                if node.job or node.executed or node.skip_job:
                    continue
                completed_parents = [bool(p in self._nodes and self._nodes[p].executed) for p in node.parents or []]
                if all(completed_parents):
                    next_nodes.append(node.name)

            # update the execution graph
            for name in next_nodes:
                if self._launch_node(self._nodes[name]) and not self._nodes[name].skip_job:
                    print('Launching step: {}'.format(name))
                    print('Parameters:\n{}'.format(
                        self._nodes[name].job.task_parameter_override if self._nodes[name].job
                        else self._nodes[name].parameters))
                    print('Configurations:\n{}'.format(self._nodes[name].configurations))
                    print('Overrides:\n{}'.format(self._nodes[name].task_overrides))
                    launched_nodes.add(name)
                    # check if node is cached do not wait for event but run the loop again
                    if self._nodes[name].executed:
                        pooling_counter = 0
                else:
                    getLogger('clearml.automation.controller').warning(
                        'Skipping launching step \'{}\': {}'.format(name, self._nodes[name]))

            # update current state (in configuration, so that we could later continue an aborted pipeline)
            self._force_task_configuration_update()

            # visualize pipeline state (plot)
            self.update_execution_plot()

            # quit if all pipelines nodes are fully executed.
            if not next_nodes and not self._running_nodes:
                break

        # stop all currently running jobs:
        for node in list(self._nodes.values()):
            if node.executed is False and not node.continue_on_fail:
                self._pipeline_task_status_failed = True

            if node.job and not node.job.is_stopped():
                node.job.abort()
            elif not node.job and not node.executed:
                # mark Node as skipped if it has no Job object and it is not executed
                node.skip_job = True

        # visualize pipeline state (plot)
        self.update_execution_plot()

        if self._stop_event:
            # noinspection PyBroadException
            try:
                self._stop_event.set()
            except Exception:
                pass

    def _parse_step_ref(self, value):
        # type: (Any) -> Optional[str]
        """
        Return the step reference. For example "${step1.parameters.Args/param}"
        :param value: string
        :return:
        """
        # look for all the step references
        pattern = self._step_ref_pattern
        updated_value = value
        if isinstance(value, str):
            for g in pattern.findall(value):
                # update with actual value
                new_val = self.__parse_step_reference(g)
                updated_value = updated_value.replace(g, new_val, 1)
        return updated_value

    def _parse_task_overrides(self, task_overrides):
        # type: (dict) -> dict
        """
        Return the step reference. For example "${step1.parameters.Args/param}"
        :param task_overrides: string
        :return:
        """
        updated_overrides = {}
        for k, v in task_overrides.items():
            updated_overrides[k] = self._parse_step_ref(v)

        return updated_overrides

    def _verify_node_name(self, name):
        # type: (str) -> None
        if name in self._nodes:
            raise ValueError('Node named \'{}\' already exists in the pipeline dag'.format(name))
        if name in self._reserved_pipeline_names:
            raise ValueError('Node named \'{}\' is a reserved keyword, use a different name'.format(name))

    def _scan_monitored_nodes(self):
        # type: () -> None
        """
        Scan all nodes and monitor their metrics/artifacts/models
        """
        for node in list(self._nodes.values()):
            self._monitor_node(node)

    def _monitor_node(self, node):
        # type: (PipelineController.Node) -> None
        """
        If Node is running, put the metrics from the node on the pipeline itself.
        :param node: Node to test
        """
        if not node:
            return

        # verify we have the node
        if node.name not in self._monitored_nodes:
            self._monitored_nodes[node.name] = {}

        # if we are done with this node, skip it
        if self._monitored_nodes[node.name].get('completed'):
            return

        if node.job and node.job.task:
            task = node.job.task
        elif node.job and node.executed and isinstance(node.executed, str):
            task = Task.get_task(task_id=node.executed)
        else:
            return

        # update the metrics
        if node.monitor_metrics:
            metrics_state = self._monitored_nodes[node.name].get('metrics', {})
            logger = self._task.get_logger()
            scalars = task.get_reported_scalars(x_axis='iter')
            for (s_title, s_series), (t_title, t_series) in node.monitor_metrics:
                values = scalars.get(s_title, {}).get(s_series)
                if values and values.get('x') is not None and values.get('y') is not None:
                    x = values['x'][-1]
                    y = values['y'][-1]
                    last_y = metrics_state.get(s_title, {}).get(s_series)
                    if last_y is None or y > last_y:
                        logger.report_scalar(title=t_title, series=t_series, value=y, iteration=int(x))
                        last_y = y
                    if not metrics_state.get(s_title):
                        metrics_state[s_title] = {}
                    metrics_state[s_title][s_series] = last_y

            self._monitored_nodes[node.name]['metrics'] = metrics_state

        if node.monitor_artifacts:
            task.reload()
            artifacts = task.data.execution.artifacts
            self._task.reload()
            output_artifacts = []
            for s_artifact, t_artifact in node.monitor_artifacts:
                # find artifact
                for a in artifacts:
                    if a.key != s_artifact:
                        continue

                    new_a = copy(a)
                    new_a.key = t_artifact
                    output_artifacts.append(new_a)
                    break

            # update artifacts directly on the Task
            if output_artifacts:
                # noinspection PyProtectedMember
                self._task._add_artifacts(output_artifacts)

        if node.monitor_models:
            task.reload()
            output_models = task.data.models.output
            self._task.reload()
            target_models = []
            for s_model, t_model in node.monitor_models:
                # find artifact
                for a in output_models:
                    if a.name != s_model:
                        continue

                    new_a = copy(a)
                    new_a.name = t_model
                    target_models.append(new_a)
                    break

            # update artifacts directly on the Task
            if target_models:
                self._task.reload()
                models = self._task.data.models
                keys = [a.name for a in models.output]
                models.output = [a for a in models.output or [] if a.name not in keys] + target_models
                # noinspection PyProtectedMember
                self._task._edit(models=models)

        # update the state (so that we do not scan the node twice)
        if node.job.is_stopped():
            self._monitored_nodes[node.name]['completed'] = True

    @classmethod
    def _get_pipeline_task(cls):
        # type: () -> Task
        """
        Return the pipeline Task (either the current one, or the parent Task of the currently running Task)
        Raise ValueError if we could not locate the pipeline Task

        :return: Pipeline Task
        """
        # get main Task.
        task = Task.current_task()
        if str(task.task_type) == str(Task.TaskTypes.controller) and cls._tag in task.get_system_tags():
            return task
        # get the parent Task, it should be the pipeline
        if not task.parent:
            raise ValueError("Could not locate parent Pipeline Task")
        parent = Task.get_task(task_id=task.parent)
        if str(parent.task_type) == str(Task.TaskTypes.controller) and cls._tag in parent.get_system_tags():
            return parent
        raise ValueError("Could not locate parent Pipeline Task")

    def __verify_step_reference(self, node, step_ref_string):
        # type: (PipelineController.Node, str) -> Optional[str]
        """
        Verify the step reference. For example "${step1.parameters.Args/param}"
        Raise ValueError on misconfiguration

        :param Node node: calling reference node (used for logging)
        :param str step_ref_string: For example "${step1.parameters.Args/param}"
        :return: If step reference is used, return the pipeline step name, otherwise return None
        """
        parts = step_ref_string[2:-1].split('.')
        v = step_ref_string
        if len(parts) < 2:
            raise ValueError("Node '{}', parameter '{}' is invalid".format(node.name, v))
        prev_step = parts[0]
        input_type = parts[1]

        # check if we reference the pipeline arguments themselves
        if prev_step == self._pipeline_step_ref:
            if input_type not in self._pipeline_args:
                raise ValueError("Node '{}', parameter '{}', step name '{}' is invalid".format(node.name, v, prev_step))
            return None

        if prev_step not in self._nodes:
            raise ValueError("Node '{}', parameter '{}', step name '{}' is invalid".format(node.name, v, prev_step))
        if input_type not in ('artifacts', 'parameters', 'models', 'id'):
            raise ValueError(
                "Node {}, parameter '{}', input type '{}' is invalid".format(node.name, v, input_type))

        if input_type != 'id' and len(parts) < 3:
            raise ValueError("Node '{}', parameter '{}' is invalid".format(node.name, v))

        if input_type == 'models':
            try:
                model_type = parts[2].lower()
            except Exception:
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model_type is missing {}".format(
                        node.name, v, input_type, parts))
            if model_type not in ('input', 'output'):
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', "
                    "model_type is invalid (input/output) found {}".format(
                        node.name, v, input_type, model_type))

            if len(parts) < 4:
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model index is missing".format(
                        node.name, v, input_type))

            # check casting
            try:
                int(parts[3])
            except Exception:
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model index is missing {}".format(
                        node.name, v, input_type, parts))

            if len(parts) < 5:
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model property is missing".format(
                        node.name, v, input_type))

            if not hasattr(BaseModel, parts[4]):
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model property is invalid {}".format(
                        node.name, v, input_type, parts[4]))
        return prev_step

    def __parse_step_reference(self, step_ref_string):
        """
        return the adjusted value for "${step...}"
        :param step_ref_string: reference string of the form ${step_name.type.value}"
        :return: str with value
        """
        parts = step_ref_string[2:-1].split('.')
        if len(parts) < 2:
            raise ValueError("Could not parse reference '{}'".format(step_ref_string))
        prev_step = parts[0]
        input_type = parts[1].lower()

        # check if we reference the pipeline arguments themselves
        if prev_step == self._pipeline_step_ref:
            if parts[1] not in self._pipeline_args:
                raise ValueError("Could not parse reference '{}', "
                                 "pipeline argument '{}' could not be found".format(step_ref_string, parts[1]))
            return self._pipeline_args[parts[1]]

        if prev_step not in self._nodes or (
                not self._nodes[prev_step].job and
                not self._nodes[prev_step].executed and
                not self._nodes[prev_step].base_task_id
        ):
            raise ValueError("Could not parse reference '{}', step '{}' could not be found".format(
                step_ref_string, prev_step))

        if input_type not in (
                'artifacts', 'parameters', 'models', 'id',
                'script', 'execution', 'container', 'output',
                'comment', 'models', 'tags', 'system_tags', 'project'):
            raise ValueError("Could not parse reference '{}', type '{}' not valid".format(step_ref_string, input_type))
        if input_type != 'id' and len(parts) < 3:
            raise ValueError("Could not parse reference '{}', missing fields in '{}'".format(step_ref_string, parts))

        task = self._nodes[prev_step].job.task if self._nodes[prev_step].job \
            else Task.get_task(task_id=self._nodes[prev_step].executed or self._nodes[prev_step].base_task_id)
        task.reload()
        if input_type == 'artifacts':
            # fix \. to use . in artifacts
            artifact_path = ('.'.join(parts[2:])).replace('\\.', '\\_dot_\\')
            artifact_path = artifact_path.split('.')

            obj = task.artifacts
            for p in artifact_path:
                p = p.replace('\\_dot_\\', '.')
                if isinstance(obj, dict):
                    obj = obj.get(p)
                elif hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    raise ValueError("Could not locate artifact {} on previous step {}".format(
                        '.'.join(parts[1:]), prev_step))
            return str(obj)
        elif input_type == 'parameters':
            step_params = task.get_parameters()
            param_name = '.'.join(parts[2:])
            if param_name not in step_params:
                raise ValueError("Could not locate parameter {} on previous step {}".format(
                    '.'.join(parts[1:]), prev_step))
            return step_params.get(param_name)
        elif input_type == 'models':
            model_type = parts[2].lower()
            if model_type not in ('input', 'output'):
                raise ValueError("Could not locate model {} on previous step {}".format(
                    '.'.join(parts[1:]), prev_step))
            try:
                model_idx = int(parts[3])
                model = task.models[model_type][model_idx]
            except Exception:
                raise ValueError("Could not locate model {} on previous step {}, index {} is invalid".format(
                    '.'.join(parts[1:]), prev_step, parts[3]))

            return str(getattr(model, parts[4]))
        elif input_type == 'id':
            return task.id
        elif input_type in (
                'script', 'execution', 'container', 'output',
                'comment', 'models', 'tags', 'system_tags', 'project'):
            # noinspection PyProtectedMember
            return task._get_task_property('.'.join(parts[1:]))

        return None

    @classmethod
    def __get_node_status(cls, a_node):
        # type: (PipelineController.Node) -> str
        if not a_node:
            return "pending"
        if a_node.skip_job:
            return "skipped"
        if a_node.job and a_node.job.is_cached_task():
            return "cached"
        if a_node.job and a_node.job.task:
            # no need to refresh status
            return str(a_node.job.task.data.status)
        if a_node.executed:
            return "executed"
        return "pending"

    @classmethod
    def __create_task_link(cls, a_node, task_link_template):
        # type: (PipelineController.Node, str) -> str
        if not a_node:
            return ''
        # create the detailed parameter table
        task_id = project_id = None
        if a_node.job:
            project_id = a_node.job.task.project
            task_id = a_node.job.task.id
        elif a_node.executed:
            task_id = a_node.executed
            if cls._task_project_lookup.get(task_id):
                project_id = cls._task_project_lookup[task_id]
            else:
                # noinspection PyBroadException
                try:
                    project_id = Task.get_task(task_id=task_id).project
                except Exception:
                    project_id = '*'
                cls._task_project_lookup[task_id] = project_id

        if not task_id:
            return ''

        return '<a href="{}"> {} </a>'.format(task_link_template.format(project=project_id, task=task_id), task_id)


class PipelineDecorator(PipelineController):
    _added_decorator = []  # type: List[dict]
    _singleton = None  # type: Optional[PipelineDecorator]
    _eager_step_artifact = 'eager_step'
    _eager_execution_instance = False
    _debug_execute_step_process = False
    _debug_execute_step_function = False
    _default_execution_queue = None

    def __init__(
            self,
            name,  # type: str
            project,  # type: str
            version,  # type: str
            pool_frequency=0.2,  # type: float
            add_pipeline_tags=False,  # type: bool
            target_project=None,  # type: Optional[str]
            abort_on_failure=False,  # type: bool
    ):
        # type: (...) -> ()
        """
        Create a new pipeline controller. The newly created object will launch and monitor the new experiments.

        :param name: Provide pipeline name (if main Task exists it overrides its name)
        :param project: Provide project storing the pipeline (if main Task exists  it overrides its project)
        :param version: Must provide pipeline version. This version allows to uniquely identify the pipeline
            template execution. Examples for semantic versions: version='1.0.1' , version='23', version='1.2'
        :param float pool_frequency: The pooling frequency (in minutes) for monitoring experiments / states.
        :param bool add_pipeline_tags: (default: False) if True, add `pipe: <pipeline_task_id>` tag to all
            steps (Tasks) created by this pipeline.
        :param str target_project: If provided, all pipeline steps are cloned into the target project
        :param bool abort_on_failure: If False (default), failed pipeline steps will not cause the pipeline
            to stop immediately, instead any step that is not connected (or indeirectly connected) to the failed step,
            will still be executed. Nonetheless the pipeline itself will be marked failed, unless the failed step
            was specifically defined with "continue_on_fail=True".
            If True, any failed step will cause the pipeline to immediately abort, stop all running steps,
            and mark the pipeline as failed.
        """
        super(PipelineDecorator, self).__init__(
            name=name,
            project=project,
            version=version,
            pool_frequency=pool_frequency,
            add_pipeline_tags=add_pipeline_tags,
            target_project=target_project,
        )

        # if we are in eager execution, make sure parent class knows it
        if self._eager_execution_instance:
            self._mock_execution = True

        if PipelineDecorator._default_execution_queue:
            super(PipelineDecorator, self).set_default_execution_queue(
                PipelineDecorator._default_execution_queue)

        for n in self._added_decorator:
            self.add_function_step(**n)
        self._added_decorator.clear()
        PipelineDecorator._singleton = self
        self._reference_callback = []
        # map eager steps task id to the new step name
        self._eager_steps_task_id = {}  # type: Dict[str, str]

    def _daemon(self):
        # type: () -> ()
        """
        The main pipeline execution loop. This loop is executed on its own dedicated thread.
        override the daemon function, we only need to update the state

        :return:
        """
        pooling_counter = 0
        launched_nodes = set()
        last_monitor_report = last_plot_report = time()
        while self._stop_event:
            # stop request
            if self._stop_event.wait(self._pool_frequency if pooling_counter else 0.01):
                break

            pooling_counter += 1

            # check the pipeline time limit
            if self._pipeline_time_limit and (time() - self._start_time) > self._pipeline_time_limit:
                break

            # check the state of all current jobs
            # if no a job ended, continue
            completed_jobs = []
            nodes_failed_stop_pipeline = []
            force_execution_plot_update = False
            for j in self._running_nodes:
                node = self._nodes[j]
                if not node.job:
                    continue
                if node.job.is_stopped():
                    completed_jobs.append(j)
                    node_failed = node.job.is_failed()
                    node.executed = node.job.task_id() if not node_failed else False
                    if j in launched_nodes:
                        launched_nodes.remove(j)
                    # check if we need to stop all running steps
                    if node_failed and self._abort_running_steps_on_failure and not node.continue_on_fail:
                        nodes_failed_stop_pipeline.append(node.name)
                elif node.timeout:
                    started = node.job.task.data.started
                    if (datetime.now().astimezone(started.tzinfo) - started).total_seconds() > node.timeout:
                        node.job.abort()
                        completed_jobs.append(j)
                        node.executed = node.job.task_id()
                elif j in launched_nodes and node.job.is_running():
                    # make sure update the execution graph when the job started running
                    # (otherwise it will still be marked queued)
                    launched_nodes.remove(j)
                    force_execution_plot_update = True

            # update running jobs
            self._running_nodes = [j for j in self._running_nodes if j not in completed_jobs]

            # nothing changed, we can sleep
            if not completed_jobs and self._running_nodes:
                # force updating the pipeline state (plot) at least every 5 min.
                if force_execution_plot_update or time()-last_plot_report > self._update_execution_plot_interval:
                    last_plot_report = time()
                    last_monitor_report = time()
                    self.update_execution_plot()
                elif time()-last_monitor_report > self._monitor_node_interval:
                    last_monitor_report = time()
                    self._scan_monitored_nodes()
                continue

            # callback on completed jobs
            if self._experiment_completed_cb or self._post_step_callbacks:
                for job in completed_jobs:
                    job_node = self._nodes.get(job)
                    if not job_node:
                        continue
                    if self._experiment_completed_cb:
                        self._experiment_completed_cb(self, job_node)
                    if self._post_step_callbacks.get(job_node.name):
                        self._post_step_callbacks[job_node.name](self, job_node)

            # check if we need to stop the pipeline, and abort all running steps
            if nodes_failed_stop_pipeline:
                print('Aborting pipeline and stopping all running steps, node {} failed'.format(
                    nodes_failed_stop_pipeline))
                break

            # update current state (in configuration, so that we could later continue an aborted pipeline)
            self._force_task_configuration_update()

            # visualize pipeline state (plot)
            self.update_execution_plot()

        # stop all currently running jobs, protect against changes while iterating):
        for node in list(self._nodes.values()):
            if node.executed is False and not node.continue_on_fail:
                self._pipeline_task_status_failed = True

            if node.job and not node.job.is_stopped():
                node.job.abort()
            elif not node.job and not node.executed:
                # mark Node as skipped if it has no Job object and it is not executed
                node.skip_job = True
                # if this is a standalone node, we need to remove it from the graph
                if not node.parents:
                    # check if this node is anyone's parent
                    found_parent = False
                    for v in list(self._nodes.values()):
                        if node.name in (v.parents or []):
                            found_parent = True
                            break
                    if not found_parent:
                        self._nodes.pop(node.name, None)

        # visualize pipeline state (plot)
        self.update_execution_plot()

        if self._stop_event:
            # noinspection PyBroadException
            try:
                self._stop_event.set()
            except Exception:
                pass

    def update_execution_plot(self):
        # type: () -> ()
        """
        Update sankey diagram of the current pipeline
        """
        self._update_eager_generated_steps()
        super(PipelineDecorator, self).update_execution_plot()

    def _update_eager_generated_steps(self):
        # noinspection PyProtectedMember
        self._task.reload()
        artifacts = self._task.data.execution.artifacts
        # check if we have a new step on the DAG
        eager_artifacts = []
        for a in artifacts:
            if a.key and a.key.startswith('{}:'.format(self._eager_step_artifact)):
                # expected value: '"eager_step":"parent-node-task-id":"eager-step-task-id'
                eager_artifacts.append(a)

        # verify we have the step, if we do not, add it.
        delete_artifact_keys = []
        for artifact in eager_artifacts:
            _, parent_step_task_id, eager_step_task_id = artifact.key.split(':', 2)

            # deserialize node definition
            eager_node_def = json.loads(artifact.type_data.preview)
            eager_node_name, eager_node_def = list(eager_node_def.items())[0]

            # verify we do not have any new nodes on the DAG (i.e. a step generating a Node eagerly)
            parent_node = None
            for node in list(self._nodes.values()):
                if not node.job and not node.executed:
                    continue
                t_id = node.executed or node.job.task_id
                if t_id == parent_step_task_id:
                    parent_node = node
                    break

            if not parent_node:
                # should not happen
                continue

            new_step_node_name = '{}_{}'.format(parent_node.name, eager_node_name)
            counter = 1
            while new_step_node_name in self._nodes:
                new_step_node_name = '{}_{}'.format(new_step_node_name, counter)
                counter += 1

            eager_node_def['name'] = new_step_node_name
            eager_node_def['parents'] = [parent_node.name]
            is_cached = eager_node_def.pop('is_cached', None)
            self._nodes[new_step_node_name] = self.Node(**eager_node_def)
            self._nodes[new_step_node_name].job = RunningJob(existing_task=eager_step_task_id)
            if is_cached:
                self._nodes[new_step_node_name].job.force_set_is_cached(is_cached)

            # make sure we will not rescan it.
            delete_artifact_keys.append(artifact.key)

        # remove all processed eager step artifacts
        if delete_artifact_keys:
            # noinspection PyProtectedMember
            self._task._delete_artifacts(delete_artifact_keys)
            self._force_task_configuration_update()

    def _create_task_from_function(
            self, docker, docker_args, docker_bash_setup_script,
            function, function_input_artifacts, function_kwargs, function_return,
            packages, project_name, task_name, task_type, repo, branch, commit,
            helper_functions,
    ):
        def sanitize(function_source):
            matched = re.match(r"[\s]*@PipelineDecorator.component[\s\\]*\(", function_source)
            if matched:
                function_source = function_source[matched.span()[1]:]
                # find the last ")"
                open_parenthesis = 0
                last_index = -1
                for i, c in enumerate(function_source):
                    if not open_parenthesis and c == ')':
                        last_index = i
                        break
                    elif c == ')':
                        open_parenthesis -= 1
                    elif c == '(':
                        open_parenthesis += 1
                if last_index >= 0:
                    function_source = function_source[last_index+1:].lstrip()
            return function_source

        task_definition = CreateFromFunction.create_task_from_function(
            a_function=function,
            function_kwargs=function_kwargs or None,
            function_input_artifacts=function_input_artifacts,
            function_return=function_return,
            project_name=project_name,
            task_name=task_name,
            task_type=task_type,
            repo=repo,
            branch=branch,
            commit=commit,
            packages=packages,
            docker=docker,
            docker_args=docker_args,
            docker_bash_setup_script=docker_bash_setup_script,
            output_uri=None,
            helper_functions=helper_functions,
            dry_run=True,
            _sanitize_function=sanitize,
        )
        return task_definition

    def _find_executed_node_leaves(self):
        # type: () -> List[PipelineController.Node]
        all_parents = set([p for n in list(self._nodes.values()) if n.executed for p in n.parents])
        executed_leaves = [name for name, n in list(self._nodes.items()) if n.executed and name not in all_parents]
        return executed_leaves

    def _adjust_task_hashing(self, task_hash):
        # type: (dict) -> dict
        """
        Fix the Task hashing so that parameters pointing to the current Task artifact are encoded using the
        hash content of the artifact, instead of the Task.id
        :param task_hash: Task representation dict
        :return: Adjusted Task representation dict
        """
        if task_hash.get('hyper_params'):
            updated_params = {}
            for k, v in task_hash['hyper_params'].items():
                if k.startswith("{}/".format(CreateFromFunction.input_artifact_section)) and \
                        str(v).startswith("{}.".format(self._task.id)):
                    task_id, artifact_name = str(v).split(".", 1)
                    if artifact_name in self._task.artifacts:
                        updated_params[k] = self._task.artifacts[artifact_name].hash
            task_hash['hyper_params'].update(updated_params)

        return task_hash

    @classmethod
    def component(
            cls,
            _func=None, *,
            return_values=('return_object', ),  # type: Union[str, List[str]]
            name=None,  # type: Optional[str]
            cache=False,  # type: bool
            packages=None,  # type: Optional[Union[str, Sequence[str]]]
            parents=None,  # type:  Optional[List[str]]
            execution_queue=None,  # type: Optional[str]
            continue_on_fail=False,  # type: bool
            docker=None,  # type: Optional[str]
            docker_args=None,  # type: Optional[str]
            docker_bash_setup_script=None,  # type: Optional[str]
            task_type=None,  # type: Optional[str]
            repo=None,  # type: Optional[str]
            repo_branch=None,  # type: Optional[str]
            repo_commit=None,  # type: Optional[str]
            helper_functions=None,  # type: Optional[Sequence[Callable]]
            monitor_metrics=None,  # type: Optional[List[Union[Tuple[str, str], Tuple[(str, str), (str, str)]]]]
            monitor_artifacts=None,  # type: Optional[List[Union[str, Tuple[str, str]]]]
            monitor_models=None  # type: Optional[List[Union[str, Tuple[str, str]]]]
    ):
        # type: (...) -> Callable
        """
        pipeline component function to be executed remotely

        :param _func: wrapper function
        :param return_values: Provide a list of names for all the results.
            Notice! If not provided no results will be stored as artifacts.
        :param name: Set the name of the remote task. Required if base_task_id is None.
        :param cache: If True, before launching the new step,
            after updating with the latest configuration, check if an exact Task with the same parameter/code
            was already executed. If it was found, use it instead of launching a new Task.
            Default: False, a new cloned copy of base_task is always used.
            Notice: If the git repo reference does not have a specific commit ID, the Task will never be used.
        :param packages: Manually specify a list of required packages or a local requirements.txt file.
            Example: ["tqdm>=2.1", "scikit-learn"] or "./requirements.txt"
            If not provided, packages are automatically added based on the imports used in the function.
        :param parents: Optional list of parent nodes in the DAG.
            The current step in the pipeline will be sent for execution only after all the parent nodes
            have been executed successfully.
        :param execution_queue: Optional, the queue to use for executing this specific step.
            If not provided, the task will be sent to the default execution queue, as defined on the class
        :param continue_on_fail: (default False). If True, failed step will not cause the pipeline to stop
            (or marked as failed). Notice, that steps that are connected (or indirectly connected)
            to the failed step will be skipped.
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environment
        :param task_type: Optional, The task type to be created. Supported values: 'training', 'testing', 'inference',
            'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc', 'custom'
        :param repo: Optional, specify a repository to attach to the function, when remotely executing.
            Allow users to execute the function inside the specified repository, enabling to load modules/script
            from a repository Notice the execution work directory will be the repository root folder.
            Supports both git repo url link, and local repository path.
            Example remote url: 'https://github.com/user/repo.git'
            Example local repo copy: './repo' -> will automatically store the remote
            repo url and commit ID based on the locally cloned copy
        :param repo_branch: Optional, specify the remote repository branch (Ignored, if local repo path is used)
        :param repo_commit: Optional, specify the repository commit id (Ignored, if local repo path is used)
        :param helper_functions: Optional, a list of helper functions to make available
            for the standalone pipeline step function Task.
        :param monitor_metrics: Optional, log the step's metrics on the pipeline Task.
            Format is a list of pairs metric (title, series) to log:
                [(step_metric_title, step_metric_series), ]
                Example: [('test', 'accuracy'), ]
            Or a list of tuple pairs, to specify a different target metric for to use on the pipeline Task:
                [((step_metric_title, step_metric_series), (target_metric_title, target_metric_series)), ]
                Example: [[('test', 'accuracy'), ('model', 'accuracy')], ]
        :param monitor_artifacts: Optional, log the step's artifacts on the pipeline Task.
            Provided a list of artifact names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: [('processed_data', 'final_processed_data'), ]
            Alternatively user can also provide a list of artifacts to monitor
            (target artifact name will be the same as original artifact name)
            Example: ['processed_data', ]
        :param monitor_models: Optional, log the step's output models on the pipeline Task.
            Provided a list of model names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: [('model_weights', 'final_model_weights'), ]
            Alternatively user can also provide a list of models to monitor
            (target models name will be the same as original model)
            Example: ['model_weights', ]
            To select the latest (lexicographic) model use "model_*", or the last created model with just "*"
            Example:  ['model_weights_*', ]

        :return: function wrapper
        """
        def decorator_wrap(func):
            _name = name or str(func.__name__)
            function_return = return_values if isinstance(return_values, (tuple, list)) else [return_values]

            inspect_func = inspect.getfullargspec(func)
            # add default argument values
            if inspect_func.args:
                default_values = list(inspect_func.defaults or [])
                default_values = ([None] * (len(inspect_func.args)-len(default_values))) + default_values
                function_kwargs = {k: v for k, v in zip(inspect_func.args, default_values)}
            else:
                function_kwargs = dict()

            add_step_spec = dict(
                name=_name,
                function=func,
                function_kwargs=function_kwargs,
                function_return=function_return,
                cache_executed_step=cache,
                packages=packages,
                parents=parents,
                execution_queue=execution_queue,
                continue_on_fail=continue_on_fail,
                docker=docker,
                docker_args=docker_args,
                docker_bash_setup_script=docker_bash_setup_script,
                task_type=task_type,
                repo=repo,
                repo_branch=repo_branch,
                repo_commit=repo_commit,
                helper_functions=helper_functions,
                monitor_metrics=monitor_metrics,
                monitor_models=monitor_models,
                monitor_artifacts=monitor_artifacts,
            )

            if cls._singleton:
                cls._singleton.add_function_step(**add_step_spec)
            else:
                cls._added_decorator.append(add_step_spec)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if cls._debug_execute_step_function:
                    args = [v._remoteref() if isinstance(v, LazyEvalWrapper) else v for v in args]
                    kwargs = {k: v._remoteref() if isinstance(v, LazyEvalWrapper) else v for k, v in kwargs.items()}

                    func_return = []

                    def result_wrapper(a_func_return, return_index):
                        if not a_func_return:
                            a_func_return.append(func(*args, **kwargs))
                        a_func_return = a_func_return[0]
                        return a_func_return if return_index is None else a_func_return[return_index]

                    if len(function_return) == 1:
                        return LazyEvalWrapper(
                            callback=functools.partial(result_wrapper, func_return, None),
                            remote_reference=functools.partial(result_wrapper, func_return, None))
                    else:
                        return_w = [LazyEvalWrapper(
                            callback=functools.partial(result_wrapper, func_return, i),
                            remote_reference=functools.partial(result_wrapper, func_return, i))
                            for i, _ in enumerate(function_return)]
                        return return_w

                # resolve all lazy objects if we have any:
                kwargs_artifacts = {}
                for i, v in enumerate(args):
                    kwargs[inspect_func.args[i]] = v

                kwargs_artifacts.update(
                    {k: v._remoteref() for k, v in kwargs.items() if isinstance(v, LazyEvalWrapper)}
                )
                kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, LazyEvalWrapper)}

                # check if we have the singleton
                if not cls._singleton:
                    # todo: somehow make sure the generated tasks list the parent pipeline as parent
                    original_tags = Task.current_task().get_tags(), Task.current_task().get_system_tags()
                    # This is an adhoc pipeline step,
                    PipelineDecorator._eager_execution_instance = True
                    a_pipeline = PipelineDecorator(
                        name=name,
                        project='DevOps',  # it will not actually be used
                        version='0.0.0',
                        pool_frequency=111,
                        add_pipeline_tags=False,
                        target_project=None,
                    )

                    target_queue = \
                        PipelineDecorator._default_execution_queue or \
                        Task.current_task().data.execution.queue
                    if target_queue:
                        PipelineDecorator.set_default_execution_queue(target_queue)
                    else:
                        # if we are are not running from a queue, we are probably in debug mode
                        a_pipeline._clearml_job_class = LocalClearmlJob
                        a_pipeline._default_execution_queue = 'mock'

                    # restore tags, the pipeline might add a few
                    Task.current_task().set_tags(original_tags[0])
                    Task.current_task().set_system_tags(original_tags[1])

                # get original node name
                _node_name = _name
                # get node
                _node = cls._singleton._nodes[_node_name]

                # if we already have a JOB on the node, this means we are calling the same function/task
                # twice inside the pipeline, this means we need to replicate the node.
                if _node.job:
                    _node = _node.copy()
                    # find a new name
                    counter = 1
                    while _node.name in cls._singleton._nodes:
                        _node.name = '{}_{}'.format(_node_name, counter)
                        counter += 1
                    _node_name = _node.name
                    cls._singleton._nodes[_node.name] = _node

                # update artifacts kwargs
                for k, v in kwargs_artifacts.items():
                    if k in kwargs:
                        kwargs.pop(k, None)
                    _node.parameters["{}/{}".format(CreateFromFunction.input_artifact_section, k)] = v
                    if v and '.' in str(v):
                        parent_id, _ = str(v).split('.', 1)
                        # find parent and push it into the _node.parents
                        for n, node in cls._singleton._nodes.items():
                            if n != _node.name and node.executed and node.executed == parent_id:
                                if n not in _node.parents:
                                    _node.parents.append(n)
                                break
                for k, v in kwargs.items():
                    if v is None or isinstance(v, (bool, int, float, str)):
                        _node.parameters["{}/{}".format(CreateFromFunction.kwargs_section, k)] = v
                    elif isinstance(v, (list, tuple)) and all(isinstance(i, (bool, int, float, str)) for i in v):
                        _node.parameters["{}/{}".format(CreateFromFunction.kwargs_section, k)] = v
                    else:
                        # we need to create an artifact
                        artifact_name = 'result_{}_{}'.format(re.sub(r'\W+', '', _node.name), k)
                        cls._singleton._task.upload_artifact(
                            name=artifact_name, artifact_object=v, wait_on_upload=True)
                        _node.parameters["{}/{}".format(CreateFromFunction.input_artifact_section, k)] = \
                            "{}.{}".format(cls._singleton._task.id, artifact_name)
                        # now add all the executed nodes as parents (only the leaves of the DAG, no need for parents)
                        _node.parents = list(
                            set((_node.parents or []) + cls._singleton._find_executed_node_leaves())
                            - set(list(_node.name)))

                # verify the new step
                cls._singleton._verify_node(_node)
                # launch the new step
                cls._singleton._launch_node(_node)
                # check if we generated the pipeline we need to update the new eager step
                if PipelineDecorator._eager_execution_instance and _node.job:
                    # check if we need to add the pipeline tag on the new node
                    pipeline_tags = [t for t in Task.current_task().get_tags() or []
                                     if str(t).startswith(cls._node_tag_prefix)]
                    if pipeline_tags and _node.job and _node.job.task:
                        pipeline_tags = list(set((_node.job.task.get_tags() or []) + pipeline_tags))
                        _node.job.task.set_tags(pipeline_tags)
                    # force parent task as pipeline
                    _node.job.task._edit(parent=Task.current_task().parent)
                    # store the new generated node, so we can later serialize it
                    pipeline_dag = cls._singleton._serialize()
                    # check if node is cached
                    if _node.job.is_cached_task():
                        pipeline_dag[_node_name]['is_cached'] = True
                    # store entire definition on the parent pipeline
                    from clearml.backend_api.services import tasks
                    artifact = tasks.Artifact(
                        key='{}:{}:{}'.format(cls._eager_step_artifact, Task.current_task().id, _node.job.task_id()),
                        type="json",
                        mode='output',
                        type_data=tasks.ArtifactTypeData(
                            preview=json.dumps({_node_name: pipeline_dag[_node_name]}),
                            content_type='application/pipeline')
                    )
                    req = tasks.AddOrUpdateArtifactsRequest(
                        task=Task.current_task().parent, artifacts=[artifact], force=True)
                    res = Task.current_task().send(req, raise_on_errors=False)
                    if not res or not res.response or not res.response.updated:
                        pass

                def results_reference(return_name):
                    # wait until job is completed
                    _node.job.wait(pool_period=0.2)
                    if _node.job.is_failed() and not _node.continue_on_fail:
                        raise ValueError(
                            'Pipeline step "{}", Task ID={} failed'.format(_node.name, _node.job.task_id()))

                    _node.executed = _node.job.task_id()
                    return "{}.{}".format(_node.job.task_id(), return_name)

                def result_wrapper(return_name):
                    # wait until job is completed
                    _node.job.wait(pool_period=0.2)
                    if _node.job.is_failed():
                        raise ValueError(
                            'Pipeline step "{}", Task ID={} failed'.format(_node.name, _node.job.task_id()))

                    _node.executed = _node.job.task_id()
                    return Task.get_task(_node.job.task_id()).artifacts[return_name].get()

                return_w = [LazyEvalWrapper(
                        callback=functools.partial(result_wrapper, n),
                        remote_reference=functools.partial(results_reference, n)) for n in function_return]

                return return_w[0] if len(return_w) == 1 else return_w

            return wrapper

        return decorator_wrap if _func is None else decorator_wrap(_func)

    @classmethod
    def pipeline(
            cls,
            _func=None, *,  # noqa
            name,  # type: str
            project,  # type: str
            version,  # type: str
            default_queue=None,  # type: Optional[str]
            pool_frequency=0.2,  # type: float
            add_pipeline_tags=False,  # type: bool
            target_project=None,  # type: Optional[str]
            abort_on_failure=False,  # type: bool
            pipeline_execution_queue='services',  # type: Optional[str]
    ):
        # type: (...) -> Callable
        """
        Decorate pipeline logic function.

        :param name: Provide pipeline name (if main Task exists it overrides its name)
        :param project: Provide project storing the pipeline (if main Task exists  it overrides its project)
        :param version: Must provide pipeline version. This version allows to uniquely identify the pipeline
            template execution. Examples for semantic versions: version='1.0.1' , version='23', version='1.2'
        :param default_queue: default pipeline step queue
        :param float pool_frequency: The pooling frequency (in minutes) for monitoring experiments / states.
        :param bool add_pipeline_tags: (default: False) if True, add `pipe: <pipeline_task_id>` tag to all
            steps (Tasks) created by this pipeline.
        :param str target_project: If provided, all pipeline steps are cloned into the target project
        :param bool abort_on_failure: If False (default), failed pipeline steps will not cause the pipeline
            to stop immediately, instead any step that is not connected (or indirectly connected) to the failed step,
            will still be executed. Nonetheless the pipeline itself will be marked failed, unless the failed step
            was specifically defined with "continue_on_fail=True".
            If True, any failed step will cause the pipeline to immediately abort, stop all running steps,
            and mark the pipeline as failed.
        :param pipeline_execution_queue: remote pipeline execution queue (default 'services' queue).
            If None is passed, execute the pipeline logic locally (pipeline steps are still executed remotely)
        """
        def decorator_wrap(func):

            def internal_decorator(*args, **kwargs):
                pipeline_kwargs = dict(**(kwargs or {}))
                inspect_func = inspect.getfullargspec(func)
                if args:
                    if not inspect_func.args:
                        raise ValueError("Could not parse function arguments")

                    pipeline_kwargs.update({inspect_func.args[i]: v for i, v in enumerate(args)})

                # add default function arguments if we have defaults for all arguments
                if inspect_func.args:
                    default_values = list(inspect_func.defaults or [])
                    default_values = ([None] * (len(inspect_func.args) - len(default_values))) + default_values
                    default_kwargs = {k: v for k, v in zip(inspect_func.args, default_values)}
                    default_kwargs.update(pipeline_kwargs)
                    pipeline_kwargs = default_kwargs

                # run the entire pipeline locally, as python functions
                if cls._debug_execute_step_function:
                    ret_val = func(**pipeline_kwargs)
                    LazyEvalWrapper.trigger_all_remote_references()
                    return ret_val

                if default_queue:
                    cls.set_default_execution_queue(default_queue)

                a_pipeline = PipelineDecorator(
                    name=name,
                    project=project,
                    version=version,
                    pool_frequency=pool_frequency,
                    add_pipeline_tags=add_pipeline_tags,
                    target_project=target_project,
                    abort_on_failure=abort_on_failure,
                )

                if PipelineDecorator._debug_execute_step_process:
                    a_pipeline._clearml_job_class = LocalClearmlJob
                    a_pipeline._default_execution_queue = 'mock'

                a_pipeline._clearml_job_class.register_hashing_callback(a_pipeline._adjust_task_hashing)

                # add pipeline arguments
                if pipeline_kwargs:
                    a_pipeline.get_parameters().update(pipeline_kwargs)

                # serialize / deserialize state only if we are running locally
                a_pipeline._start(wait=False)

                # sync arguments back
                for k in pipeline_kwargs.keys():
                    if k in a_pipeline.get_parameters():
                        pipeline_kwargs[k] = a_pipeline.get_parameters()[k]

                # run the actual pipeline
                if not PipelineDecorator._debug_execute_step_process and pipeline_execution_queue:
                    # rerun the pipeline on a remote machine
                    a_pipeline._task.execute_remotely(queue_name=pipeline_execution_queue)
                    # when we get here it means we are running remotely

                # this time the pipeline is executed only on the remote machine
                try:
                    func(**pipeline_kwargs)
                except Exception:
                    a_pipeline.stop(mark_failed=True)
                    raise

                triggered_exception = None
                try:
                    LazyEvalWrapper.trigger_all_remote_references()
                except Exception as ex:
                    triggered_exception = ex

                # make sure we wait for all nodes to finish
                waited = True
                while waited:
                    waited = False
                    for node in list(a_pipeline._nodes.values()):
                        if node.executed or not node.job or node.job.is_stopped():
                            continue
                        node.job.wait(pool_period=15)
                        waited = True
                # now we can stop the pipeline
                a_pipeline.stop()
                # now we can raise the exception
                if triggered_exception:
                    raise triggered_exception
                return

            return internal_decorator

        return decorator_wrap if _func is None else decorator_wrap(_func)

    @classmethod
    def set_default_execution_queue(cls, default_execution_queue):
        # type: (Optional[str]) -> None
        """
        Set the default execution queue for if pipeline step does not specify an execution queue

        :param default_execution_queue: The execution queue to use if no execution queue is provided
        """
        cls._default_execution_queue = str(default_execution_queue) if default_execution_queue else None

    @classmethod
    def run_locally(cls):
        # type: () -> ()
        """
        Set local mode, run all functions locally as subprocess or serially as functions

        Run the full pipeline DAG locally, where steps are executed as sub-processes Tasks
        Notice: running the DAG locally assumes the local code execution (i.e. it will not clone & apply git diff)

        """
        cls._debug_execute_step_process = True
        cls._debug_execute_step_function = False

    @classmethod
    def debug_pipeline(cls):
        # type: () -> ()
        """
        Set debugging mode, run all functions locally as subprocess or serially as functions
        Run the full pipeline DAG locally, where steps are executed as sub-processes Tasks
        Notice:
            running the DAG locally assumes the local code execution (i.e. it will not clone & apply git diff)
            Pipeline steps are executed as functions (no Task will be created), fo ease debugging J
        """
        cls._debug_execute_step_process = True
        cls._debug_execute_step_function = True

    @classmethod
    def get_current_pipeline(cls):
        # type: () -> "PipelineDecorator"
        """
        Return the currently running pipeline instance
        """
        return cls._singleton
