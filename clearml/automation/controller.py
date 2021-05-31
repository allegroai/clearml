import re
from copy import copy
from datetime import datetime
from logging import getLogger
from threading import Thread, Event, RLock
from time import time

from attr import attrib, attrs
from typing import Sequence, Optional, Mapping, Callable, Any, Union, List

from ..backend_interface.util import get_or_create_project
from ..debugging.log import LoggerRoot
from ..task import Task
from ..automation import ClearmlJob
from ..model import BaseModel
from ..utilities.process.mp import leave_process


class PipelineController(object):
    """
    Pipeline controller.
    Pipeline is a DAG of base tasks, each task will be cloned (arguments changed as required) executed and monitored
    The pipeline process (task) itself can be executed manually or by the clearml-agent services queue.
    Notice: The pipeline controller lives as long as the pipeline itself is being executed.
    """
    _tag = 'pipeline'
    _step_pattern = r"\${[^}]*}"
    _config_section = 'Pipeline'
    _task_project_lookup = {}

    @attrs
    class Node(object):
        name = attrib(type=str)
        base_task_id = attrib(type=str)
        queue = attrib(type=str, default=None)
        parents = attrib(type=list, default=[])
        timeout = attrib(type=float, default=None)
        parameters = attrib(type=dict, default={})
        task_overrides = attrib(type=dict, default={})
        executed = attrib(type=str, default=None)
        clone_task = attrib(type=bool, default=True)
        job = attrib(type=ClearmlJob, default=None)
        skip_job = attrib(type=bool, default=False)
        cache_executed_step = attrib(type=bool, default=False)

    def __init__(
            self,
            pool_frequency=0.2,  # type: float
            default_execution_queue=None,  # type: Optional[str]
            pipeline_time_limit=None,  # type: Optional[float]
            auto_connect_task=True,  # type: Union[bool, Task]
            always_create_task=False,  # type: bool
            add_pipeline_tags=False,  # type: bool
            target_project=None,  # type: Optional[str]
            pipeline_name=None,  # type: Optional[str]
            pipeline_project=None,  # type: Optional[str]
    ):
        # type: (...) -> ()
        """
        Create a new pipeline controller. The newly created object will launch and monitor the new experiments.

        :param float pool_frequency: The pooling frequency (in minutes) for monitoring experiments / states.
        :param str default_execution_queue: The execution queue to use if no execution queue is provided
        :param float pipeline_time_limit: The maximum time (minutes) for the entire pipeline process. The
            default is ``None``, indicating no time limit.
        :param bool auto_connect_task: Store pipeline arguments and configuration in the Task
            - ``True`` - The pipeline argument and configuration will be stored in the current Task. All arguments will
                be under the hyper-parameter section ``Pipeline``, and the pipeline DAG will be stored as a
                Task configuration object named ``Pipeline``.
                Notice that when running remotely the DAG definitions will be taken from the Task itself (e.g. editing
                the configuration in the UI will be reflected in the actual DAG created).
            - ``False`` - Do not store DAG configuration on the Task.
                In remote execution the DAG will always be created from code.
            - ``Task`` - A specific Task object to connect the pipeline with.
        :param bool always_create_task: Always create a new Task
            - ``True`` - No current Task initialized. Create a new task named ``Pipeline`` in the ``base_task_id``
                project.
            - ``False`` - Use the :py:meth:`task.Task.current_task` (if exists) to report statistics.
        :param bool add_pipeline_tags: (default: False) if True, add `pipe: <pipeline_task_id>` tag to all
            steps (Tasks) created by this pipeline.
        :param str target_project: If provided, all pipeline steps are cloned into the target project
        :param pipeline_name: Optional, provide pipeline name if main Task is not present (default current date)
        :param pipeline_project: Optional, provide project storing the pipeline if main Task is not present
        """
        self._nodes = {}
        self._running_nodes = []
        self._start_time = None
        self._pipeline_time_limit = pipeline_time_limit * 60. if pipeline_time_limit else None
        self._default_execution_queue = default_execution_queue
        self._pool_frequency = pool_frequency * 60.
        self._thread = None
        self._stop_event = None
        self._experiment_created_cb = None
        self._experiment_completed_cb = None
        self._pre_step_callbacks = {}
        self._post_step_callbacks = {}
        self._target_project = target_project or ''
        self._add_pipeline_tags = add_pipeline_tags
        self._task = auto_connect_task if isinstance(auto_connect_task, Task) else Task.current_task()
        self._step_ref_pattern = re.compile(self._step_pattern)
        self._reporting_lock = RLock()
        self._pipeline_task_status_failed = None
        if not self._task and always_create_task:
            self._task = Task.init(
                project_name=pipeline_project or 'Pipelines',
                task_name=pipeline_name or 'Pipeline {}'.format(datetime.now()),
                task_type=Task.TaskTypes.controller,
            )

        self._auto_connect_task = bool(auto_connect_task) and bool(self._task)
        # make sure we add to the main Task the pipeline tag
        if self._task:
            self._task.add_tags([self._tag])

    def add_step(
            self,
            name,  # type: str
            base_task_id=None,  # type: Optional[str]
            parents=None,  # type: Optional[Sequence[str]]
            parameter_override=None,  # type: Optional[Mapping[str, Any]]
            task_overrides=None,  # type: Optional[Mapping[str, Any]]
            execution_queue=None,  # type: Optional[str]
            time_limit=None,  # type: Optional[float]
            base_task_project=None,  # type: Optional[str]
            base_task_name=None,  # type: Optional[str]
            clone_base_task=True,  # type: bool
            pre_execute_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node, dict], bool]]  # noqa
            post_execute_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node], None]]  # noqa
            cache_executed_step=False,  # type: bool
    ):
        # type: (...) -> bool
        """
        Add a step to the pipeline execution DAG.
        Each step must have a unique name (this name will later be used to address the step)

        :param str name: Unique of the step. For example `stage1`
        :param str base_task_id: The Task ID to use for the step. Each time the step is executed,
            the base Task is cloned, then the cloned task will be sent for execution.
        :param list parents: Optional list of parent nodes in the DAG.
            The current step in the pipeline will be sent for execution only after all the parent nodes
            have been executed successfully.
        :param dict parameter_override: Optional parameter overriding dictionary.
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
        :param dict task_overrides: Optional task section overriding dictionary.
            The dict values can reference a previously executed step using the following form '${step_name}'
            Examples:
            - clear git repository commit ID
                parameter_override={'script.version_num': '' }
            - git repository commit branch
                parameter_override={'script.branch': '${stage1.script.branch}' }
            - container image
                parameter_override={'container.image': '${stage1.container.image}' }
        :param str execution_queue: Optional, the queue to use for executing this specific step.
            If not provided, the task will be sent to the default execution queue, as defined on the class
        :param float time_limit: Default None, no time limit.
            Step execution time limit, if exceeded the Task is aborted and the pipeline is stopped and marked failed.
        :param str base_task_project: If base_task_id is not given,
            use the base_task_project and base_task_name combination to retrieve the base_task_id to use for the step.
        :param str base_task_name: If base_task_id is not given,
            use the base_task_project and base_task_name combination to retrieve the base_task_id to use for the step.
        :param bool clone_base_task: If True (default) the pipeline will clone the base task, and modify/enqueue
            the cloned Task. If False, the base-task is used directly, notice it has to be in draft-mode (created).
        :param Callable pre_execute_callback: Callback function, called when the step (Task) is created
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

        :param Callable post_execute_callback: Callback function, called when a step (Task) is completed
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

        :return: True if successful
        """

        # always store callback functions (even when running remotely)
        if pre_execute_callback:
            self._pre_step_callbacks[name] = pre_execute_callback
        if post_execute_callback:
            self._post_step_callbacks[name] = post_execute_callback

        # when running remotely do nothing, we will deserialize ourselves when we start
        # if we are not cloning a Task, we assume this step is created from code, not from the configuration
        if clone_base_task and self._has_stored_configuration():
            return True

        if name in self._nodes:
            raise ValueError('Node named \'{}\' already exists in the pipeline dag'.format(name))

        if not base_task_id:
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

        self._nodes[name] = self.Node(
            name=name, base_task_id=base_task_id, parents=parents or [],
            queue=execution_queue, timeout=time_limit,
            parameters=parameter_override or {},
            clone_task=clone_base_task,
            task_overrides=task_overrides,
            cache_executed_step=cache_executed_step,
        )

        if self._task and not self._task.running_locally():
            self.update_execution_plot()

        return True

    def start(
            self,
            step_task_created_callback=None,  # type: Optional[Callable[[PipelineController, PipelineController.Node, dict], bool]]  # noqa
            step_task_completed_callback=None  # type: Optional[Callable[[PipelineController, PipelineController.Node], None]]  # noqa
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


        :return: True, if the controller started. False, if the controller did not start.

        """
        if self._thread:
            return True

        params, pipeline_dag = self._serialize_pipeline_task()

        # deserialize back pipeline state
        if not params['continue_pipeline']:
            for k in pipeline_dag:
                pipeline_dag[k]['executed'] = None

        self._default_execution_queue = params['default_queue']
        self._add_pipeline_tags = params['add_pipeline_tags']
        self._target_project = params['target_project'] or ''
        self._deserialize(pipeline_dag)

        # if we continue the pipeline, make sure that we re-execute failed tasks
        if params['continue_pipeline']:
            for node in self._nodes.values():
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
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()
        return True

    def start_remotely(self, queue='services', exit_process=True):
        # type: (str, bool) -> Task
        """
        Start the current pipeline remotely (on the selected services queue)
        The current process will be stopped if exit_process is True.

        :param queue: queue name to launch the pipeline on
        :param exit_process: If True exit the current process after launching on the enqueuing on the queue

        :return: The remote Task object
        """
        if not self._task:
            raise ValueError(
                "Could not find main Task, "
                "PipelineController must be created with `always_create_task=True`")

        # serialize state only if we are running locally
        if Task.running_locally() or not self._task.is_main_task():
            self._serialize_pipeline_task()
            self.update_execution_plot()

        # stop current Task and execute remotely or no-op
        self._task.execute_remotely(queue_name=queue, exit_process=exit_process, clone=False)

        if not Task.running_locally() and self._task.is_main_task():
            self.start()
            self.wait()
            self.stop()
            leave_process(0)
        else:
            return self._task

    def stop(self, timeout=None):
        # type: (Optional[float]) -> ()
        """
        Stop the pipeline controller and the optimization thread.

        :param float timeout: Wait timeout for the optimization thread to exit (minutes).
            The default is ``None``, indicating do not wait terminate immediately.
        """
        self.wait(timeout=timeout)
        if self._task and self._pipeline_task_status_failed:
            print('Setting pipeline controller Task as failed (due to failed steps) !')
            self._task.close()
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

    def _serialize_pipeline_task(self):
        # type: () -> (dict, dict)
        """
        Serialize current pipeline state into the main Task

        :return: params, pipeline_dag
        """
        params = {'continue_pipeline': False,
                  'default_queue': self._default_execution_queue,
                  'add_pipeline_tags': self._add_pipeline_tags,
                  'target_project': self._target_project,
                  }
        pipeline_dag = self._serialize()

        # serialize pipeline state
        if self._task and self._auto_connect_task:
            self._task.connect_configuration(pipeline_dag, name=self._config_section)
            self._task.connect(params, name=self._config_section)

        return params, pipeline_dag

    def _serialize(self):
        # type: () -> dict
        """
        Store the definition of the pipeline DAG into a dictionary.
        This dictionary will be used to store the DAG as a configuration on the Task
        :return:
        """
        dag = {name: dict((k, v) for k, v in node.__dict__.items() if k not in ('job', 'name'))
               for name, node in self._nodes.items()}

        return dag

    def _deserialize(self, dag_dict):
        # type: (dict) -> ()
        """
        Restore the DAG from a dictionary.
        This will be used to create the DAG from the dict stored on the Task, when running remotely.
        :return:
        """
        # make sure that we override nodes that we do not clone.
        for name in self._nodes:
            if self._nodes[name].clone_task and name in dag_dict and dag_dict[name].get('clone_task'):
                dag_dict[name] = dict(
                    (k, v) for k, v in self._nodes[name].__dict__.items() if k not in ('job', 'name'))

        self._nodes = {
            k: self.Node(name=k, **v) if not v.get('clone_task') or k not in self._nodes else self._nodes[k]
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
        for node in self._nodes.values():
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
        if not node.base_task_id:
            raise ValueError("Node '{}', base_task_id is empty".format(node.name))

        if not self._default_execution_queue and not node.queue:
            raise ValueError("Node '{}' missing execution queue, "
                             "no default queue defined and no specific node queue defined".format(node.name))

        task = Task.get_task(task_id=node.base_task_id)
        if not task:
            raise ValueError("Node '{}', base_task_id={} is invalid".format(node.name, node.base_task_id))

        pattern = self._step_ref_pattern

        for v in node.parameters.values():
            if isinstance(v, str):
                for g in pattern.findall(v):
                    self.__verify_step_reference(node, g)

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
            for k, node in self._nodes.items():
                if k in visited:
                    continue
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

        node.job = ClearmlJob(
            base_task_id=node.base_task_id, parameter_override=updated_hyper_parameters,
            tags=['pipe: {}'.format(self._task.id)] if self._add_pipeline_tags and self._task else None,
            parent=self._task.id if self._task else None,
            disable_clone_task=not node.clone_task,
            task_overrides=task_overrides,
            allow_caching=node.cache_executed_step,
            **extra_args
        )

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
        else:
            node.job.launch(queue_name=node.queue or self._default_execution_queue)

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
                node_params.append(node.job.task_parameter_override if node.job else node.parameters) or {}
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
            param_str = str(param)
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
                name=self._config_section, config_type='dictionary', config_dict=pipeline_dag)

    def _daemon(self):
        # type: () -> ()
        """
        The main pipeline execution loop. This loop is executed on its own dedicated thread.
        :return:
        """
        pooling_counter = 0
        launched_nodes = set()
        last_plot_report = time()
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
            for j in self._running_nodes:
                node = self._nodes[j]
                if not node.job:
                    continue
                if node.job.is_stopped():
                    completed_jobs.append(j)
                    node.executed = node.job.task_id() if not node.job.is_failed() else False
                    if j in launched_nodes:
                        launched_nodes.remove(j)
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
                if force_execution_plot_update or time()-last_plot_report > 5.*60:
                    last_plot_report = time()
                    self.update_execution_plot()
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

            # Pull the next jobs in the pipeline, based on the completed list
            next_nodes = []
            for node in self._nodes.values():
                # check if already processed or needs to be skipped
                if node.job or node.executed or node.skip_job:
                    continue
                completed_parents = [bool(p in self._nodes and self._nodes[p].executed) for p in node.parents or []]
                if all(completed_parents):
                    next_nodes.append(node.name)

            # update the execution graph
            for name in next_nodes:
                if self._launch_node(self._nodes[name]):
                    print('Launching step: {}'.format(name))
                    print('Parameters:\n{}'.format(self._nodes[name].job.task_parameter_override))
                    self._running_nodes.append(name)
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
        for node in self._nodes.values():
            if node.executed is False:
                self._pipeline_task_status_failed = True
            if node.job and node.executed and not node.job.is_stopped():
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

    def __verify_step_reference(self, node, step_ref_string):
        # type: (PipelineController.Node, str) -> bool
        """
        Verify the step reference. For example "${step1.parameters.Args/param}"
        :param Node node: calling reference node (used for logging)
        :param str step_ref_string: For example "${step1.parameters.Args/param}"
        :return: True if valid reference
        """
        parts = step_ref_string[2:-1].split('.')
        v = step_ref_string
        if len(parts) < 2:
            raise ValueError("Node '{}', parameter '{}' is invalid".format(node.name, v))
        prev_step = parts[0]
        input_type = parts[1]
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
        return True

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
