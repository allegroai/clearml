import re
from copy import copy
from datetime import datetime
from logging import getLogger
from threading import Thread, Event
from time import time

from attr import attrib, attrs
from typing import Sequence, Optional, Mapping, Callable, Any, Union

from ..debugging.log import LoggerRoot
from ..task import Task
from ..automation import TrainsJob
from ..model import BaseModel


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

    @attrs
    class Node(object):
        name = attrib(type=str)
        base_task_id = attrib(type=str)
        queue = attrib(type=str, default=None)
        parents = attrib(type=list, default=[])
        timeout = attrib(type=float, default=None)
        parameters = attrib(type=dict, default={})
        executed = attrib(type=str, default=None)
        job = attrib(type=TrainsJob, default=None)

    def __init__(
            self,
            pool_frequency=0.2,  # type: float
            default_execution_queue=None,  # type: Optional[str]
            pipeline_time_limit=None,  # type: Optional[float]
            auto_connect_task=True,  # type: Union[bool, Task]
            always_create_task=False,  # type: bool
            add_pipeline_tags=False,  # type: bool
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

            - ``False`` - Do not store with Task.
            - ``Task`` - A specific Task object to connect the pipeline with.
        :param bool always_create_task: Always create a new Task
            - ``True`` - No current Task initialized. Create a new task named ``Pipeline`` in the ``base_task_id``
              project.

            - ``False`` - Use the :py:meth:`task.Task.current_task` (if exists) to report statistics.
        :param bool add_pipeline_tags: (default: False) if True, add `pipe: <pipeline_task_id>` tag to all
            steps (Tasks) created by this pipeline.
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
        self._add_pipeline_tags = add_pipeline_tags
        self._task = auto_connect_task if isinstance(auto_connect_task, Task) else Task.current_task()
        self._step_ref_pattern = re.compile(self._step_pattern)
        if not self._task and always_create_task:
            self._task = Task.init(
                project_name='Pipelines',
                task_name='Pipeline {}'.format(datetime.now()),
                task_type=Task.TaskTypes.controller,
            )

        # make sure all the created tasks are our children, as we are creating them
        if self._task:
            self._task.add_tags([self._tag])
            self._auto_connect_task = bool(auto_connect_task)

    def add_step(
            self,
            name,  # type: str
            base_task_id=None,  # type: Optional[str]
            parents=None,  # type: Optional[Sequence[str]]
            parameter_override=None,  # type: Optional[Mapping[str, Any]]
            execution_queue=None,  # type: Optional[str]
            time_limit=None,  # type: Optional[float]
            base_task_project=None,  # type: Optional[str]
            base_task_name=None,  # type: Optional[str]
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
                Artifact access
                    parameter_override={'Args/input_file': '${stage1.artifacts.mydata.url}' }
                Model access (last model used)
                    parameter_override={'Args/input_file': '${stage1.models.output.-1.url}' }
                Parameter access
                    parameter_override={'Args/input_file': '${stage3.parameters.Args/input_file}' }
                Task ID
                    parameter_override={'Args/input_file': '${stage3.id}' }
        :param str execution_queue: Optional, the queue to use for executing this specific step.
            If not provided, the task will be sent to the default execution queue, as defined on the class
        :param float time_limit: Default None, no time limit.
            Step execution time limit, if exceeded the Task is aborted and the pipeline is stopped and marked failed.
        :param str base_task_project: If base_task_id is not given,
            use the base_task_project and base_task_name combination to retrieve the base_task_id to use for the step.
        :param str base_task_name: If base_task_id is not given,
            use the base_task_project and base_task_name combination to retrieve the base_task_id to use for the step.
        :return: True if successful
        """
        # when running remotely do nothing, we will deserialize ourselves when we start
        if self._has_stored_configuration():
            return True

        if name in self._nodes:
            raise ValueError('Node named \'{}\' already exists in the pipeline dag'.format(name))

        if not base_task_id:
            if not base_task_project or not base_task_name:
                raise ValueError('Either base_task_id or base_task_project/base_task_name must be provided')
            base_task = Task.get_task(project_name=base_task_project, task_name=base_task_name)
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
            parameters=parameter_override or {})

        return True

    def start(self, run_remotely=False, step_task_created_callback=None):
        # type: (Union[bool, str], Optional[Callable[[PipelineController.Node, dict], None]]) -> bool
        """
        Start the pipeline controller.
        If the calling process is stopped, then the controller stops as well.

        :param bool run_remotely: (default False), If True stop the current process and continue execution
            on a remote machine. This is done by calling the Task.execute_remotely with the queue name 'services'.
            If `run_remotely` is a string, it will specify the execution queue for the pipeline remote execution.
        :param Callable step_task_created_callback: Callback function, called when a step (Task) is created
            and before it is sent for execution.

            .. code-block:: py

                def step_created_callback(
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :return: True, if the controller started. False, if the controller did not start.

        """
        if self._thread:
            return True

        # serialize pipeline state
        pipeline_dag = self._serialize()
        self._task.connect_configuration(pipeline_dag, name=self._config_section)
        params = {'continue_pipeline': False,
                  'default_queue': self._default_execution_queue,
                  'add_pipeline_tags': self._add_pipeline_tags,
                  }
        self._task.connect(params, name=self._config_section)
        # deserialize back pipeline state
        if not params['continue_pipeline']:
            for k in pipeline_dag:
                pipeline_dag[k]['executed'] = None

        self._default_execution_queue = params['default_queue']
        self._add_pipeline_tags = params['add_pipeline_tags']
        self._deserialize(pipeline_dag)

        # if we continue the pipeline, make sure that we re-execute failed tasks
        if params['continue_pipeline']:
            for node in self._nodes.values():
                if node.executed is False:
                    node.executed = None

        if not self._verify():
            raise ValueError("Failed verifying pipeline execution graph, "
                             "it has either inaccessible nodes, or contains cycles")

        self._update_execution_plot()

        if run_remotely:
            self._task.execute_remotely(queue_name='services' if not isinstance(run_remotely, str) else run_remotely)
            # we will not get here if we are not running remotely

        self._start_time = time()
        self._stop_event = Event()
        self._experiment_created_cb = step_task_created_callback
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()
        return True

    def stop(self, timeout=None):
        # type: (Optional[float]) -> ()
        """
        Stop the pipeline controller and the optimization thread.

        :param float timeout: Wait timeout for the optimization thread to exit (minutes).
            The default is ``None``, indicating do not wait terminate immediately.
        """
        pass

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
        return self._thread is not None

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

        :return: execution tree, as a nested dictionary
        Example:
            {
                'stage1' : Node() {
                    name: 'stage1'
                    job: TrainsJob
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
        self._nodes = {k: self.Node(name=k, **v) for k, v in dag_dict.items()}

    def _has_stored_configuration(self):
        """
        Return True if we are running remotely and we have stored configuration on the Task
        """
        if self._task and not self._task.running_locally() and self._task.is_main_task():
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
        Launch a single node (create and enqueue a TrainsJob)

        :param node: Node to launch
        :return: Return True if a new job was launched
        """
        if node.job or node.executed:
            return False

        updated_hyper_parameters = {}
        for k, v in node.parameters.items():
            updated_hyper_parameters[k] = self._parse_step_ref(v)

        node.job = TrainsJob(
            base_task_id=node.base_task_id, parameter_override=updated_hyper_parameters,
            tags=['pipe: {}'.format(self._task.id)] if self._add_pipeline_tags and self._task else None,
            parent=self._task.id if self._task else None)
        if self._experiment_created_cb:
            self._experiment_created_cb(node, updated_hyper_parameters)
        node.job.launch(queue_name=node.queue or self._default_execution_queue)
        return True

    def _update_execution_plot(self):
        # type: () -> ()
        """
        Update sankey diagram of the current pipeline
        """
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
                sankey_node['color'].append(
                    ("blue" if not node.job or not node.job.is_failed() else "red")
                    if node.executed is not None else ("green" if node.job else "lightsteelblue"))

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

        # create the detailed parameter table
        table_values = [["Pipeline Step", "Task ID", "Parameters"]]
        table_values += [
            [v, self._nodes[v].executed or (self._nodes[v].job.task_id() if self._nodes[v].job else ''), str(n)]
            for v, n in zip(visited, node_params)]

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

    def _force_task_configuration_update(self):
        pipeline_dag = self._serialize()
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

        while self._stop_event:
            # stop request
            if pooling_counter and self._stop_event.wait(self._pool_frequency):
                break

            pooling_counter += 1

            # check the pipeline time limit
            if self._pipeline_time_limit and (time() - self._start_time) > self._pipeline_time_limit:
                break

            # check the state of all current jobs
            # if no a job ended, continue
            completed_jobs = []
            for j in self._running_nodes:
                node = self._nodes[j]
                if not node.job:
                    continue
                if node.job.is_stopped():
                    completed_jobs.append(j)
                    node.executed = node.job.task_id() if not node.job.is_failed() else False
                elif node.timeout:
                    started = node.job.task.data.started
                    if (datetime.now().astimezone(started.tzinfo) - started).total_seconds() > node.timeout:
                        node.job.abort()
                        completed_jobs.append(j)
                        node.executed = node.job.task_id()

            # update running jobs
            self._running_nodes = [j for j in self._running_nodes if j not in completed_jobs]

            # nothing changed, we can sleep
            if not completed_jobs and self._running_nodes:
                continue

            # Pull the next jobs in the pipeline, based on the completed list
            next_nodes = []
            for node in self._nodes.values():
                # check if already processed.
                if node.job or node.executed:
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
                else:
                    getLogger('clearml.automation.controller').error(
                        'ERROR: Failed launching step \'{}\': {}'.format(name, self._nodes[name]))

            # update current state (in configuration, so that we could later continue an aborted pipeline)
            self._force_task_configuration_update()

            # visualize pipeline state (plot)
            self._update_execution_plot()

            # quit if all pipelines nodes are fully executed.
            if not next_nodes and not self._running_nodes:
                break

        # stop all currently running jobs:
        failing_pipeline = False
        for node in self._nodes.values():
            if node.executed is False:
                failing_pipeline = True
            if node.job and node.executed and not node.job.is_stopped():
                node.job.abort()

        if failing_pipeline and self._task:
            self._task.mark_failed(status_reason='Pipeline step failed')

        if self._stop_event:
            # noinspection PyBroadException
            try:
                self._stop_event.set()
            except Exception:
                pass

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
        if prev_step not in self._nodes or not self._nodes[prev_step].job:
            raise ValueError("Could not parse reference '{}', step {} could not be found".format(
                step_ref_string, prev_step))
        if input_type not in ('artifacts', 'parameters', 'models', 'id'):
            raise ValueError("Could not parse reference '{}', type {} not valid".format(step_ref_string, input_type))
        if input_type != 'id' and len(parts) < 3:
            raise ValueError("Could not parse reference '{}', missing fields in {}".format(step_ref_string, parts))

        task = self._nodes[prev_step].job.task if self._nodes[prev_step].job \
            else Task.get_task(task_id=self._nodes[prev_step].executed)
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
        return None

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
