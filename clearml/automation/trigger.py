import json
import logging
from datetime import datetime
from threading import enumerate as enumerate_threads
from typing import List, Optional, Dict, Union, Callable

from attr import attrs, attrib

from .job import ClearmlJob
from .scheduler import BaseScheduleJob, BaseScheduler, ExecutedJob
from ..task import Task
from ..backend_api.session.client import APIClient
from ..backend_interface.util import datetime_to_isoformat, datetime_from_isoformat


@attrs
class BaseTrigger(BaseScheduleJob):
    _only_fields = {"id", "name", "last_update", }
    _update_field = None

    project = attrib(default=None, type=str)
    match_name = attrib(default=None, type=str)
    tags = attrib(default=None, type=list)  # type: List[str]
    required_tags = attrib(default=None, type=list)  # type: List[str]
    add_tag = attrib(default=None, type=str)
    last_update = attrib(default=None, type=datetime, converter=datetime_from_isoformat)

    # remember the previous state of Ids answering the specific query
    # any new object.id returned that is not in the list, is a new event
    # we store a dict of {object_id: datetime}
    # allowing us to ignore repeating object updates triggering multiple times
    _triggered_instances = attrib(type=dict, default=None)  # type: Dict[str, datetime]

    def build_query(self, ref_time):
        return {
            'name': self.match_name or None,
            'project': [self.project] if self.project else None,
            'tags': ((self.tags or []) + (self.required_tags or [])) or None,
            self._update_field: ">{}".format(ref_time.isoformat() if ref_time else self.last_update.isoformat())
        }

    def verify(self):
        # type: () -> None
        super(BaseTrigger, self).verify()
        if self.tags and (not isinstance(self.tags, (list, tuple)) or
                          not all(isinstance(s, str) for s in self.tags)):
            raise ValueError("Tags must be a list of strings: {}".format(self.tags))
        if self.required_tags and (not isinstance(self.required_tags, (list, tuple)) or
                                   not all(isinstance(s, str) for s in self.required_tags)):
            raise ValueError("Required tags must be a list of strings: {}".format(self.required_tags))
        if self.project and not isinstance(self.project, str):
            raise ValueError("Project must be a string: {}".format(self.project))
        if self.match_name and not isinstance(self.match_name, str):
            raise ValueError("Match name must be a string: {}".format(self.match_name))

    def get_key(self):
        return getattr(self, '_key', None)


@attrs
class ModelTrigger(BaseTrigger):
    _task_param = '${model.id}'
    _key = "models"
    _only_fields = {"id", "name", "last_update", "ready", "tags"}
    _update_field = "last_update"

    on_publish = attrib(type=bool, default=None)
    on_archive = attrib(type=bool, default=None)

    def build_query(self, ref_time):
        query = super(ModelTrigger, self).build_query(ref_time)
        if self.on_publish:
            query.update({'ready': True})
        if self.on_archive:
            system_tags = list(set(query.get('system_tags', []) + ['archived']))
            query.update({'system_tags': system_tags})
        return query


@attrs
class DatasetTrigger(BaseTrigger):
    _task_param = '${dataset.id}'
    _key = "tasks"
    _only_fields = {"id", "name", "last_update", "status", "completed", "tags"}
    _update_field = "last_update"

    on_publish = attrib(type=bool, default=None)
    on_archive = attrib(type=bool, default=None)

    def build_query(self, ref_time):
        query = super(DatasetTrigger, self).build_query(ref_time)
        query.update({
            'system_tags': list(set(query.get('system_tags', []) + ['dataset'])),
            'task_types': list(set(query.get('task_types', []) + [str(Task.TaskTypes.data_processing)])),
            'status': ['published' if self.on_publish else 'completed']
        })

        if self.on_archive:
            system_tags = list(set(query.get('system_tags', []) + ['archived']))
            query.update({'system_tags': system_tags})

        return query


@attrs
class TaskTrigger(BaseTrigger):
    _task_param = '${task.id}'
    _key = "tasks"
    _only_fields = {"id", "name", "last_update", "status", "completed", "tags"}
    _update_field = "last_update"

    metrics = attrib(default=None, type=str)
    variant = attrib(default=None, type=str)
    threshold = attrib(default=None, type=float)
    value_sign = attrib(default=None, type=str)
    exclude_dev = attrib(default=None, type=bool)
    on_status = attrib(type=list, default=None)

    def build_query(self, ref_time):
        query = super(TaskTrigger, self).build_query(ref_time)
        if self.exclude_dev:
            system_tags = list(set(query.get('system_tags', []) + ['-development']))
            query.update({'system_tags': system_tags})
        if self.on_status:
            query.update({'status': self.on_status})
        if self.metrics and self.variant and self.threshold:
            metrics, title, series, values = ClearmlJob.get_metric_req_params(self.metrics, self.variant)
            sign_max = (self.value_sign or '').lower() in ('max', 'maximum')
            filter_key = "last_metrics.{}.{}.{}".format(title, series, "max_value" if sign_max else "min_value")
            filter_value = [self.threshold, None] if sign_max else [None, self.threshold]
            query.update({filter_key: filter_value})
        return query

    def verify(self):
        # type: () -> None
        super(TaskTrigger, self).verify()
        if (self.metrics or self.variant or self.threshold is not None) and \
                not (self.metrics and self.variant and self.threshold is not None):
            raise ValueError("You must provide metric/variant/threshold")
        valid_status = [str(s) for s in Task.TaskStatusEnum]
        if self.on_status and not all(s in valid_status for s in self.on_status):
            raise ValueError("You on_status contains invalid status value: {}".format(self.on_status))
        valid_signs = ['min', 'minimum', 'max', 'maximum']
        if self.value_sign and self.value_sign not in valid_signs:
            raise ValueError("Invalid value_sign `{}`, valid options are: {}".format(self.value_sign, valid_signs))


@attrs
class ExecutedTrigger(ExecutedJob):
    trigger = attrib(type=str, default=None)


class TriggerScheduler(BaseScheduler):
    """
    Trigger Task execution if an event happens in the system
    Examples:
        New model is published/tagged,
        New Dataset is created,
        General Task failed,
        Task metric below/above threshold, alert every X minutes
    """
    _datasets_section = "datasets"
    _models_section = "models"
    _tasks_section = "tasks"
    _state_section = "state"

    def __init__(
            self,
            pooling_frequency_minutes=3.0,
            sync_frequency_minutes=15,
            force_create_task_name=None,
            force_create_task_project=None
    ):
        # type: (float, float, Optional[str], Optional[str]) -> None
        """
        Create a Task trigger service

        :param pooling_frequency_minutes: Check for new events every X minutes (default 2)
        :param sync_frequency_minutes: Sync task scheduler configuration every X minutes.
        Allow to change scheduler in runtime by editing the Task configuration object
        :param force_create_task_name: Optional, force creation of Task Scheduler service,
        even if main Task.init already exists.
        :param force_create_task_project: Optional, force creation of Task Scheduler service,
        even if main Task.init already exists.
        """
        super(TriggerScheduler, self).__init__(
            sync_frequency_minutes=sync_frequency_minutes,
            force_create_task_name=force_create_task_name,
            force_create_task_project=force_create_task_project,
            pooling_frequency_minutes=pooling_frequency_minutes,
        )
        self._task_triggers = []
        self._dataset_triggers = []
        self._model_triggers = []
        self._executed_triggers = []
        self._client = None

    def add_model_trigger(
            self,
            schedule_task_id=None,  # type: Union[str, Task]
            schedule_queue=None,  # type: str
            schedule_function=None,  # type: Callable[[str], None]
            trigger_project=None,  # type: str
            trigger_name=None,  # type: Optional[str]
            trigger_on_publish=None,  # type: bool
            trigger_on_tags=None,  # type: Optional[List[str]]
            trigger_on_archive=None,  # type: bool
            trigger_required_tags=None,  # type: Optional[List[str]]
            name=None,  # type: Optional[str]
            target_project=None,  # type: Optional[str]
            add_tag=True,  # type: Union[bool, str]
            single_instance=False,  # type: bool
            reuse_task=False,  # type: bool
            task_parameters=None,  # type: Optional[dict]
            task_overrides=None,  # type: Optional[dict]
    ):
        # type: (...) -> None
        """
        Create a cron job alike scheduling for a pre existing Task or function.
        Trigger the Task/function execution on changes in the model repository
        Notice it is recommended to give the trigger a descriptive unique name, if not provided a task ID is used.

        Notice `task_overrides` can except reference to the trigger model ID:
        example: task_overrides={'Args/model_id': '${model.id}'}
        Notice if schedule_function is passed, use the following function interface:
        ```py
        def schedule_function(model_id):
            pass
        ```

        :param schedule_task_id: Task/task ID to be cloned and scheduled for execution
        :param schedule_queue: Queue name or ID to put the Task into (i.e. schedule)
        :param schedule_function: Optional, instead of providing Task ID to be scheduled,
            provide a function to be called. Notice the function is called from the scheduler context
            (i.e. running on the same machine as the scheduler)
        :param name: Name or description for the cron Task (should be unique if provided otherwise randomly generated)
        :param trigger_project: Only monitor models from this specific project (not recursive)
        :param trigger_name: Trigger only on models with name matching (regexp)
        :param trigger_on_publish: Trigger when model is published.
        :param trigger_on_tags: Trigger when all tags in the list are present
        :param trigger_on_archive: Trigger when model is archived
        :param trigger_required_tags: Trigger only on models with the following additional tags (must include all tags)
        :param target_project: Specify target project to put the cloned scheduled Task in.
        :param add_tag: Add tag to the executed Task. Provide specific tag (str) or
            pass True (default) to use the trigger name as tag
        :param single_instance: If True, do not launch the Task job if the previous instance is still running
        (skip until the next scheduled time period). Default False.
        :param reuse_task: If True, re-enqueue the same Task (i.e. do not clone it) every time, default False.
        :param task_parameters: Configuration parameters to the executed Task.
        for example: {'Args/batch': '12'} Notice: not available when reuse_task=True
        :param task_overrides: Change task definition.
        for example {'script.version_num': None, 'script.branch': 'main'} Notice: not available when reuse_task=True
        :return: True if job is successfully added to the scheduling list
        """
        trigger = ModelTrigger(
            base_task_id=schedule_task_id,
            base_function=schedule_function,
            queue=schedule_queue,
            name=name,
            target_project=target_project,
            single_instance=single_instance,
            task_parameters=task_parameters,
            task_overrides=task_overrides,
            add_tag=(add_tag if isinstance(add_tag, str) else (name or schedule_task_id)) if add_tag else None,
            clone_task=not bool(reuse_task),
            match_name=trigger_name,
            project=Task.get_project_id(trigger_project) if trigger_project else None,
            tags=trigger_on_tags,
            required_tags=trigger_required_tags,
            on_publish=trigger_on_publish,
            on_archive=trigger_on_archive,
        )
        trigger.verify()
        self._model_triggers.append(trigger)

    def add_dataset_trigger(
        self,
        schedule_task_id=None,  # type: Union[str, Task]
        schedule_queue=None,  # type: str
        schedule_function=None,  # type: Callable[[str], None]
        trigger_project=None,  # type: str
        trigger_name=None,  # type: Optional[str]
        trigger_on_publish=None,  # type: bool
        trigger_on_tags=None,  # type: Optional[List[str]]
        trigger_on_archive=None,  # type: bool
        trigger_required_tags=None,  # type: Optional[List[str]]
        name=None,  # type: Optional[str]
        target_project=None,  # type: Optional[str]
        add_tag=True,  # type: Union[bool, str]
        single_instance=False,  # type: bool
        reuse_task=False,  # type: bool
        task_parameters=None,  # type: Optional[dict]
        task_overrides=None,  # type: Optional[dict]
    ):
        # type: (...) -> None
        """
        Create a cron job alike scheduling for a pre existing Task or function.
        Trigger the Task/function execution on changes in the dataset repository (notice this is not the hyper-datasets)
        Notice it is recommended to give the trigger a descriptive unique name, if not provided a task ID is used.

        Notice `task_overrides` can except reference to the trigger model ID:
        example: task_overrides={'Args/dataset_id': '${dataset.id}'}
        Notice if schedule_function is passed, use the following function interface:
        ```py
        def schedule_function(dataset_id):
            pass
        ```

        :param schedule_task_id: Task/task ID to be cloned and scheduled for execution
        :param schedule_queue: Queue name or ID to put the Task into (i.e. schedule)
        :param schedule_function: Optional, instead of providing Task ID to be scheduled,
            provide a function to be called. Notice the function is called from the scheduler context
            (i.e. running on the same machine as the scheduler)
        :param name: Name or description for the cron Task (should be unique if provided otherwise randomly generated)
        :param trigger_project: Only monitor datasets from this specific project (not recursive)
        :param trigger_name: Trigger only on datasets with name matching (regexp)
        :param trigger_on_publish: Trigger when dataset is published.
        :param trigger_on_tags: Trigger when all tags in the list are present
        :param trigger_on_archive: Trigger when dataset is archived
        :param trigger_required_tags: Trigger only on datasets with the
            following additional tags (must include all tags)
        :param target_project: Specify target project to put the cloned scheduled Task in.
        :param add_tag: Add tag to the executed Task. Provide specific tag (str) or
            pass True (default) to use the trigger name as tag
        :param single_instance: If True, do not launch the Task job if the previous instance is still running
        (skip until the next scheduled time period). Default False.
        :param reuse_task: If True, re-enqueue the same Task (i.e. do not clone it) every time, default False.
        :param task_parameters: Configuration parameters to the executed Task.
        for example: {'Args/batch': '12'} Notice: not available when reuse_task=True/
        :param task_overrides: Change task definition.
        for example {'script.version_num': None, 'script.branch': 'main'} Notice: not available when reuse_task=True
        :return: True if job is successfully added to the scheduling list
        """
        if trigger_project:
            trigger_project_list = Task.get_projects(
                name="^{}/\\.datasets/.*".format(trigger_project), search_hidden=True, _allow_extra_fields_=True
            )
            for project in trigger_project_list:
                trigger = DatasetTrigger(
                    base_task_id=schedule_task_id,
                    base_function=schedule_function,
                    queue=schedule_queue,
                    name=name,
                    target_project=target_project,
                    single_instance=single_instance,
                    task_parameters=task_parameters,
                    task_overrides=task_overrides,
                    add_tag=(add_tag if isinstance(add_tag, str) else (name or schedule_task_id)) if add_tag else None,
                    clone_task=not bool(reuse_task),
                    match_name=trigger_name,
                    project=project.id,
                    tags=trigger_on_tags,
                    required_tags=trigger_required_tags,
                    on_publish=trigger_on_publish,
                    on_archive=trigger_on_archive,
                )
                trigger.verify()
                self._dataset_triggers.append(trigger)
        else:
            trigger = DatasetTrigger(
                base_task_id=schedule_task_id,
                base_function=schedule_function,
                queue=schedule_queue,
                name=name,
                target_project=target_project,
                single_instance=single_instance,
                task_parameters=task_parameters,
                task_overrides=task_overrides,
                add_tag=(add_tag if isinstance(add_tag, str) else (name or schedule_task_id)) if add_tag else None,
                clone_task=not bool(reuse_task),
                match_name=trigger_name,
                tags=trigger_on_tags,
                required_tags=trigger_required_tags,
                on_publish=trigger_on_publish,
                on_archive=trigger_on_archive,
            )
            trigger.verify()
            self._dataset_triggers.append(trigger)

    def add_task_trigger(
            self,
            schedule_task_id=None,  # type: Union[str, Task]
            schedule_queue=None,  # type: str
            schedule_function=None,  # type: Callable[[str], None]
            trigger_project=None,  # type: str
            trigger_name=None,  # type: Optional[str]
            trigger_on_tags=None,  # type: Optional[List[str]]
            trigger_on_status=None,  # type: Optional[List[str]]
            trigger_exclude_dev_tasks=None,  # type: Optional[bool]
            trigger_on_metric=None,  # type: Optional[str]
            trigger_on_variant=None,  # type: Optional[str]
            trigger_on_threshold=None,  # type: Optional[float]
            trigger_on_sign=None,  # type: Optional[str]
            trigger_required_tags=None,  # type: Optional[List[str]]
            name=None,  # type: Optional[str]
            target_project=None,  # type: Optional[str]
            add_tag=True,  # type: Union[bool, str]
            single_instance=False,  # type: bool
            reuse_task=False,  # type: bool
            task_parameters=None,  # type: Optional[dict]
            task_overrides=None,  # type: Optional[dict]
    ):
        # type: (...) -> None
        """
        Create a cron job alike scheduling for a pre existing Task or function.
        Trigger the Task/function execution on changes in the Task
        Notice it is recommended to give the trigger a descriptive unique name, if not provided a task ID is used.

        Notice `task_overrides` can except reference to the trigger model ID:
        example: task_overrides={'Args/task_id': '${task.id}'}
        Notice if schedule_function is passed, use the following function interface:
        ```py
        def schedule_function(task_id):
            pass
        ```

        :param schedule_task_id: Task/task ID to be cloned and scheduled for execution
        :param schedule_queue: Queue name or ID to put the Task into (i.e. schedule)
        :param schedule_function: Optional, instead of providing Task ID to be scheduled,
            provide a function to be called. Notice the function is called from the scheduler context
            (i.e. running on the same machine as the scheduler)
        :param name: Name or description for the cron Task (should be unique if provided otherwise randomly generated)
        :param trigger_project: Only monitor tasks from this specific project (not recursive)
        :param trigger_name: Trigger only on tasks with name matching (regexp)
        :param trigger_on_tags: Trigger when all tags in the list are present
        :param trigger_required_tags: Trigger only on tasks with the following additional tags (must include all tags)
        :param trigger_on_status: Trigger on Task status change.
            expect list of status strings, e.g. ['failed', 'published']
        :param trigger_exclude_dev_tasks: If True only trigger on Tasks executed by clearml-agent (and not manually)
        :param trigger_on_metric: Trigger on metric/variant above/under threshold (metric=title, variant=series)
        :param trigger_on_variant: Trigger on metric/variant above/under threshold (metric=title, variant=series)
        :param trigger_on_threshold: Trigger on metric/variant above/under threshold (float number)
        :param trigger_on_sign: possible values "max"/"maximum" or "min"/"minimum",
            trigger Task if metric below "min" or "above" maximum. Default: "minimum"
        :param target_project: Specify target project to put the cloned scheduled Task in.
        :param add_tag: Add tag to the executed Task. Provide specific tag (str) or
            pass True (default) to use the trigger name as tag
        :param single_instance: If True, do not launch the Task job if the previous instance is still running
        (skip until the next scheduled time period). Default False.
        :param reuse_task: If True, re-enqueue the same Task (i.e. do not clone it) every time, default False.
        :param task_parameters: Configuration parameters to the executed Task.
        for example: {'Args/batch': '12'} Notice: not available when reuse_task=True/
        :param task_overrides: Change task definition.
        for example {'script.version_num': None, 'script.branch': 'main'} Notice: not available when reuse_task=True
        :return: True if job is successfully added to the scheduling list
        """
        trigger = TaskTrigger(
            base_task_id=schedule_task_id,
            base_function=schedule_function,
            queue=schedule_queue,
            name=name,
            target_project=target_project,
            single_instance=single_instance,
            task_parameters=task_parameters,
            task_overrides=task_overrides,
            add_tag=(add_tag if isinstance(add_tag, str) else (name or schedule_task_id)) if add_tag else None,
            clone_task=not bool(reuse_task),
            match_name=trigger_name,
            project=Task.get_project_id(trigger_project) if trigger_project else None,
            tags=trigger_on_tags,
            required_tags=trigger_required_tags,
            on_status=trigger_on_status,
            exclude_dev=trigger_exclude_dev_tasks,
            metrics=trigger_on_metric,
            variant=trigger_on_variant,
            threshold=trigger_on_threshold,
            value_sign=trigger_on_sign
        )
        trigger.verify()
        self._task_triggers.append(trigger)

    def start(self):
        """
        Start the Task trigger loop (notice this function does not return)
        """
        super(TriggerScheduler, self).start()

    def get_triggers(self):
        # type: () -> List[BaseTrigger]
        """
        Return all triggers (models, datasets, tasks)
        :return: List of trigger objects
        """
        return self._model_triggers + self._dataset_triggers + self._task_triggers

    def _step(self):
        if not self._client:
            self._client = APIClient()

        executed = False
        for trigger in (self._model_triggers + self._dataset_triggers + self._task_triggers):
            ref_time = datetime_from_isoformat(trigger.last_update or datetime.utcnow())
            objects = []
            try:
                # noinspection PyProtectedMember
                objects = getattr(self._client, trigger.get_key()).get_all(
                    _allow_extra_fields_=True,
                    only_fields=list(trigger._only_fields or []),
                    **trigger.build_query(ref_time)
                )
                trigger.last_update = max([o.last_update for o in objects] or [ref_time])
                if not objects:
                    continue
            except Exception as ex:
                self._log("Exception occurred while checking trigger '{}' state: {}".format(trigger, ex))

            executed |= bool(objects)

            # actually handle trigger
            for obj in objects:
                # create a unique instance list
                if not trigger._triggered_instances:
                    trigger._triggered_instances = {}

                if obj.id in trigger._triggered_instances:
                    continue

                trigger._triggered_instances[obj.id] = datetime.utcnow()
                self._launch_job(trigger, obj.id)

        return executed

    # noinspection PyMethodOverriding
    def _launch_job(self, job, trigger_id):
        # type: (BaseTrigger, str) -> None
        if job.base_task_id:
            task_parameters = None
            if job.task_parameters:
                task_parameters = {
                    k: trigger_id if v == job._task_param else v # noqa
                    for k, v in job.task_parameters.items()
                }
            task_job = self._launch_job_task(
                job,
                task_parameters=task_parameters,
                add_tags=job.add_tag or None,
            )
            if task_job:
                self._executed_triggers.append(
                    ExecutedTrigger(
                        name=job.name,
                        task_id=task_job.task_id(),
                        started=datetime.utcnow(),
                        trigger=str(job.__class__.__name__))
                )
        if job.base_function:
            thread_job = self._launch_job_function(job, func_args=(trigger_id, ))
            if thread_job:
                self._executed_triggers.append(
                    ExecutedTrigger(
                        name=job.name,
                        thread_id=str(thread_job.ident),
                        started=datetime.utcnow(),
                        trigger=str(job.__class__.__name__))
                )

    def _serialize(self):
        # noinspection PyProtectedMember
        self._task._set_configuration(
            config_type='json',
            description='Dataset trigger configuration',
            config_text=json.dumps(
                [j.to_dict() for j in self._dataset_triggers], default=datetime_to_isoformat),
            name=self._datasets_section,
        )
        # noinspection PyProtectedMember
        self._task._set_configuration(
            config_type='json',
            description='Model trigger configuration',
            config_text=json.dumps(
                [j.to_dict() for j in self._model_triggers], default=datetime_to_isoformat),
            name=self._models_section,
        )
        # noinspection PyProtectedMember
        self._task._set_configuration(
            config_type='json',
            description='Task trigger configuration',
            config_text=json.dumps(
                [j.to_dict() for j in self._task_triggers], default=datetime_to_isoformat),
            name=self._tasks_section,
        )

    def _deserialize(self):
        self._task.reload()
        self._dataset_triggers = self.__deserialize_section(
            section=self._datasets_section, trigger_class=DatasetTrigger, current_triggers=self._dataset_triggers)
        self._model_triggers = self.__deserialize_section(
            section=self._models_section, trigger_class=ModelTrigger, current_triggers=self._model_triggers)
        self._task_triggers = self.__deserialize_section(
            section=self._tasks_section, trigger_class=TaskTrigger, current_triggers=self._task_triggers)

    def __deserialize_section(self, section, trigger_class, current_triggers):
        # noinspection PyProtectedMember
        json_str = self._task._get_configuration_text(name=section)
        try:
            return self.__deserialize_triggers(json.loads(json_str), trigger_class, current_triggers)
        except Exception as ex:
            self._log('Failed deserializing configuration: {}'.format(ex), level=logging.WARN)
            return current_triggers

    @staticmethod
    def __deserialize_triggers(trigger_jobs, trigger_class, current_triggers):
        # type:  (List[dict], BaseTrigger, List[BaseTrigger]) -> List[BaseTrigger]
        trigger_jobs = [trigger_class().update(j) for j in trigger_jobs]  # noqa

        trigger_jobs = {j.name: j for j in trigger_jobs}
        current_triggers = {j.name: j for j in current_triggers}

        # select only valid jobs, and update the valid ones state from the current one
        new_triggers = [
            current_triggers[name].update(j) if name in current_triggers else j
            for name, j in trigger_jobs.items()
        ]
        # verify all jobs
        for j in new_triggers:
            j.verify()

        return new_triggers

    def _serialize_state(self):
        json_str = json.dumps(
            dict(
                dataset_triggers=[j.to_dict(full=True) for j in self._dataset_triggers],
                model_triggers=[j.to_dict(full=True) for j in self._model_triggers],
                task_triggers=[j.to_dict(full=True) for j in self._task_triggers],
                # pooling_frequency_minutes=self._pooling_frequency_minutes,
                # sync_frequency_minutes=self._sync_frequency_minutes,
            ),
            default=datetime_to_isoformat
        )
        self._task.upload_artifact(
            name=self._state_section,
            artifact_object=json_str,
            preview='scheduler internal state'
        )

    def _deserialize_state(self):
        # get artifact
        self._task.reload()
        artifact_object = self._task.artifacts.get(self._state_section)
        if artifact_object is None:
            return
        state_json_str = artifact_object.get()
        if state_json_str is None:
            return

        state_dict = json.loads(state_json_str)
        self._dataset_triggers = self.__deserialize_triggers(
            state_dict.get('dataset_triggers', []),
            trigger_class=DatasetTrigger,  # noqa
            current_triggers=self._dataset_triggers
        )
        self._model_triggers = self.__deserialize_triggers(
            state_dict.get('model_triggers', []),
            trigger_class=ModelTrigger,  # noqa
            current_triggers=self._model_triggers
        )
        self._task_triggers = self.__deserialize_triggers(
            state_dict.get('task_triggers', []),
            trigger_class=TaskTrigger,  # noqa
            current_triggers=self._task_triggers
        )

    def _update_execution_plots(self):
        # type: () -> None
        if not self._task:
            return

        task_link_template = self._task.get_output_log_web_page() \
            .replace('/{}/'.format(self._task.project), '/{project}/') \
            .replace('/{}/'.format(self._task.id), '/{task}/')

        # plot the already executed Tasks
        executed_table = [['trigger', 'name', 'task id', 'started', 'finished']]
        for executed_job in sorted(self._executed_triggers, key=lambda x: x.started, reverse=True):
            if not executed_job.finished:
                if executed_job.task_id:
                    t = Task.get_task(task_id=executed_job.task_id)
                    if t.status not in ('in_progress', 'queued'):
                        executed_job.finished = t.data.completed or datetime.utcnow()
                elif executed_job.thread_id:
                    # noinspection PyBroadException
                    try:
                        a_thread = [t for t in enumerate_threads() if t.ident == executed_job.thread_id]
                        if not a_thread or not a_thread[0].is_alive():
                            executed_job.finished = datetime.utcnow()
                    except Exception:
                        pass

            executed_table += [
                [executed_job.trigger,
                 executed_job.name,
                 '<a href="{}">{}</a>'.format(task_link_template.format(
                     project='*', task=executed_job.task_id), executed_job.task_id)
                 if executed_job.task_id else 'function',
                 str(executed_job.started).split('.', 1)[0],
                 str(executed_job.finished).split('.', 1)[0]
                 ]
            ]

        # plot the schedule definition
        self._task.get_logger().report_table(
            title='Triggers Executed', series=' ', iteration=0,
            table_plot=executed_table
        )
        self.__report_trigger_table(triggers=self._model_triggers, title='Model Triggers')
        self.__report_trigger_table(triggers=self._dataset_triggers, title='Dataset Triggers')
        self.__report_trigger_table(triggers=self._task_triggers, title='Task Triggers')

    def __report_trigger_table(self, triggers, title):
        if not triggers:
            return

        task_link_template = self._task.get_output_log_web_page() \
            .replace('/{}/'.format(self._task.project), '/{project}/') \
            .replace('/{}/'.format(self._task.id), '/{task}/')

        columns = [k for k in BaseTrigger().__dict__.keys() if not k.startswith('_')]
        columns += [k for k in triggers[0].__dict__.keys() if k not in columns and not k.startswith('_')]

        column_task_id = columns.index('base_task_id')

        scheduler_table = [columns]
        for j in triggers:
            j_dict = j.to_dict()
            j_dict['base_function'] = "{}.{}".format(
                getattr(j.base_function, '__module__', ''),
                getattr(j.base_function, '__name__', '')
            ) if j.base_function else ''

            if not j_dict.get('base_task_id'):
                j_dict['clone_task'] = ''

            row = [
                str(j_dict.get(c)).split('.', 1)[0] if isinstance(j_dict.get(c), datetime) else str(j_dict.get(c) or '')
                for c in columns
            ]
            if row[column_task_id]:
                row[column_task_id] = '<a href="{}">{}</a>'.format(
                    task_link_template.format(project='*', task=row[column_task_id]), row[column_task_id])
            scheduler_table += [row]

        self._task.get_logger().report_table(
            title=title, series=' ', iteration=0,
            table_plot=scheduler_table
        )
