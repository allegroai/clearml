import json
from datetime import datetime
from time import sleep, time
from typing import List, Union, Optional

from attr import attrs, attrib
from dateutil.relativedelta import relativedelta

from .job import ClearmlJob
from ..backend_interface.util import datetime_from_isoformat, datetime_to_isoformat
from ..task import Task


@attrs
class ScheduleJob(object):
    _weekdays = ('sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday')

    name = attrib(type=str)
    base_task_id = attrib(type=str)
    queue = attrib(type=str, default=None)
    target_project = attrib(type=str, default=None)
    execution_limit_hours = attrib(type=float, default=None)
    recurring = attrib(type=bool, default=True)
    starting_time = attrib(type=datetime, converter=datetime_from_isoformat, default=None)
    single_instance = attrib(type=bool, default=False)
    task_parameters = attrib(type=dict, default={})
    task_overrides = attrib(type=dict, default={})
    clone_task = attrib(type=bool, default=True)
    minute = attrib(type=float, default=None)
    hour = attrib(type=float, default=None)
    day = attrib(default=None)
    month = attrib(type=float, default=None)
    year = attrib(type=float, default=None)
    _executed_instances = attrib(type=list, default=[])
    _next_run = attrib(type=datetime, converter=datetime_from_isoformat, default=None)
    _last_executed = attrib(type=datetime, converter=datetime_from_isoformat, default=None)
    _execution_timeout = attrib(type=datetime, converter=datetime_from_isoformat, default=None)

    def to_dict(self, full=False):
        return {k: v for k, v in self.__dict__.items() if full or not str(k).startswith('_')}

    def update(self, a_job):
        for k, v in a_job.to_dict().items():
            setattr(self, k, v)
        return self

    def next_run(self):
        # type () -> Optional[datetime]
        return self._next_run

    def get_execution_timeout(self):
        # type () -> Optional[datetime]
        return self._execution_timeout

    def get_last_executed_task_id(self):
        # type () -> Optional[str]
        return self._executed_instances[-1] if self._executed_instances else None

    def next(self):
        # type () -> Optional[datetime]
        """
        :return: Return the next run datetime, None if no scheduling needed
        """
        if not self.recurring and self._last_executed:
            self._next_run = None
            return self._next_run

        # make sure we have a starting time
        if not self.starting_time:
            self.starting_time = datetime.utcnow()

        # check if we have a specific date
        if self.year and self.year > 2000:
            # this is by definition a single execution only
            if self._last_executed:
                return None

            self._next_run = datetime(
                year=int(self.year),
                month=int(self.month or 1),
                day=int(self.day or 1),
                hour=int(self.hour or 0),
                minute=int(self.minute or 0)
            )
            return self._next_run

        weekday = self.day if isinstance(self.day, str) and self.day.lower() in self._weekdays else None
        if weekday:
            weekday = self._weekdays.index(weekday)

        # check if we have a specific day of the week
        prev_timestamp = self._last_executed or self.starting_time
        # make sure that if we have a specific day we zero the minutes/hours/seconds
        if self.year:
            prev_timestamp = datetime(
                year=prev_timestamp.year,
                month=self.month or prev_timestamp.month,
                day=self.day or prev_timestamp.day,
            )
        elif self.month:
            prev_timestamp = datetime(
                year=prev_timestamp.year,
                month=prev_timestamp.month,
                day=self.day or prev_timestamp.day,
            )
        elif self.day or weekday is not None:
            prev_timestamp = datetime(
                year=prev_timestamp.year,
                month=prev_timestamp.month,
                day=prev_timestamp.day if weekday is None else 1,
            )

        self._next_run = prev_timestamp + relativedelta(
            years=self.year or 0,
            months=0 if self.year else (self.month or 0),
            days=0 if self.month or self.year else ((self.day or 0) if weekday is None else 0),
            hours=self.hour or 0,
            minutes=self.minute or 0,
            weekday=weekday,
        )
        return self._next_run

    def run(self, task_id):
        # type (Optional[str]) -> datetime
        self._last_executed = datetime.utcnow()
        if task_id:
            self._executed_instances.append(str(task_id))
        if self.execution_limit_hours and task_id:
            self._execution_timeout = self._last_executed + relativedelta(
                hours=int(self.execution_limit_hours),
                minutes=int((self.execution_limit_hours - int(self.execution_limit_hours)) * 60)
            )
        else:
            self._execution_timeout = None
        return self._last_executed


@attrs
class ExecutedJob(object):
    name = attrib(type=str)
    task_id = attrib(type=str)
    started = attrib(type=datetime, converter=datetime_from_isoformat)
    finished = attrib(type=datetime, converter=datetime_from_isoformat, default=None)

    def to_dict(self, full=False):
        return {k: v for k, v in self.__dict__.items() if full or not str(k).startswith('_')}


class TaskScheduler(object):
    """
    Task Scheduling controller.
    Notice time-zone is ALWAYS UTC
    """
    _configuration_section = 'schedule'

    def __init__(self, sync_frequency_minutes=15, force_create_task_name=None, force_create_task_project=None):
        # type: (float, Optional[str], Optional[str]) -> None
        """
        Create a Task scheduler service

        :param sync_frequency_minutes: Sync task scheduler configuration every X minutes.
        Allow to change scheduler in runtime by editing the Task configuration object
        :param force_create_task_name: Optional, force creation of Task Scheduler service,
        even if main Task.init already exists.
        :param force_create_task_project: Optional, force creation of Task Scheduler service,
        even if main Task.init already exists.
        """
        self._last_sync = 0
        self._sync_frequency_minutes = sync_frequency_minutes
        self._schedule_jobs = []  # List[ScheduleJob]
        self._timeout_jobs = {}  # Dict[str, datetime]
        self._executed_jobs = []  # List[ExecutedJob]
        self._thread = None
        if force_create_task_name or not Task.current_task():
            self._task = Task.init(
                project_name=force_create_task_project or 'DevOps',
                task_name=force_create_task_name or 'Scheduler',
                task_type=Task.TaskTypes.service,
                auto_resource_monitoring=False,
            )
        else:
            self._task = Task.current_task()

    def add_task(
            self,
            task_id,  # type(Union[str, Task])
            queue,  # type(str)
            name=None,  # type(Optional[str])
            target_project=None,  # type(Optional[str])
            minute=None,  # type(Optional[float])
            hour=None,  # type(Optional[float])
            day=None,  # type(Optional[Union[float, str]])
            month=None,  # type(Optional[float])
            year=None,  # type(Optional[float])
            limit_execution_time=None,  # type(Optional[float])
            single_instance=False,  # type(bool)
            recurring=True,  # type(bool)
            reuse_task=False,  # type(bool)
            task_parameters=None,  # type(Optional[dict])
            task_overrides=None,  # type(Optional[dict])
    ):
        # type(...) -> bool
        """
        Create a cron job alike scheduling for a pre existing Task.
        Notice it is recommended to give the Task a descriptive name, if not provided a random UUID is used.
        Examples:
        Launch every 15 minutes
        add_task(task_id='1235', queue='default', minute=15)
        Launch every 1 hour
        add_task(task_id='1235', queue='default', hour=1)
        Launch every 1 hour and a half
        add_task(task_id='1235', queue='default', hour=1.5)
        Launch every day at 22:30 (10:30 pm)
        add_task(task_id='1235', queue='default', minute=30, hour=22, day=1)
        Launch every other day at 7:30 (7:30 am)
        add_task(task_id='1235', queue='default', minute=30, hour=7, day=2)
        Launch every Saturday at 8:30am
        add_task(task_id='1235', queue='default', minute=30, hour=8, day='saturday')
        Launch once a month at the 5th of each month
        add_task(task_id='1235', queue='default', month=1, day=5)
        Launch once a year on March 4th of each month
        add_task(task_id='1235', queue='default', year=1, month=3, day=4)

        :param task_id: Task/task ID to be cloned and scheduled for execution
        :param queue: Queue name or ID to put the Task into (i.e. schedule)
        :param name: Name or description for the cron Task (should be unique if provided otherwise randomly generated)
        :param target_project: Specify target project to put the cloned scheduled Task in.
        :param minute: If specified launch Task at a specific minute of the day
        :param hour: If specified launch Task at a specific hour (24h) of the day
        :param day: If specified launch Task at a specific day
        :param month: If specified launch Task at a specific month
        :param year: If specified launch Task at a specific year
        :param limit_execution_time: Limit the execution time (in hours) of the specific job.
        :param single_instance: If True, do not launch the Task job if the previous instance is still running
        (skip until the next scheduled time period). Default False.
        :param recurring: If False only launch the Task once (default: True, repeat)
        :param reuse_task: If True, re-enqueue the same Task (i.e. do not clone it) every time, default False.
        :param task_parameters: Configuration parameters to the executed Task.
        for example: {'Args/batch': '12'} Notice: not available when reuse_task=True/
        :param task_overrides: Change task definition.
        for example {'script.version_num': None, 'script.branch': 'main'} Notice: not available when reuse_task=True

        :return: True if job is successfully added to the scheduling list
        """
        task_id = task_id.id if isinstance(task_id, Task) else str(task_id)
        # noinspection PyProtectedMember
        job = ScheduleJob(
            name=name or task_id,
            base_task_id=task_id,
            queue=queue,
            execution_limit_hours=limit_execution_time,
            recurring=bool(recurring),
            single_instance=bool(single_instance),
            task_parameters=task_parameters,
            task_overrides=task_overrides,
            clone_task=not bool(reuse_task),
            starting_time=datetime.utcnow(),
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            year=year,
        )
        # make sure the queue is valid
        if not job.queue:
            self._log('Queue [], could not be found, skipping scheduled task'.format(queue), level='warning')
            return False

        self._schedule_jobs.append(job)
        return True

    def get_scheduled_tasks(self):
        # type: () -> List[ScheduleJob]
        """
        Return the current set of scheduled jobs

        :return: List of ScheduleJob instances
        """
        return self._schedule_jobs

    def remove_task(self, task_id):
        # type: (Union[str, Task]) -> bool
        """
        Remove a Task ID from the scheduled task list.

        :param task_id: Task or Task ID to be removed
        :return: return True of the Task ID was found in the scheduled jobs list and was removed.
        """
        task_id = task_id.id if isinstance(task_id, Task) else str(task_id)
        if not any(t.base_task_id == task_id for t in self._schedule_jobs):
            return False
        self._schedule_jobs = [t.base_task_id != task_id for t in self._schedule_jobs]
        return True

    def start(self):
        # type: () -> None
        """
        Start the Task TaskScheduler loop (notice this function does not return)
        """
        if Task.running_locally():
            self._serialize_state()
            self._serialize()
        else:
            self._deserialize_state()
            self._deserialize()

        while True:
            try:
                self._step()
            except Exception as ex:
                self._log('Warning: Exception caught during scheduling step: {}'.format(ex))
                # rate control
                sleep(15)

    def _step(self):
        # type: () -> None
        """
        scheduling processing step
        """
        # sync with backend
        if time() - self._last_sync > 60. * self._sync_frequency_minutes:
            self._last_sync = time()
            self._deserialize()
            self._update_execution_plots()

        # update next execution datetime
        for j in self._schedule_jobs:
            j.next()

        # get idle timeout (aka sleeping)
        scheduled_jobs = sorted(
            [j for j in self._schedule_jobs if j.next_run() is not None],
            key=lambda x: x.next_run()
        )
        timeout_jobs = sorted(list(self._timeout_jobs.values()))
        if not scheduled_jobs and not timeout_jobs:
            # sleep and retry
            seconds = 60. * self._sync_frequency_minutes
            self._log('Nothing to do, sleeping for {:.2f} minutes.'.format(seconds / 60.))
            sleep(seconds)
            return

        next_time_stamp = scheduled_jobs[0].next_run() if scheduled_jobs else None
        if timeout_jobs:
            next_time_stamp = min(timeout_jobs[0], next_time_stamp) \
                if next_time_stamp else timeout_jobs[0]

        sleep_time = (next_time_stamp - datetime.utcnow()).total_seconds()
        if sleep_time > 0:
            # sleep until we need to run a job or maximum sleep time
            seconds = min(sleep_time, 60. * self._sync_frequency_minutes)
            self._log('Waiting for next run, sleeping for {:.2f} minutes, until next sync.'.format(seconds / 60.))
            sleep(seconds)
            return

        # check if this is a Task timeout check
        if timeout_jobs and next_time_stamp == timeout_jobs[0]:
            # mark aborted
            task_id = [k for k, v in self._timeout_jobs.items() if v == timeout_jobs[0]][0]
            self._cancel_task(task_id=task_id)
            self._timeout_jobs.pop(task_id, None)
        else:
            job = scheduled_jobs[0]
            # check if this is a single instance, then we need to abort the Task
            if job.single_instance and job.get_last_executed_task_id():
                t = Task.get_task(task_id=job.get_last_executed_task_id())
                if t.status in ('in_progress', 'queued'):
                    self._log(
                        'Skipping Task {} scheduling, previous Task instance {} still running'.format(
                            job.name, t.id
                        ))
                    job.run(None)
                    return

            # actually run the job
            task_job = ClearmlJob(
                base_task_id=job.base_task_id,
                parameter_override=job.task_parameters,
                task_overrides=job.task_overrides,
                disable_clone_task=not job.clone_task,
                allow_caching=False,
                target_project=job.target_project,
            )
            self._log('Scheduling Job {}, Task {} on queue {}.'.format(
                job.name, task_job.task_id(), job.queue))
            if task_job.launch(queue_name=job.queue):
                # mark as run
                job.run(task_job.task_id())
                self._executed_jobs.append(ExecutedJob(
                    name=job.name, task_id=task_job.task_id(), started=datetime.utcnow()))
                # add timeout check
                if job.get_execution_timeout():
                    # we should probably make sure we are not overwriting a Task
                    self._timeout_jobs[job.get_execution_timeout()] = task_job.task_id()

        self._update_execution_plots()

    def start_remotely(self, queue='services'):
        # type: (str) -> None
        """
        Start the Task TaskScheduler loop (notice this function does not return)

        :param queue: Remote queue to run the scheduler on, default 'services' queue.
        """
        self._task.execute_remotely(queue_name=queue, exit_process=True)
        self.start()

    def _serialize(self):
        # type: () -> None
        """
        Serialize Task scheduling configuration only (no internal state)
        """
        # noinspection PyProtectedMember
        self._task._set_configuration(
            config_type='json',
            description='schedule tasks configuration',
            config_text=self._serialize_schedule_into_string(),
            name=self._configuration_section,
        )

    def _serialize_state(self):
        # type: () -> None
        """
        Serialize internal state only
        """
        json_str = json.dumps(
            dict(
                scheduled_jobs=[j.to_dict(full=True) for j in self._schedule_jobs],
                timeout_jobs=self._timeout_jobs,
                executed_jobs=[j.to_dict(full=True) for j in self._executed_jobs],
            ),
            default=datetime_to_isoformat
        )
        self._task.upload_artifact(
            name='state',
            artifact_object=json_str,
            preview='scheduler internal state'
        )

    def _deserialize_state(self):
        # type: () -> None
        """
        Deserialize internal state only
        """
        # get artifact
        self._task.reload()
        artifact_object = self._task.artifacts.get('state')
        if artifact_object is not None:
            state_json_str = artifact_object.get()
            if state_json_str is not None:
                state_dict = json.loads(state_json_str)
                self._schedule_jobs = [ScheduleJob(**j) for j in state_dict.get('scheduled_jobs', [])]
                self._timeout_jobs = state_dict.get('timeout_jobs') or {}
                self._executed_jobs = [ExecutedJob(**j) for j in state_dict.get('executed_jobs', [])]

    def _deserialize(self):
        # type: () -> None
        """
        Deserialize Task scheduling configuration only
        """
        self._log('Syncing scheduler')
        self._task.reload()
        # noinspection PyProtectedMember
        json_str = self._task._get_configuration_text(name=self._configuration_section)
        try:
            scheduled_jobs = [ScheduleJob(**j) for j in json.loads(json_str)]
        except Exception as ex:
            self._log('Failed deserializing configuration: {}'.format(ex), level='warning')
            return

        scheduled_jobs = {j.name: j for j in scheduled_jobs}
        current_scheduled_jobs = {j.name: j for j in self._schedule_jobs}

        # select only valid jobs, and update the valid ones state from the current one
        self._schedule_jobs = [
            current_scheduled_jobs[name].update(j) if name in current_scheduled_jobs else j
            for name, j in scheduled_jobs.items()
        ]

    def _serialize_schedule_into_string(self):
        # type: () -> str
        return json.dumps([j.to_dict() for j in self._schedule_jobs], default=datetime_to_isoformat)

    def _update_execution_plots(self):
        # type: () -> None
        """
        Update the configuration and execution table plots
        """
        if not self._task:
            return

        task_link_template = self._task.get_output_log_web_page() \
            .replace('/{}/'.format(self._task.project), '/{project}/') \
            .replace('/{}/'.format(self._task.id), '/{task}/')

        # plot the schedule definition
        columns = [
            'name', 'base_task_id', 'target_project', 'queue',
            'minute', 'hour', 'day', 'month', 'year',
            'starting_time', 'execution_limit_hours', 'recurring',
            'single_instance', 'task_parameters', 'task_overrides', 'clone_task',
        ]
        scheduler_table = [columns]
        for j in self._schedule_jobs:
            j_dict = j.to_dict()
            row = [
                str(j_dict.get(c)).split('.', 1)[0] if isinstance(j_dict.get(c), datetime) else str(j_dict.get(c))
                for c in columns
            ]
            row[1] = '<a href="{}">{}</a>'.format(
                    task_link_template.format(project='*', task=row[1]), row[1])
            scheduler_table += [row]

        # plot the already executed Tasks
        executed_table = [['name', 'task id', 'started', 'finished']]
        for executed_job in sorted(self._executed_jobs, key=lambda x: x.started, reverse=True):
            if not executed_job.finished:
                t = Task.get_task(task_id=executed_job.task_id)
                if t.status not in ('in_progress', 'queued'):
                    executed_job.finished = t.data.completed or datetime.utcnow()
            executed_table += [
                [executed_job.name,
                 '<a href="{}">{}</a>'.format(
                    task_link_template.format(project='*', task=executed_job.task_id), executed_job.task_id),
                 str(executed_job.started).split('.', 1)[0], str(executed_job.finished).split('.', 1)[0]
                 ]
            ]

        self._task.get_logger().report_table(
            title='Schedule Tasks', series=' ', iteration=0,
            table_plot=scheduler_table
        )
        self._task.get_logger().report_table(
            title='Executed Tasks', series=' ', iteration=0,
            table_plot=executed_table
        )

    def _log(self, message, level=None):
        if self._task:
            self._task.get_logger().report_text(message)
        else:
            print(message)

    @staticmethod
    def _cancel_task(task_id):
        # type: (str) -> ()
        t = Task.get_task(task_id=task_id)
        status = t.status
        if status in ('in_progress',):
            t.stopped(force=True)
        elif status in ('queued',):
            Task.dequeue(t)

