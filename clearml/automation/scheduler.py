import json
import logging
from datetime import datetime
from threading import Thread, enumerate as enumerate_threads
from time import sleep, time
from typing import List, Union, Optional, Callable, Sequence, Dict

from attr import attrs, attrib
from dateutil.relativedelta import relativedelta

from .job import ClearmlJob
from ..backend_interface.util import datetime_from_isoformat, datetime_to_isoformat, mutually_exclusive
from ..task import Task


@attrs
class BaseScheduleJob(object):
    name = attrib(type=str, default=None)
    base_task_id = attrib(type=str, default=None)
    base_function = attrib(type=Callable, default=None)
    queue = attrib(type=str, default=None)
    target_project = attrib(type=str, default=None)
    single_instance = attrib(type=bool, default=False)
    task_parameters = attrib(type=dict, default={})
    task_overrides = attrib(type=dict, default={})
    clone_task = attrib(type=bool, default=True)
    _executed_instances = attrib(type=list, default=None)

    def to_dict(self, full=False):
        return {k: v for k, v in self.__dict__.items()
                if not callable(v) and (full or not str(k).startswith('_'))}

    def update(self, a_job):
        # type: (Union[Dict, BaseScheduleJob]) -> BaseScheduleJob
        converters = {a.name: a.converter for a in getattr(self, '__attrs_attrs__', [])}
        for k, v in (a_job.to_dict(full=True) if not isinstance(a_job, dict) else a_job).items():
            if v is not None and not callable(getattr(self, k, v)):
                setattr(self, k, converters[k](v) if converters.get(k) else v)
        return self

    def verify(self):
        # type: () -> None
        if self.base_function and not self.name:
            raise ValueError("Entry 'name' must be supplied for function scheduling")
        if self.base_task_id and not self.queue:
            raise ValueError("Target 'queue' must be provided for function scheduling")
        if not self.base_function and not self.base_task_id:
            raise ValueError("Either schedule function or task-id must be provided")

    def get_last_executed_task_id(self):
        # type: () -> Optional[str]
        return self._executed_instances[-1] if self._executed_instances else None

    def run(self, task_id):
        # type: (Optional[str]) -> None
        if task_id:
            # make sure we have a new instance
            if not self._executed_instances:
                self._executed_instances = []
            self._executed_instances.append(str(task_id))


@attrs
class ScheduleJob(BaseScheduleJob):
    _weekdays_ind = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')

    execution_limit_hours = attrib(type=float, default=None)
    recurring = attrib(type=bool, default=True)
    starting_time = attrib(type=datetime, converter=datetime_from_isoformat, default=None)
    minute = attrib(type=float, default=None)
    hour = attrib(type=float, default=None)
    day = attrib(default=None)
    weekdays = attrib(default=None)
    month = attrib(type=float, default=None)
    year = attrib(type=float, default=None)
    _next_run = attrib(type=datetime, converter=datetime_from_isoformat, default=None)
    _execution_timeout = attrib(type=datetime, converter=datetime_from_isoformat, default=None)
    _last_executed = attrib(type=datetime, converter=datetime_from_isoformat, default=None)

    def verify(self):
        # type: () -> None
        def check_integer(value):
            try:
                return False if not isinstance(value, (int, float)) or \
                               int(value) != float(value) else True
            except (TypeError, ValueError):
                return False

        super(ScheduleJob, self).verify()
        if self.weekdays and self.day not in (None, 0, 1):
            raise ValueError("`weekdays` and `day` combination is not valid (day must be None,0 or 1)")
        if self.weekdays and any(w not in self._weekdays_ind for w in self.weekdays):
            raise ValueError("`weekdays` must be a list of strings, valid values are: {}".format(self._weekdays_ind))
        if not (self.minute or self.hour or self.day or self.month or self.year):
            raise ValueError("Schedule time/date was not provided")
        if self.minute and not check_integer(self.minute):
            raise ValueError("Schedule `minute` must be an integer")
        if self.hour and not check_integer(self.hour):
            raise ValueError("Schedule `hour` must be an integer")
        if self.day and not check_integer(self.day):
            raise ValueError("Schedule `day` must be an integer")
        if self.month and not check_integer(self.month):
            raise ValueError("Schedule `month` must be an integer")
        if self.year and not check_integer(self.year):
            raise ValueError("Schedule `year` must be an integer")

    def next_run(self):
        # type: () -> Optional[datetime]
        return self._next_run

    def get_execution_timeout(self):
        # type: () -> Optional[datetime]
        return self._execution_timeout

    def next(self):
        # type: () -> Optional[datetime]
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
            if self.weekdays:
                self._next_run += relativedelta(weekday=self.get_weekday_ord(self.weekdays[0]))

            return self._next_run

        weekday = None
        if self.weekdays:
            # get previous weekday
            _weekdays = [self.get_weekday_ord(w) for w in self.weekdays]
            try:
                prev_weekday_ind = _weekdays.index(self._last_executed.weekday()) if self._last_executed else -1
            except ValueError:
                # in case previous execution was not in the weekday (for example executed immediately at scheduling)
                prev_weekday_ind = -1
            weekday = _weekdays[(prev_weekday_ind+1) % len(_weekdays)]

        # check if we have a specific day of the week
        prev_timestamp = self._last_executed or self.starting_time
        # make sure that if we have a specific day we zero the minutes/hours/seconds
        if self.year:
            prev_timestamp = datetime(
                year=prev_timestamp.year,
                month=self.month or prev_timestamp.month,
                day=self.day or 1,
            )
        elif self.month:
            prev_timestamp = datetime(
                year=prev_timestamp.year,
                month=prev_timestamp.month,
                day=self.day or 1,
            )
        elif self.day is None and weekday is not None:
            # notice we assume every X hours on specific weekdays
            # other combinations (i.e. specific time at weekdays, is covered later)
            next_timestamp = datetime(
                year=prev_timestamp.year,
                month=prev_timestamp.month,
                day=prev_timestamp.day,
                hour=prev_timestamp.hour,
                minute=prev_timestamp.minute,
            )
            next_timestamp += relativedelta(
                years=self.year or 0,
                months=0 if self.year else (self.month or 0),
                hours=self.hour or 0,
                minutes=self.minute or 0,
                weekday=weekday if not self._last_executed else None,
            )
            # start a new day
            if next_timestamp.day != prev_timestamp.day:
                next_timestamp = datetime(
                    year=prev_timestamp.year,
                    month=prev_timestamp.month,
                    day=prev_timestamp.day,
                ) + relativedelta(
                    years=self.year or 0,
                    months=0 if self.year else (self.month or 0),
                    hours=self.hour or 0,
                    minutes=self.minute or 0,
                    weekday=weekday
                )
            self._next_run = next_timestamp
            return self._next_run
        elif self.day is not None and weekday is not None:
            # push to the next day (so we only have once a day)
            prev_timestamp = datetime(
                year=prev_timestamp.year,
                month=prev_timestamp.month,
                day=prev_timestamp.day,
            ) + relativedelta(days=1)
        elif self.day:
            # reset minutes in the hour (we will be adding additional hour anyhow
            prev_timestamp = datetime(
                year=prev_timestamp.year,
                month=prev_timestamp.month,
                day=prev_timestamp.day,
            )
        elif self.hour:
            # reset minutes in the hour (we will be adding additional hour anyhow
            prev_timestamp = datetime(
                year=prev_timestamp.year,
                month=prev_timestamp.month,
                day=prev_timestamp.day,
                hour=prev_timestamp.hour,
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
        # type: (Optional[str]) -> datetime
        super(ScheduleJob, self).run(task_id)
        self._last_executed = datetime.utcnow()
        if self.execution_limit_hours and task_id:
            self._execution_timeout = self._last_executed + relativedelta(
                hours=int(self.execution_limit_hours),
                minutes=int((self.execution_limit_hours - int(self.execution_limit_hours)) * 60)
            )
        else:
            self._execution_timeout = None
        return self._last_executed

    @classmethod
    def get_weekday_ord(cls, weekday):
        # type: (Union[int, str]) -> int
        if isinstance(weekday, int):
            return min(6, max(weekday, 0))
        return cls._weekdays_ind.index(weekday)


@attrs
class ExecutedJob(object):
    name = attrib(type=str, default=None)
    started = attrib(type=datetime, converter=datetime_from_isoformat, default=None)
    finished = attrib(type=datetime, converter=datetime_from_isoformat, default=None)
    task_id = attrib(type=str, default=None)
    thread_id = attrib(type=str, default=None)

    def to_dict(self, full=False):
        return {k: v for k, v in self.__dict__.items() if full or not str(k).startswith('_')}


class BaseScheduler(object):
    def __init__(
            self,
            sync_frequency_minutes=15,
            force_create_task_name=None,
            force_create_task_project=None,
            pooling_frequency_minutes=None
    ):
        # type: (float, Optional[str], Optional[str], Optional[float]) -> None
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
        self._pooling_frequency_minutes = pooling_frequency_minutes
        self._sync_frequency_minutes = sync_frequency_minutes
        if force_create_task_name or not Task.current_task():
            self._task = Task.init(
                project_name=force_create_task_project or 'DevOps',
                task_name=force_create_task_name or 'Scheduler',
                task_type=Task.TaskTypes.service,
                auto_resource_monitoring=False,
            )
        else:
            self._task = Task.current_task()

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
            # sync with backend
            try:
                if time() - self._last_sync > 60. * self._sync_frequency_minutes:
                    self._last_sync = time()
                    self._deserialize()
                    self._update_execution_plots()
            except Exception as ex:
                self._log('Warning: Exception caught during deserialization: {}'.format(ex))
                self._last_sync = time()

            try:
                if self._step():
                    self._serialize_state()
                    self._update_execution_plots()
            except Exception as ex:
                self._log('Warning: Exception caught during scheduling step: {}'.format(ex))
                # rate control
                sleep(15)

            # sleep until the next pool (default None)
            if self._pooling_frequency_minutes:
                self._log("Sleeping until the next pool in {} minutes".format(self._pooling_frequency_minutes))
                sleep(self._pooling_frequency_minutes*60.)

    def start_remotely(self, queue='services'):
        # type: (str) -> None
        """
        Start the Task TaskScheduler loop (notice this function does not return)

        :param queue: Remote queue to run the scheduler on, default 'services' queue.
        """
        self._task.execute_remotely(queue_name=queue, exit_process=True)
        self.start()

    def _update_execution_plots(self):
        # type: () -> None
        """
        Update the configuration and execution table plots
        """
        pass

    def _serialize(self):
        # type: () -> None
        """
        Serialize Task scheduling configuration only (no internal state)
        """
        pass

    def _serialize_state(self):
        # type: () -> None
        """
        Serialize internal state only
        """
        pass

    def _deserialize_state(self):
        # type: () -> None
        """
        Deserialize internal state only
        """
        pass

    def _deserialize(self):
        # type: () -> None
        """
        Deserialize Task scheduling configuration only
        """
        pass

    def _step(self):
        # type: () -> bool
        """
        scheduling processing step. Return True if a new Task was scheduled.
        """
        pass

    def _log(self, message, level=logging.INFO):
        if self._task:
            self._task.get_logger().report_text(message, level=level)
        else:
            print(message)

    def _launch_job(self, job):
        # type: (ScheduleJob) -> None
        self._launch_job_task(job)
        self._launch_job_function(job)

    def _launch_job_task(self, job, task_parameters=None, add_tags=None):
        # type: (BaseScheduleJob, Optional[dict], Optional[List[str]]) -> Optional[ClearmlJob]
        # make sure this is not a function job
        if job.base_function:
            return None

        # check if this is a single instance, then we need to abort the Task
        if job.single_instance and job.get_last_executed_task_id():
            t = Task.get_task(task_id=job.get_last_executed_task_id())
            if t.status in ('in_progress', 'queued'):
                self._log(
                    'Skipping Task {} scheduling, previous Task instance {} still running'.format(
                        job.name, t.id
                    ))
                job.run(None)
                return None

        # actually run the job
        task_job = ClearmlJob(
            base_task_id=job.base_task_id,
            parameter_override=task_parameters or job.task_parameters,
            task_overrides=job.task_overrides,
            disable_clone_task=not job.clone_task,
            allow_caching=False,
            target_project=job.target_project,
            tags=[add_tags] if add_tags and isinstance(add_tags, str) else add_tags,
        )
        self._log('Scheduling Job {}, Task {} on queue {}.'.format(
            job.name, task_job.task_id(), job.queue))
        if task_job.launch(queue_name=job.queue):
            # mark as run
            job.run(task_job.task_id())
        return task_job

    def _launch_job_function(self, job, func_args=None):
        # type: (BaseScheduleJob, Optional[Sequence]) -> Optional[Thread]
        # make sure this IS a function job
        if not job.base_function:
            return None

        # check if this is a single instance, then we need to abort the Task
        if job.single_instance and job.get_last_executed_task_id():
            # noinspection PyBroadException
            try:
                a_thread = [t for t in enumerate_threads() if t.ident == job.get_last_executed_task_id()]
                if a_thread:
                    a_thread = a_thread[0]
            except Exception:
                a_thread = None

            if a_thread and a_thread.is_alive():
                self._log(
                    "Skipping Task '{}' scheduling, previous Thread instance '{}' still running".format(
                        job.name, a_thread.ident
                    ))
                job.run(None)
                return None

        self._log("Scheduling Job '{}', Task '{}' on background thread".format(
            job.name, job.base_function))
        t = Thread(target=job.base_function, args=func_args or ())
        t.start()
        # mark as run
        job.run(t.ident)
        return t

    @staticmethod
    def _cancel_task(task_id):
        # type: (str) -> ()
        if not task_id:
            return
        t = Task.get_task(task_id=task_id)
        status = t.status
        if status in ('in_progress',):
            t.stopped(force=True)
        elif status in ('queued',):
            Task.dequeue(t)


class TaskScheduler(BaseScheduler):
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
        super(TaskScheduler, self).__init__(
            sync_frequency_minutes=sync_frequency_minutes,
            force_create_task_name=force_create_task_name,
            force_create_task_project=force_create_task_project
        )
        self._schedule_jobs = []  # List[ScheduleJob]
        self._timeout_jobs = {}  # Dict[datetime, str]
        self._executed_jobs = []  # List[ExecutedJob]

    def add_task(
            self,
            schedule_task_id=None,  # type: Union[str, Task]
            schedule_function=None,   # type: Callable
            queue=None,  # type: str
            name=None,  # type: Optional[str]
            target_project=None,  # type: Optional[str]
            minute=None,  # type: Optional[int]
            hour=None,  # type: Optional[int]
            day=None,  # type: Optional[int]
            weekdays=None,  # type: Optional[List[str]]
            month=None,  # type: Optional[int]
            year=None,  # type: Optional[int]
            limit_execution_time=None,  # type: Optional[float]
            single_instance=False,  # type: bool
            recurring=True,  # type: bool
            execute_immediately=False,  # type: bool
            reuse_task=False,  # type: bool
            task_parameters=None,  # type: Optional[dict]
            task_overrides=None,  # type: Optional[dict]
    ):
        # type: (...) -> bool
        """
        Create a cron job-like scheduling for a pre-existing Task.
        Notice, it is recommended to give the schedule entry a descriptive unique name,
        if not provided, a name is randomly generated.

        When timespec parameters are specified exclusively, they define the time between task launches (see
        `year` and `weekdays` exceptions). When multiple timespec parameter are specified, the parameter representing
        the longest duration defines the time between task launches, and the shorter timespec parameters define specific
        times.

        Examples:
        Launch every 15 minutes
            add_task(schedule_task_id='1235', queue='default', minute=15)
        Launch every 1 hour
            add_task(schedule_task_id='1235', queue='default', hour=1)
        Launch every 1 hour at hour:30 minutes (i.e. 1:30, 2:30 etc.)
            add_task(schedule_task_id='1235', queue='default', hour=1, minute=30)
        Launch every day at 22:30 (10:30 pm)
            add_task(schedule_task_id='1235', queue='default', minute=30, hour=22, day=1)
        Launch every other day at 7:30 (7:30 am)
            add_task(schedule_task_id='1235', queue='default', minute=30, hour=7, day=2)
        Launch every Saturday at 8:30am (notice `day=0`)
            add_task(schedule_task_id='1235', queue='default', minute=30, hour=8, day=0, weekdays=['saturday'])
        Launch every 2 hours on the weekends Saturday/Sunday (notice `day` is not passed)
            add_task(schedule_task_id='1235', queue='default', hour=2, weekdays=['saturday', 'sunday'])
        Launch once a month at the 5th of each month
            add_task(schedule_task_id='1235', queue='default', month=1, day=5)
        Launch once a year on March 4th
            add_task(schedule_task_id='1235', queue='default', year=1, month=3, day=4)

        :param schedule_task_id: ID of Task to be cloned and scheduled for execution
        :param schedule_function: Optional, instead of providing Task ID to be scheduled,
            provide a function to be called. Notice the function is called from the scheduler context
            (i.e. running on the same machine as the scheduler)
        :param queue: Name or ID of queue to put the Task into (i.e. schedule)
        :param name: Name or description for the cron Task (should be unique if provided, otherwise randomly generated)
        :param target_project: Specify target project to put the cloned scheduled Task in.
        :param minute: Time (in minutes) between task launches. If specified together with `hour`, `day`, `month`,
            and / or  `year`, it defines the minute of the hour
        :param hour: Time (in hours) between task launches. If specified together with `day`, `month`, and / or
            `year`, it defines the hour of day.
        :param day: Time (in days) between task executions. If specified together with  `month` and / or `year`,
            it defines the day of month
        :param weekdays: Days of week to launch task (accepted inputs: 'monday', 'tuesday', 'wednesday',
            'thursday', 'friday', 'saturday', 'sunday')
        :param month: Time (in months) between task launches. If specified with `year`, it defines a specific month
        :param year: Specific year if value >= current year. Time (in years) between task launches if
            value <= 100
        :param limit_execution_time: Limit the execution time (in hours) of the specific job.
        :param single_instance: If True, do not launch the Task job if the previous instance is still running
        (skip until the next scheduled time period). Default False.
        :param recurring: If False only launch the Task once (default: True, repeat)
        :param execute_immediately: If True, schedule the Task to be executed immediately
        then recurring based on the timing schedule arguments. Default False.
        :param reuse_task: If True, re-enqueue the same Task (i.e. do not clone it) every time, default False.
        :param task_parameters: Configuration parameters to the executed Task.
        for example: {'Args/batch': '12'} Notice: not available when reuse_task=True
        :param task_overrides: Change task definition.
        for example {'script.version_num': None, 'script.branch': 'main'} Notice: not available when reuse_task=True

        :return: True if job is successfully added to the scheduling list
        """
        mutually_exclusive(schedule_function=schedule_function, schedule_task_id=schedule_task_id)
        task_id = schedule_task_id.id if isinstance(schedule_task_id, Task) else str(schedule_task_id or '')

        # noinspection PyProtectedMember
        job = ScheduleJob(
            name=name or task_id,
            base_task_id=task_id,
            base_function=schedule_function,
            queue=queue,
            target_project=target_project,
            execution_limit_hours=limit_execution_time,
            recurring=bool(recurring),
            single_instance=bool(single_instance),
            task_parameters=task_parameters,
            task_overrides=task_overrides,
            clone_task=not bool(reuse_task),
            starting_time=datetime.fromtimestamp(0) if execute_immediately else datetime.utcnow(),
            minute=minute,
            hour=hour,
            day=day,
            weekdays=weekdays,
            month=month,
            year=year,
        )
        # raise exception if not valid
        job.verify()

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
        # type: (Union[str, Task, Callable]) -> bool
        """
        Remove a Task ID from the scheduled task list.

        :param task_id: Task or Task ID to be removed
        :return: return True of the Task ID was found in the scheduled jobs list and was removed.
        """
        if isinstance(task_id, (Task, str)):
            task_id = task_id.id if isinstance(task_id, Task) else str(task_id)
            if not any(t.base_task_id == task_id for t in self._schedule_jobs):
                return False
            self._schedule_jobs = [t for t in self._schedule_jobs if t.base_task_id != task_id]
        else:
            if not any(t.base_function == task_id for t in self._schedule_jobs):
                return False
            self._schedule_jobs = [t for t in self._schedule_jobs if t.base_function != task_id]
        return True

    def start(self):
        # type: () -> None
        """
        Start the Task TaskScheduler loop (notice this function does not return)
        """
        super(TaskScheduler, self).start()

    def _step(self):
        # type: () -> bool
        """
        scheduling processing step
        """
        # update next execution datetime
        for j in self._schedule_jobs:
            j.next()

        # get idle timeout (aka sleeping)
        scheduled_jobs = sorted(
            [j for j in self._schedule_jobs if j.next_run() is not None],
            key=lambda x: x.next_run()
        )
        # sort by key
        timeout_job_datetime = min(self._timeout_jobs, key=self._timeout_jobs.get) if self._timeout_jobs else None
        if not scheduled_jobs and timeout_job_datetime is None:
            # sleep and retry
            seconds = 60. * self._sync_frequency_minutes
            self._log('Nothing to do, sleeping for {:.2f} minutes.'.format(seconds / 60.))
            sleep(seconds)
            return False

        next_time_stamp = scheduled_jobs[0].next_run() if scheduled_jobs else None
        if timeout_job_datetime is not None:
            next_time_stamp = (
                min(next_time_stamp, timeout_job_datetime) if next_time_stamp else timeout_job_datetime
            )

        sleep_time = (next_time_stamp - datetime.utcnow()).total_seconds()
        if sleep_time > 0:
            # sleep until we need to run a job or maximum sleep time
            seconds = min(sleep_time, 60. * self._sync_frequency_minutes)
            self._log('Waiting for next run, sleeping for {:.2f} minutes, until next sync.'.format(seconds / 60.))
            sleep(seconds)
            return False

        # check if this is a Task timeout check
        if timeout_job_datetime is not None and next_time_stamp == timeout_job_datetime:
            task_id = self._timeout_jobs[timeout_job_datetime]
            self._log('Aborting job due to timeout: {}'.format(task_id))
            self._cancel_task(task_id=task_id)
            self._timeout_jobs.pop(timeout_job_datetime, None)
        else:
            self._log('Launching job: {}'.format(scheduled_jobs[0]))
            self._launch_job(scheduled_jobs[0])

        return True

    def start_remotely(self, queue='services'):
        # type: (str) -> None
        """
        Start the Task TaskScheduler loop (notice this function does not return)

        :param queue: Remote queue to run the scheduler on, default 'services' queue.
        """
        super(TaskScheduler, self).start_remotely(queue=queue)

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
                timeout_jobs={datetime_to_isoformat(k): v for k, v in self._timeout_jobs.items()},
                executed_jobs=[j.to_dict(full=True) for j in self._executed_jobs]
            ),
            default=datetime_to_isoformat
        )
        self._task.upload_artifact(name="state", artifact_object=json_str, preview="scheduler internal state")

    def _deserialize_state(self):
        # type: () -> None
        """
        Deserialize internal state only
        """
        # get artifact
        self._task.reload()
        artifact_object = self._task.artifacts.get('state')
        if artifact_object is not None:
            state_json_str = artifact_object.get(force_download=True)
            if state_json_str is not None:
                state_dict = json.loads(state_json_str)
                self._schedule_jobs = self.__deserialize_scheduled_jobs(
                    serialized_jobs_dicts=state_dict.get('scheduled_jobs', []),
                    current_jobs=self._schedule_jobs
                )
                self._timeout_jobs = {datetime_from_isoformat(k): v for k, v in (state_dict.get('timeout_jobs') or {})}
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
            self._schedule_jobs = self.__deserialize_scheduled_jobs(
                serialized_jobs_dicts=json.loads(json_str),
                current_jobs=self._schedule_jobs
            )
        except Exception as ex:
            self._log('Failed deserializing configuration: {}'.format(ex), level=logging.WARN)
            return

    @staticmethod
    def __deserialize_scheduled_jobs(serialized_jobs_dicts, current_jobs):
        # type: (List[Dict], List[ScheduleJob]) -> List[ScheduleJob]
        scheduled_jobs = [ScheduleJob().update(j) for j in serialized_jobs_dicts]
        scheduled_jobs = {j.name: j for j in scheduled_jobs}
        current_scheduled_jobs = {j.name: j for j in current_jobs}

        # select only valid jobs, and update the valid ones state from the current one
        new_scheduled_jobs = [
            current_scheduled_jobs[name].update(j) if name in current_scheduled_jobs else j
            for name, j in scheduled_jobs.items()
        ]
        # verify all jobs
        for j in new_scheduled_jobs:
            j.verify()

        return new_scheduled_jobs

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
            'name', 'base_task_id', 'base_function', 'next_run', 'target_project', 'queue',
            'minute', 'hour', 'day', 'month', 'year',
            'starting_time', 'execution_limit_hours', 'recurring',
            'single_instance', 'task_parameters', 'task_overrides', 'clone_task',
        ]
        scheduler_table = [columns]
        for j in self._schedule_jobs:
            j_dict = j.to_dict()
            j_dict['next_run'] = j.next()
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
            if row[1]:
                row[1] = '<a href="{}">{}</a>'.format(
                        task_link_template.format(project='*', task=row[1]), row[1])
            scheduler_table += [row]

        # plot the already executed Tasks
        executed_table = [['name', 'task id', 'started', 'finished']]
        for executed_job in sorted(self._executed_jobs, key=lambda x: x.started, reverse=True):
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
                [executed_job.name,
                 '<a href="{}">{}</a>'.format(task_link_template.format(
                     project='*', task=executed_job.task_id), executed_job.task_id)
                 if executed_job.task_id else 'function',
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

    def _launch_job_task(self, job, task_parameters=None, add_tags=None):
        # type: (ScheduleJob, Optional[dict], Optional[List[str]]) -> Optional[ClearmlJob]
        task_job = super(TaskScheduler, self)._launch_job_task(job, task_parameters=task_parameters, add_tags=add_tags)
        # make sure this is not a function job
        if task_job:
            self._executed_jobs.append(ExecutedJob(
                name=job.name, task_id=task_job.task_id(), started=datetime.utcnow()))
            # add timeout check
            if job.get_execution_timeout():
                # we should probably make sure we are not overwriting a Task
                self._timeout_jobs[job.get_execution_timeout()] = task_job.task_id()
        return task_job

    def _launch_job_function(self, job, func_args=None):
        # type: (ScheduleJob, Optional[Sequence]) -> Optional[Thread]
        thread_job = super(TaskScheduler, self)._launch_job_function(job, func_args=func_args)
        # make sure this is not a function job
        if thread_job:
            self._executed_jobs.append(ExecutedJob(
                name=job.name, thread_id=str(thread_job.ident), started=datetime.utcnow()))
            # execution timeout is not supported with function callbacks.

        return thread_job
