import re
from collections import defaultdict, deque
from enum import Enum
from itertools import chain
from threading import Event, Thread
from time import sleep, time

import attr
from attr.validators import instance_of

from .cloud_driver import CloudDriver
from .. import Task
from ..backend_api import Session
from ..backend_api.session import defs
from ..backend_api.session.client import APIClient
from ..debugging import get_logger

# Worker's id in clearml would be composed from prefix, name, instance_type and cloud_id separated by ":"
# Example: 'test:m1:g4dn.4xlarge:i-07cf7d6750455cb62'
# cloud_id might be missing

_workers_pattern = re.compile(
    r"""^
    (?P<prefix>[^:]+):
    (?P<name>[^:]+):
    (?P<instance_type>[^:]+)
    (:(?P<cloud_id>[^:/]+))?
    $
    """, re.VERBOSE
)

MINUTE = 60.0


class WorkerId:
    def __init__(self, worker_id):
        self.prefix = self.name = self.instance_type = self.cloud_id = ""
        match = _workers_pattern.match(worker_id)
        if not match:
            raise ValueError("bad worker ID: {!r}".format(worker_id))

        self.prefix = match["prefix"]
        self.name = match["name"]
        self.instance_type = match["instance_type"]
        self.cloud_id = match["cloud_id"] or ''


class State(str, Enum):
    STARTING = 'starting'
    READY = 'ready'
    RUNNING = 'running'
    STOPPED = 'stopped'


@attr.s
class ScalerConfig:
    max_idle_time_min = attr.ib(validator=instance_of(int), default=15)
    polling_interval_time_min = attr.ib(validator=instance_of((float, int)), default=5)
    max_spin_up_time_min = attr.ib(validator=instance_of(int), default=30)
    workers_prefix = attr.ib(default="dynamic_worker")
    resource_configurations = attr.ib(default=None)
    queues = attr.ib(default=None)

    @classmethod
    def from_config(cls, config):
        return cls(
            max_idle_time_min=config['hyper_params']['max_idle_time_min'],
            polling_interval_time_min=config['hyper_params']['polling_interval_time_min'],
            max_spin_up_time_min=config['hyper_params']['max_spin_up_time_min'],
            workers_prefix=config['hyper_params']['workers_prefix'],
            resource_configurations=config['configurations']['resource_configurations'],
            queues=config['configurations']['queues'],
        )


class AutoScaler(object):
    def __init__(self, config, driver: CloudDriver, logger=None):
        self.logger = logger or get_logger('auto_scaler')
        # Should be after we create logger
        self.state = State.STARTING

        self.driver = driver
        self.logger.info('using %s driver', self.driver.kind())
        self.driver.set_scaler(self)

        self.resource_configurations = config.resource_configurations
        self.queues = config.queues  # queue name -> list of resources
        self.resource_to_queue = {
            item[0]: queue
            for queue, resources in self.queues.items()
            for item in resources
        }

        if not self.sanity_check():
            raise ValueError('health check failed')

        self.max_idle_time_min = float(config.max_idle_time_min)
        self.polling_interval_time_min = float(config.polling_interval_time_min)
        self.max_spin_up_time_min = float(config.max_spin_up_time_min)

        # make sure we have our own unique prefix, in case we have multiple dynamic auto-scalers
        # they will mix each others instances
        self.workers_prefix = config.workers_prefix

        session = Session()
        self.set_auth(session)

        # Set up the environment variables for clearml
        defs.ENV_HOST.set(session.get_api_server_host())
        defs.ENV_WEB_HOST.set(session.get_app_server_host())
        defs.ENV_FILES_HOST.set(session.get_files_server_host())
        defs.ENV_ACCESS_KEY.set(session.access_key)
        defs.ENV_SECRET_KEY.set(session.secret_key)
        if self.auth_token:
            defs.ENV_AUTH_TOKEN.set(self.auth_token)

        self.api_client = APIClient()
        self._stop_event = Event()
        self.state = State.READY

    def set_auth(self, session):
        if session.access_key and session.secret_key:
            self.access_key = session.access_key
            self.secret_key = session.secret_key
            self.auth_token = None
            return

        self.access_key = self.secret_key = None
        self.auth_token = defs.ENV_AUTH_TOKEN.get(default=None)

    def sanity_check(self):
        if has_duplicate_resource(self.queues):
            self.logger.error(
                "Error: at least one resource name is used in multiple queues. "
                "A resource name can only appear in a single queue definition."
            )
            return False
        return True

    def start(self):
        self.state = State.RUNNING
        # Loop until stopped, it is okay we are stateless
        while self._running():
            try:
                self.supervisor()
            except Exception as ex:
                self.logger.exception('Error: %r, retrying in 15 seconds', ex)
                sleep(15)

    def stop(self):
        self.logger.info('stopping')
        self._stop_event.set()
        self.state = State.STOPPED

    def ensure_queues(self):
        # Verify the requested queues exist and create those that doesn't exist
        all_queues = {q.name for q in list(self.api_client.queues.get_all(only_fields=['name']))}
        missing_queues = set(self.queues) - all_queues
        for q in missing_queues:
            self.logger.info("Creating queue %r", q)
            self.api_client.queues.create(q)

    def queue_mapping(self):
        id_to_name = {}
        name_to_id = {}
        for queue in self.api_client.queues.get_all(only_fields=['id', 'name']):
            id_to_name[queue.id] = queue.name
            name_to_id[queue.name] = queue.id

        return id_to_name, name_to_id

    def get_workers(self):
        workers = []
        for worker in self.api_client.workers.get_all():
            try:
                wid = WorkerId(worker.id)
                if wid.prefix == self.workers_prefix:
                    workers.append(worker)
            except ValueError:
                self.logger.info('ignoring unknown worker: %r', worker.id)
        return workers

    def stale_workers(self, spun_workers):
        now = time()
        for worker_id, (resource, spin_time) in list(spun_workers.items()):
            if now - spin_time > self.max_spin_up_time_min * MINUTE:
                self.logger.info('Stuck spun instance %s of type %s', worker_id, resource)
                yield worker_id

    def extra_allocations(self):
        """Hook for subclass to use"""
        return []

    def gen_worker_prefix(self, resource, resource_conf):
        return '{workers_prefix}:{worker_type}:{instance_type}'.format(
            workers_prefix=self.workers_prefix,
            worker_type=resource,
            instance_type=resource_conf["instance_type"],
        )

    def supervisor(self):
        """
        Spin up or down resources as necessary.
        - For every queue in self.queues do the following:
            1. Check if there are tasks waiting in the queue.
            2. Check if there are enough idle workers available for those tasks.
            3. In case more instances are required, and we haven't reached max instances allowed,
               create the required instances with regards to the maximum number defined in self.queues
               Choose which instance to create according to their order in self.queues. Won't create more instances
               if maximum number defined has already reached.
        - spin down instances according to their idle time. instance which is idle for more than self.max_idle_time_min
        minutes would be removed.
        """
        self.ensure_queues()

        idle_workers = {}
        # Workers that we spun but have not yet reported back to the API
        spun_workers = {}  # worker_id -> (resource type, spin time)
        previous_workers = set()
        unknown_workers = deque(maxlen=256)
        task_logger = get_task_logger()
        up_machines = defaultdict(int)

        while self._running():
            queue_id_to_name, queue_name_to_id = self.queue_mapping()
            all_workers = self.get_workers()

            # update spun_workers (remove instances that are fully registered)
            for worker in all_workers:
                if worker.id not in previous_workers:
                    if not spun_workers.pop(worker.id, None):
                        if worker.id not in unknown_workers:
                            self.logger.info('Removed unknown worker from spun_workers: %s', worker.id)
                            unknown_workers.append(worker.id)
                    else:
                        previous_workers.add(worker.id)

            for worker_id in self.stale_workers(spun_workers):
                out = spun_workers.pop(worker_id, None)
                if out is None:
                    self.logger.warning('Ignoring unknown stale worker: %r', worker_id)
                    continue
                resource = out[0]
                try:
                    self.logger.info('Spinning down stuck worker: %r', worker_id)
                    self.driver.spin_down_worker(WorkerId(worker_id).cloud_id)
                    up_machines[resource] -= 1
                except Exception as err:
                    self.logger.info('Cannot spin down %r: %r', worker_id, err)

            self.update_idle_workers(all_workers, idle_workers)
            required_idle_resources = []  # idle resources we'll need to keep running
            allocate_new_resources = self.extra_allocations()

            # Check if we have tasks waiting on one of the designated queues
            for queue in self.queues:
                entries = self.api_client.queues.get_by_id(queue_name_to_id[queue]).entries
                self.logger.info("Found %d tasks in queue %r", len(entries), queue)
                if entries and len(entries) > 0:
                    queue_resources = self.queues[queue]

                    # If we have an idle worker matching the required resource,
                    # remove it from the required allocation resources
                    free_queue_resources = [
                        resource_name
                        for _, resource_name, _ in idle_workers.values()
                        if any(q_r for q_r in queue_resources if resource_name == q_r[0])
                    ]
                    # if we have an instance waiting to be spun
                    # remove it from the required allocation resources
                    for resource, _ in spun_workers.values():
                        if resource in [qr[0] for qr in queue_resources]:
                            free_queue_resources.append(resource)

                    required_idle_resources.extend(free_queue_resources)
                    spin_up_count = len(entries) - len(free_queue_resources)
                    spin_up_resources = []

                    # Add as many resources as possible to handle this queue's entries
                    for resource, max_instances in queue_resources:
                        if len(spin_up_resources) >= spin_up_count:
                            break
                        # check if we can add instances to `resource`
                        currently_running_workers = len(
                            [worker for worker in all_workers if WorkerId(worker.id).name == resource])
                        spun_up_workers = sum(1 for r, _ in spun_workers.values() if r == resource)
                        max_allowed = int(max_instances) - currently_running_workers - spun_up_workers
                        if max_allowed > 0:
                            spin_up_resources.extend(
                                [resource] * min(spin_up_count, max_allowed)
                            )
                    allocate_new_resources.extend(spin_up_resources)

            # Now we actually spin the new machines
            for resource in allocate_new_resources:
                task_id = None
                try:
                    if isinstance(resource, tuple):
                        worker_id, task_id = resource
                        resource = WorkerId(worker_id).name

                    queue = self.resource_to_queue[resource]
                    suffix = ', task_id={!r}'.format(task_id) if task_id else ''
                    self.logger.info(
                        'Spinning new instance resource=%r, prefix=%r, queue=%r%s',
                        resource, self.workers_prefix, queue, suffix)
                    resource_conf = self.resource_configurations[resource]
                    worker_prefix = self.gen_worker_prefix(resource, resource_conf)
                    instance_id = self.driver.spin_up_worker(resource_conf, worker_prefix, queue, task_id=task_id)
                    self.monitor_startup(instance_id)
                    worker_id = '{}:{}'.format(worker_prefix, instance_id)
                    self.logger.info('New instance ID: %s', instance_id)
                    spun_workers[worker_id] = (resource, time())
                    up_machines[resource] += 1
                except Exception as ex:
                    self.logger.exception("Failed to start new instance (resource %r), Error: %s", resource, ex)

            # Go over the idle workers list, and spin down idle workers
            for worker_id in list(idle_workers):
                timestamp, resource_name, worker = idle_workers[worker_id]
                # skip resource types that might be needed
                if resource_name in required_idle_resources:
                    continue
                # Remove from both cloud and clearml all instances that are idle for longer than MAX_IDLE_TIME_MIN
                if time() - timestamp > self.max_idle_time_min * MINUTE:
                    wid = WorkerId(worker_id)
                    cloud_id = wid.cloud_id
                    self.driver.spin_down_worker(cloud_id)
                    up_machines[wid.name] -= 1
                    self.logger.info("Spin down instance cloud id %r", cloud_id)
                    idle_workers.pop(worker_id, None)

            if task_logger:
                self.report_app_stats(task_logger, queue_id_to_name, up_machines, idle_workers)

            # Nothing else to do
            self.logger.info("Idle for %.2f seconds", self.polling_interval_time_min * MINUTE)
            sleep(self.polling_interval_time_min * MINUTE)

    def update_idle_workers(self, all_workers, idle_workers):
        if not all_workers:
            idle_workers.clear()
            return

        for worker in all_workers:
            task = getattr(worker, 'task', None)
            if not task:
                if worker.id not in idle_workers:
                    resource_name = WorkerId(worker.id).name
                    worker_time = worker_last_time(worker)
                    idle_workers[worker.id] = (worker_time, resource_name, worker)
            elif worker.id in idle_workers:
                idle_workers.pop(worker.id, None)

    def _running(self):
        return not self._stop_event.is_set()

    def report_app_stats(self, logger, queue_id_to_name, up_machines, idle_workers):
        self.logger.info('resources: %r', self.resource_to_queue)
        self.logger.info('idle worker: %r', idle_workers)
        self.logger.info('up machines: %r', up_machines)

    # Using property for state to log state change
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        prev = getattr(self, '_state', None)
        if prev:
            self.logger.info('state change: %s -> %s', prev, value)
        else:
            self.logger.info('initial state: %s', value)
        self._state = value

    def monitor_startup(self, instance_id):
        thr = Thread(target=self.instance_log_thread, args=(instance_id,))
        thr.daemon = True
        thr.start()

    def instance_log_thread(self, instance_id):
        start = time()
        # The driver will return the log content from the start on every call,
        # we keep record to avoid logging the same line twice
        # TODO: Find a cross cloud way to get incremental logs
        last_lnum = 0
        while time() - start <= self.max_spin_up_time_min * MINUTE:
            self.logger.info('getting startup logs for %r', instance_id)
            data = self.driver.console_log(instance_id)
            lines = data.splitlines()
            if not lines:
                self.logger.info('not startup logs for %r', instance_id)
            else:
                last_lnum, lines = latest_lines(lines, last_lnum)
                for line in lines:
                    self.logger.info('%r STARTUP LOG: %s', instance_id, line)
            sleep(MINUTE)


def latest_lines(lines, last):
    """Return lines after last and not empty

    >>> latest_lines(['a', 'b', '', 'c', '', 'd'], 1)
    6, ['c', 'd']
    """
    last_lnum = len(lines)
    latest = [l for n, l in enumerate(lines, 1) if n > last and l.strip()]
    return last_lnum, latest


def get_task_logger():
    task = Task.current_task()
    return task and task.get_logger()


def has_duplicate_resource(queues: dict):
    """queues: dict[name] -> [(resource, count), (resource, count) ...]"""
    seen = set()
    for name, _ in chain.from_iterable(queues.values()):
        if name in seen:
            return True
        seen.add(name)
    return False


def worker_last_time(worker):
    """Last time we heard from a worker. Current time if we can't find"""
    time_attrs = [
        'register_time',
        'last_activity_time',
        'last_report_time',
    ]
    times = [getattr(worker, attr).timestamp() for attr in time_attrs if getattr(worker, attr)]
    return max(times) if times else time()
