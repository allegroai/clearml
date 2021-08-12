import os
import re
from itertools import chain
from operator import itemgetter
from time import sleep, time
from typing import Union

import attr
from attr.validators import instance_of

from ..backend_api import Session
from ..backend_api.session.client import APIClient


class AutoScaler(object):
    @attr.s
    class Settings(object):
        git_user = attr.ib(default="")
        git_pass = attr.ib(default="")
        cloud_credentials_key = attr.ib(default="")
        cloud_credentials_secret = attr.ib(default="")
        cloud_credentials_region = attr.ib(default=None)
        default_docker_image = attr.ib(default="nvidia/cuda")
        max_idle_time_min = attr.ib(validator=instance_of(int), default=15)
        polling_interval_time_min = attr.ib(validator=instance_of(int), default=5)
        max_spin_up_time_min = attr.ib(validator=instance_of(int), default=30)
        workers_prefix = attr.ib(default="dynamic_worker")
        cloud_provider = attr.ib(default="")

        def as_dict(self):
            return attr.asdict(self)

    @attr.s
    class Configuration(object):
        resource_configurations = attr.ib(default=None)
        queues = attr.ib(default=None)
        extra_trains_conf = attr.ib(default="")     # Backwards compatibility
        extra_clearml_conf = attr.ib(default="")
        extra_vm_bash_script = attr.ib(default="")

        def as_dict(self):
            return attr.asdict(self)

    def __init__(self, settings, configuration):
        # type: (Union[dict, AutoScaler.Settings], Union[dict, AutoScaler.Configuration]) -> None
        if isinstance(settings, dict):
            settings = self.Settings(**settings)
        if isinstance(configuration, dict):
            configuration = self.Configuration(**configuration)

        self.web_server = Session.get_app_server_host()
        self.api_server = Session.get_api_server_host()
        self.files_server = Session.get_files_server_host()

        session = Session()
        self.access_key = session.access_key
        self.secret_key = session.secret_key

        self.git_user = settings.git_user
        self.git_pass = settings.git_pass
        self.cloud_credentials_key = settings.cloud_credentials_key
        self.cloud_credentials_secret = settings.cloud_credentials_secret
        self.cloud_credentials_region = settings.cloud_credentials_region
        self.default_docker_image = settings.default_docker_image

        self.extra_clearml_conf = configuration.extra_clearml_conf or configuration.extra_trains_conf
        self.extra_vm_bash_script = configuration.extra_vm_bash_script
        self.resource_configurations = configuration.resource_configurations
        self.queues = configuration.queues

        if not self.sanity_check():
            return

        self.max_idle_time_min = float(settings.max_idle_time_min)
        self.polling_interval_time_min = float(settings.polling_interval_time_min)
        self.max_spin_up_time_min = float(settings.max_spin_up_time_min)

        # make sure we have our own unique prefix, in case we have multiple dynamic auto-scalers
        # they will mix each others instances
        self.workers_prefix = settings.workers_prefix
        self.cloud_provider = settings.cloud_provider

    def sanity_check(self):
        # Sanity check - Validate queue resources
        if len(set(map(itemgetter(0), chain(*self.queues.values())))) != sum(
            map(len, self.queues.values())
        ):
            print(
                "Error: at least one resource name is used in multiple queues. "
                "A resource name can only appear in a single queue definition."
            )
            return False
        return True

    def start(self):
        # Loop forever, it is okay we are stateless
        while True:
            try:
                self.supervisor()
            except Exception as ex:
                print(
                    "Warning! exception occurred: {ex}\nRetry in 15 seconds".format(
                        ex=ex
                    )
                )
                sleep(15)

    def spin_up_worker(self, resource, worker_id_prefix, queue_name):
        """
        Creates a new worker for clearml (cloud-specific implementation).
        First, create an instance in the cloud and install some required packages.
        Then, define clearml-agent environment variables and run clearml-agent for the specified queue.
        NOTE: - Will wait until instance is running
              - This implementation assumes the instance image already has docker installed

        :param str resource: resource name, as defined in self.resource_configurations and self.queues.
        :param str worker_id_prefix: worker name prefix
        :param str queue_name: clearml queue to listen to

        :return str: worker_id prefix to identify when spin was successful
        """
        pass

    def spin_down_worker(self, instance_id):
        """
        Destroys the cloud instance (cloud-specific implementation).

        :param instance_id: Cloud instance ID to be destroyed
        :type instance_id: str
        """
        pass

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

        # Worker's id in clearml would be composed from prefix, name, instance_type and cloud_id separated by ':'
        workers_pattern = re.compile(
            r"^(?P<prefix>[^:]+):(?P<name>[^:]+):(?P<instance_type>[^:]+):(?P<cloud_id>[^:]+)"
        )

        # Set up the environment variables for clearml
        os.environ["CLEARML_API_HOST"] = self.api_server
        os.environ["CLEARML_WEB_HOST"] = self.web_server
        os.environ["CLEARML_FILES_HOST"] = self.files_server
        os.environ["CLEARML_API_ACCESS_KEY"] = self.access_key
        os.environ["CLEARML_API_SECRET_KEY"] = self.secret_key
        api_client = APIClient()

        # Verify the requested queues exist and create those that doesn't exist
        all_queues = [q.name for q in list(api_client.queues.get_all(only_fields=['name']))]
        missing_queues = [q for q in self.queues if q not in all_queues]
        for q in missing_queues:
            api_client.queues.create(q)

        idle_workers = {}
        # a dict of resource_names and lists of time_stamps of instances that were just spun
        # example
        # spun_workers['resource_type'] = [time()]
        spun_workers = {}
        previous_workers = set()
        while True:
            queue_name_to_id = {
                queue.name: queue.id for queue in api_client.queues.get_all(only_fields=['id', 'name'])
            }
            resource_to_queue = {
                item[0]: queue
                for queue, resources in self.queues.items()
                for item in resources
            }
            all_workers = [
                worker
                for worker in api_client.workers.get_all()
                if workers_pattern.match(worker.id)
                and workers_pattern.match(worker.id)["prefix"] == self.workers_prefix
            ]
            # update spun_workers (remove instances that are fully registered)
            for worker in all_workers:
                if worker.id not in previous_workers:
                    # look for the spun instance and remove it
                    resource_name = workers_pattern.match(worker.id)[
                        "name"
                    ]
                    spun_workers[resource_name] = spun_workers.get(resource_name, [])[1:]
            # remove old spun workers based on time out:
            for resource in spun_workers.keys():
                time_stamp_list = [
                    t for t in spun_workers[resource] if time() - t < self.max_spin_up_time_min*60.]
                deleted = len(spun_workers[resource]) - len(time_stamp_list)
                if deleted:
                    print('Ignoring {} stuck instances of type {}'.format(deleted, resource))

            # Workers without a task, are added to the idle list
            if not all_workers:
                idle_workers = {}
            else:
                for worker in all_workers:
                    if not hasattr(worker, "task") or not worker.task:
                        if worker.id not in idle_workers:
                            resource_name = workers_pattern.match(worker.id)[
                                "instance_type"
                            ]
                            idle_workers[worker.id] = (time(), resource_name, worker)
                    elif (
                        hasattr(worker, "task")
                        and worker.task
                        and worker.id in idle_workers
                    ):
                        idle_workers.pop(worker.id, None)

            required_idle_resources = []  # idle resources we'll need to keep running
            allocate_new_resources = []  # resources that will need to be started
            # Check if we have tasks waiting on one of the designated queues
            for queue in self.queues:
                entries = api_client.queues.get_by_id(queue_name_to_id[queue]).entries
                if entries and len(entries) > 0:
                    queue_resources = self.queues[queue]

                    # If we have an idle worker matching the required resource,
                    # remove it from the required allocation resources
                    free_queue_resources = [
                        resource
                        for _, resource, _ in idle_workers.values()
                        if any(q_r for q_r in queue_resources if resource in q_r[0])
                    ]
                    # if we have an instance waiting to be spun
                    # remove it from the required allocation resources
                    for resource, time_stamps_list in spun_workers.items():
                        if time_stamps_list and any(q_r for q_r in queue_resources if resource in q_r[0]):
                            free_queue_resources += [resource] * len(time_stamps_list)

                    required_idle_resources.extend(free_queue_resources)
                    spin_up_count = len(entries) - len(free_queue_resources)
                    spin_up_resources = []

                    # Add as many resources as possible to handle this queue's entries
                    for resource, max_instances in queue_resources:
                        if len(spin_up_resources) >= spin_up_count:
                            break
                        # check if we can add instances to `resource`
                        currently_running_workers = len(
                            [worker for worker in all_workers if workers_pattern.match(worker.id)["name"] == resource])
                        spun_up_workers = len(spun_workers.get(resource, []))
                        max_allowed = int(max_instances) - currently_running_workers - spun_up_workers
                        if max_allowed > 0:
                            spin_up_resources.extend(
                                [resource] * spin_up_count
                            )
                    allocate_new_resources.extend(spin_up_resources)

            # Now we actually spin the new machines
            for resource in allocate_new_resources:
                try:
                    print('Spinning new instance type={}'.format(resource))
                    self.spin_up_worker(
                        resource, self.workers_prefix, resource_to_queue[resource]
                    )
                    spun_workers[resource] = spun_workers.get(resource, []) + [time()]
                except Exception as ex:
                    print(f"Error: Failed to start new instance, {ex}")

            # Go over the idle workers list, and spin down idle workers
            for worker_id in list(idle_workers):
                timestamp, resources, worker = idle_workers[worker_id]
                # skip resource types that might be needed
                if resources in required_idle_resources:
                    continue
                # Remove from both aws and clearml all instances that are idle for longer than MAX_IDLE_TIME_MIN
                if time() - timestamp > self.max_idle_time_min * 60.0:
                    cloud_id = workers_pattern.match(worker_id)["cloud_id"]
                    self.spin_down_worker(cloud_id)
                    print(f"Spin down instance cloud id {cloud_id}")
                    idle_workers.pop(worker_id, None)

            # Nothing else to do
            sleep(self.polling_interval_time_min * 60.0)
