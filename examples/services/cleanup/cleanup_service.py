"""
This service will delete archived experiments and their accompanying debug samples, artifacts and models
older than 30 days.

You can configure the run by changing the `args` dictionary:
- delete_threshold_days (float): The earliest day for cleanup.
                                 Only tasks older to this will be deleted. Default: 30.
- cleanup_period_in_days (float): The time period between cleanups. Default: 1.
- run_as_service (bool): The script will be execute remotely (Default queue: "services"). Default: True.
- force_delete (bool): Allows forcing the task deletion (for every task status). Default: False.

Requirements:
- clearml_agent installed -> pip install clearml-agent

"""
import logging
import os
from datetime import datetime
from glob import glob
from shutil import rmtree
from time import sleep, time

from clearml.backend_api.session.client import APIClient

from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(
    project_name="DevOps",
    task_name="Cleanup Service",
    task_type=Task.TaskTypes.service,
    reuse_last_task_id=False,
)

# set the base docker including the mount point for the file server data data
file_server_mount = "/opt/clearml/data/fileserver/"
task.set_base_docker(
    "ubuntu:18.04 -v /opt/clearml/data/fileserver/:{}".format(file_server_mount)
)

# args for the running task
args = {
    "delete_threshold_days": 30.0,
    "cleanup_period_in_days": 1.0,
    "run_as_service": True,
    "force_delete": False,
}
args = task.connect(args)


# if we are running as a service, just enqueue ourselves into the services queue and let it run the optimization
if args["run_as_service"] and task.running_locally():
    verify = input('Stop local execution and execute remotely [y]/n ?').strip().lower()
    args["run_as_service"] = not verify or verify.startswith('y')

if args["run_as_service"]:
    # if this code is executed by `clearml-agent` the function call does nothing.
    # if executed locally, the local process will be terminated, and a remote copy will be executed instead
    task.execute_remotely(queue_name="services", exit_process=True)

print("Cleanup service started")

while True:
    print("Starting cleanup")
    client = APIClient()
    # anything that has not changed in the last month
    timestamp = time() - 60 * 60 * 24 * args["delete_threshold_days"]
    page = 0
    page_size = 100
    tasks = None
    while tasks is None or len(tasks) == page_size:
        tasks = client.tasks.get_all(
            system_tags=["archived"],
            only_fields=["id"],
            order_by=["-last_update"],
            page_size=page_size,
            page=page,
            status_changed=["<{}".format(datetime.utcfromtimestamp(timestamp))],
        )
        page += 1

        # delete and cleanup tasks
        for task in tasks:
            # noinspection PyBroadException
            try:
                # try delete task frm system
                client.tasks.delete(task=task.id, force=args["force_delete"])
                # if we succeeded, delete the task output content
                task_folders = glob(
                    os.path.join(file_server_mount, "*/*.{}/".format(task.id))
                )
                for folder in task_folders:
                    print("Deleting Task id={} data folder {}".format(task.id, folder))
                    # noinspection PyBroadException
                    try:
                        rmtree(folder)
                    except Exception:
                        logging.warning("Failed removing folder {}".format(folder))
            except Exception as ex:
                logging.warning(
                    "Could not delete Task ID={}, {}".format(
                        task.id, ex.message if hasattr(ex, "message") else ex
                    )
                )
                continue

    # sleep until the next day
    print("going to sleep for {} days".format(args["cleanup_period_in_days"]))
    sleep(60 * 60 * 24.0 * args["cleanup_period_in_days"])
