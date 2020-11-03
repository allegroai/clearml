import argparse
import concurrent
import math
import multiprocessing
import os
from os.path import expanduser

from migrant_script.migrant_classes.migrant_factory import MigrantFactory
from trains import Task
from concurrent.futures import ThreadPoolExecutor
from trains.backend_api.session.client import APIClient

from migrant_script.db_util.dblib import close

def chunks(l, n):
    n = max(1, n)
    return (l[i : i + n] for i in range(0, len(l), n))


def delete_all_tasks_from_project(pr_name):
    client = APIClient()
    tasks = Task.get_tasks(project_name=pr_name)
    for task in tasks:
        client.tasks.delete(task=task.id, force=True)


def task(migrant):
    migrant.read()
    migrant.seed()
    return migrant.size, migrant.thread_id, migrant.paths, migrant.msgs, migrant.project_link



def thread_print(size, id, jobs, msgs):
    if size - len(msgs['FAILED']) == 0:
        print("Thread ", id, ": failed to migrant the following experiments: ", jobs, ' messages: ', msgs['FAILED'])
    else:
        print("Thread ", id, ": migrant ", size - len(msgs['FAILED']), " experiments: ", jobs, ' messages: ', msgs)


def main(path):
    project_link = None
    home = expanduser('~')
    tmp_dir = home + os.sep + '.tmp_mlflow_migration'
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    workers = multiprocessing.cpu_count()
    migrant_factory = MigrantFactory(path)
    l, ids_count = migrant_factory.get_runs()
    chunk_size = math.ceil(ids_count / workers)
    print("Experiments count: ", ids_count)
    print("Chunk size: ", chunk_size)
    if chunk_size==0:
        print("No experiments to migrate")
        return
    elif chunk_size == 1:
        print("Workers count: 1")
        migrant = migrant_factory.create(l)
        size, id, jobs, msgs, project_link = task(migrant)
        thread_print(size, id, jobs, msgs)
    else:
        print("Workers count: ", workers)
        jobs = chunks(l, chunk_size)
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for job_chunk in jobs:
                migrant = migrant_factory.create(job_chunk)
                future = executor.submit(task, migrant)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    size, id, jobs, msgs, project_link = future.result()
                    thread_print(size, id, jobs, msgs)
                except Exception as e:
                    print("Error: ", e)
    close()
    print("All tasks completed")
    print("Link to the project: ", project_link)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emigration from Mflow to Trains")
    parser.add_argument(
        "Path", metavar="path", type=str, help="the path/address to mlruns directory"
    )

    args = parser.parse_args()
    main(args.Path)
