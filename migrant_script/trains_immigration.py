import argparse
import concurrent
import math
import multiprocessing
import os
import re
import sys

import migrant_script.migrant as m
from trains import Task
from concurrent.futures import ThreadPoolExecutor
from trains.backend_api.session.client import APIClient


def chunks(l, n):
    n = max(1, n)
    return (l[i : i + n] for i in range(0, len(l), n))


def delete_all_tasks():
    client = APIClient()
    tasks = Task.get_tasks(project_name="mlflow_migrant")
    for task in tasks:
        client.tasks.delete(task=task.id, force=True)


def task(migrant):
    migrant.read()
    migrant.seed()
    return migrant.size, migrant.thread_id, migrant.paths


def main(path, branch):
    workers = multiprocessing.cpu_count()
    ids_count = 0
    l = []
    experiments = list(os.walk(path))[0][1]  # returns all the dirs in 'self.__path'
    for experiment in experiments:
        if experiment.startswith("."):
            continue
        runs = list(os.walk(path + os.sep + experiment))[0][
            1
        ]  # returns all the dirs in 'self.__path\experiment'
        for run in runs:
            current_path = path + os.sep + experiment + os.sep + run + os.sep
            id = experiment + run
            l.append((id, current_path))
            ids_count += 1
    chunk_size = math.ceil(ids_count / workers)
    print("Experiments count: ", ids_count)
    print("Workers count: ", workers)
    print("Chunk size: ", chunk_size)
    if chunk_size <= 1:
        migrant = m.Migrant(branch, l)
        id, size, jobs = task(migrant)
        print("Thread ", id, ": migrant ", size, " experiments: ", jobs)
    else:
        jobs = chunks(l, chunk_size)
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for job_chunk in jobs:
                migrant = m.Migrant("local", job_chunk)
                future = executor.submit(task, migrant)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    size, id, jobs = future.result()
                    print("Thread ", id, ": migrant ", size, " experiments: ", jobs)
                except Exception as e:
                    print(e)

    print("All tasks completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emigration from Mflow to Trains")
    parser.add_argument(
        "Branch",
        metavar="branch",
        type=str,
        help="Where Mlflow runs? (Local or Remote)",
    )
    parser.add_argument(
        "Path", metavar="path", type=str, help="the path/address to mlruns directory"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.Path):
        print("The path specified does not exist")
        sys.exit()
    if not re.match(r"^(?:[Ll]ocal)|(?:[Rr]emote)", args.Branch):
        print("The branch argument must be local or remote")

    main(args.Path, args.Branch)
