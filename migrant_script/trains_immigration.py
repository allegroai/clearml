import argparse
import concurrent
import math
import multiprocessing
import os
import re
import sys


from local_migrant import LocalMigrant
from remote_migrant import RemoteMigrant
from trains import Task
from concurrent.futures import ThreadPoolExecutor
from trains.backend_api.session.client import APIClient

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
from dblib import get_run_uuids
from dblib import init_session
from dblib import close

migrant_dict = {
    "LOCAL": lambda paths: LocalMigrant(paths),
    "REMOTE": lambda addresses: RemoteMigrant(addresses),
}


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


def get_runs(branch, address):
    if branch == "LOCAL":
        return get_runs_from_local(address)
    elif branch == "REMOTE":
        return get_runs_from_remote(address)


def get_runs_from_remote(address):
    DB_engine = create_engine(address)
    Session_factory = sessionmaker(bind=DB_engine)
    Session = scoped_session(Session_factory)
    init_session(Session)
    run_ids = get_run_uuids()

    return run_ids, len(run_ids)


def get_runs_from_local(path):
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
    return l, ids_count


def main(path, branch):
    workers = multiprocessing.cpu_count()
    l, ids_count = get_runs(branch, path)
    chunk_size = math.ceil(ids_count / workers)
    print("Experiments count: ", ids_count)
    print("Chunk size: ", chunk_size)
    if chunk_size <= 1:
        print("Workers count: 1")
        migrant = migrant_dict[branch](l)
        if branch == "LOCAL":
            assert isinstance(migrant, LocalMigrant)
        elif branch == "REMOTE":
            assert isinstance(migrant, RemoteMigrant)
        id, size, jobs = task(migrant)
        print("Thread ", id, ": migrant ", size, " experiments: ", jobs)
    else:
        print("Workers count: ", workers)
        jobs = chunks(l, chunk_size)
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for job_chunk in jobs:
                migrant = migrant_dict[branch](job_chunk)
                if branch == "LOCAL":
                    assert isinstance(migrant, LocalMigrant)
                elif branch == "REMOTE":
                    assert isinstance(migrant, RemoteMigrant)
                future = executor.submit(task, migrant)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    size, id, jobs = future.result()
                    print("Thread ", id, ": migrant ", size, " experiments: ", jobs)
                except Exception as e:
                    print("Error: ", e)
    close()
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

    if not re.match(r"^(?:[Ll]ocal)|(?:[Rr]emote)", args.Branch):
        print("The branch argument must be local or remote")
        sys.exit()
    if args.Branch.upper() == "LOCAL":
        if not os.path.isdir(args.Path):
            print("The path specified does not exist")
            sys.exit()

    main(args.Path, args.Branch.upper())

    # delete_all_tasks()
    # main('/Users/tmankita/Downloads/mlruns','LOCAL')
