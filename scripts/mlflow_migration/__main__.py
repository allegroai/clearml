"""Migration from MLFlow to ClearML"""
import argparse
import concurrent
import math
import multiprocessing
import traceback
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

from clearml import Task
from tqdm import tqdm

from .db_util.dblib import close
from .sources import SourceFactory, Source
from .time_counter import Timer
from .util import chunks


def do_migration(path: str, analysis: bool):
    timer = Timer()
    project_link = None
    error_list = []
    failure_list = []
    workers = multiprocessing.cpu_count()
    source_factory = SourceFactory(path)
    l, ids_count = source_factory.get_runs()
    chunk_size = math.ceil(ids_count / workers)

    completed_migrations = 0
    project_exists = next((True for p in Task.get_projects() if p.name == "MLFlow Migration"), False)

    if chunk_size == 0:
        print("No experiments to migrate")
        return

    jobs = chunks(l, chunk_size)
    futures = []
    with tqdm(total=ids_count, desc="Progress") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for job_chunk in jobs:
                source = source_factory.create(
                    job_chunk, pbar, timer, analysis, project_exists
                )
                future = executor.submit(Source.task_from_source, source)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    msgs, project_link_per_task, migration_count = future.result()
                    completed_migrations += migration_count
                    failure_list.append(msgs["FAILED"])
                    error_list.append(msgs["ERROR"])
                    if not project_link:
                        project_link = project_link_per_task
                except Exception as e:
                    tb1 = traceback.TracebackException.from_exception(e)
                    error_list.append(["Error: " + "".join(tb1.format())])

    close()
    print("\n".join(chain(*failure_list)))
    print("\n".join(chain(*error_list)))

    print(f"Completed. {completed_migrations} experiments migrated.")

    print("Link to the migrated project: ", project_link)
    timer.print_times()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path_or_address",
        type=str,
        help="Path or address to MLFlow server (mlruns folder on local machine)",
    )
    parser.add_argument(
        "analysis",
        action="store_true",
        help="Analysis mode (default %(default)s",
        default=False,
    )

    args = parser.parse_args()

    do_migration(args.path_or_address, analysis=args.analysis)
