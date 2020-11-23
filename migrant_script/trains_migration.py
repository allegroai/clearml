import argparse
import concurrent
import math
import multiprocessing
import traceback

from tqdm import tqdm



from migrant_classes.migrant_factory import MigrantFactory
from trains import Task
from concurrent.futures import ThreadPoolExecutor
from trains.backend_api.session.client import APIClient

from db_util.dblib import close
from time_counter import Timer


def chunks(l, n):
    # type: (list[str],int) -> List[List[(str,str)]]
    """
    <Description>

    :param list[str] l:
    :param int n:
    :return:
    """
    n = max(1, n)
    return (l[i : i + n] for i in range(0, len(l), n))


def delete_all_tasks_from_project(pr_name):
    # type: (str) -> ()
    """
    <Description>

    :param str pr_name:
    """
    client = APIClient()
    tasks = Task.get_tasks(project_name=pr_name)
    for task in tasks:
        client.tasks.delete(task=task.id, force=True)


def task(migrant):
    # type: (Migrant) -> (Dict[str,List[str]],str,int)
    """
    <Description>

    :param Migrant migrant:
    """
    migrant.read()
    migrant.seed()
    return migrant.msgs, migrant.project_link, migrant.migration_count

def print_failures(l):
    # type: (List[List[str]]) -> ()
    """
    <Description>

    :param List[List[str]] l:
    """
    new_l = (m for sl in l for m in sl)
    for failed in new_l:
        print(failed)
def print_errors(l):
    # type: (List[List[str]]) -> ()
    """
    <Description>

    :param List[List[str]] l:
    """
    new_l = (m for sl in l for m in sl)
    for error in new_l:
        print(error)

def main(path, analysis):
    # type: (str,bool) -> ()
    """
    <Description>

    :param str path:
    :param bool analysis:
    """
    timer = Timer()
    project_link = None
    error_list = []
    failure_list = []
    workers = multiprocessing.cpu_count()
    migrant_factory = MigrantFactory(path)
    l, ids_count = migrant_factory.get_runs()
    chunk_size = math.ceil(ids_count / workers)

    completed_migrations = 0
    project_indicator = False;
    project_list = Task.get_projects()
    for project in project_list:
        if project.name == "mlflow migration":
            project_indicator = True;
            break;

    if chunk_size==0:
        print("No experiments to migrate")
        return
    else:
        jobs = chunks(l, chunk_size)
        futures = []
        text = 'progressbar'
        with tqdm(total=ids_count, desc=text) as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for job_chunk in jobs:
                    migrant = migrant_factory.create(job_chunk,pbar,timer,analysis,project_indicator)
                    future = executor.submit(task, migrant)
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    try:
                        msgs, project_link_per_task, migration_count = future.result()
                        completed_migrations+=migration_count
                        failure_list.append(msgs["FAILED"])
                        error_list.append(msgs["ERROR"])
                        if not project_link:
                            project_link = project_link_per_task
                    except Exception as e:
                        tb1 = traceback.TracebackException.from_exception(e)
                        error_list.append(["Error: " + ''.join(tb1.format())])
    close()
    print_failures(failure_list);
    print_errors(error_list);

    if completed_migrations == 0:
        print("Failed to migrate experiments")
    else:
        print(completed_migrations, "experiments succeeded to migrate")
    print("Link to the migrated project: ", project_link)
    timer.print_times()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migration from Mflow to Trains")
    parser.add_argument(
        "Path", metavar="path", type=str, help="path or address to MLFlow server (mlruns folder on local machine)"
    )

    parser.add_argument(
        "-a","--analysis", help="print analysis information", action="store_true",default=False
    )

    args = parser.parse_args()

    main(args.Path, analysis = args.analysis)