import os
from logging import getLogger
from time import sleep, time

from pathlib2 import Path


def get_torch_local_rank():
    """
    return the local rank of the process, notice local rank 0 does not mean global rank 0
    return None if no torch distributed is running
    """
    if os.environ.get("TORCHELASTIC_RUN_ID") is not None:
        # noinspection PyBroadException
        try:
            return int(os.environ.get("LOCAL_RANK"))
        except Exception:
            return None

    return None


def create_torch_distributed_anchor(task_id):
    """
    This will create a temporary file to pass the Task ID created by local_rank 0 of
    if None local rank 0 is calling this file, it

    Only call when running locally (i.e. without an agent),
    if running remotely there is no need to pass Task ID, it will be passed externally
    """
    local_file_name = ".clearml_torch_distributed_id"

    if get_torch_local_rank() != 0:
        return

    torch_dist_path = os.environ.get("TORCHELASTIC_ERROR_FILE")

    if not torch_dist_path:
        return

    # noinspection PyBroadException
    try:
        torch_dist_path = Path(torch_dist_path).parent.parent.parent
        # create the file
        with open(torch_dist_path / local_file_name, "wt") as f:
            f.write(str(task_id)+"\n")
    except Exception:
        # we failed for some reason?
        getLogger().warning("Failed creating torch task ID anchor file: {}".format(torch_dist_path))


def get_torch_distributed_anchor_task_id(timeout=None):
    """
    This will wait until a temporary file appears and read the Task ID created by local_rank 0 of

    Only call when running locally (i.e. without an agent),
    if running remotely there is no need to pass Task ID, it will be passed externally

    :return Task ID of the local task to report to
    """

    # check that we are not local rank 0
    _local_rank = get_torch_local_rank()
    if not _local_rank:
        return

    local_file_name = ".clearml_torch_distributed_id"

    torch_dist_path = os.environ.get("TORCHELASTIC_ERROR_FILE")
    if not torch_dist_path:
        return

    task_id = None
    # noinspection PyBroadException
    try:
        torch_dist_path = Path(torch_dist_path).parent.parent.parent / local_file_name

        tic = time()
        # wait until disturbed file exists
        while not torch_dist_path.is_file():
            # if we found nothing, return None
            if timeout is not None and time() - tic > timeout:
                getLogger().warning("Failed detecting rank zero clearml Task ID, creating a new Task")
                return None
            # wait
            sleep(0.25)

        # create the file
        with open(torch_dist_path, "rt") as f:
            task_id = f.read().strip(" \n")
    except Exception:
        # we failed for some reason?
        pass

    getLogger().warning("Torch Distributed Local Rank {} Task ID {} detected".format(_local_rank, task_id))
    return task_id
