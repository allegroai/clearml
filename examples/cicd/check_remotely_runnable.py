import sys
import time
from clearml import Task


def check_task_status(task_id, timeout=600):
    """Make sure the task can run by checking for iteration reports."""
    # Get the task object
    task = Task.get_task(task_id=task_id)
    start_time = time.time()
    if task:
        while time.time() - start_time < timeout:
            task_status = task.get_status()
            print(task_status)
            print(task.get_last_iteration())

            if task_status == 'queued':
                # If queued, just reset the timeout timer
                start_time = time.time()
            if task_status in ['failed', 'stopped']:
                raise ValueError("Task did not run correctly, check logs in webUI.")
            elif task_status == 'in_progress':
                # Try to get the first iteration metric
                if task.get_last_iteration() > 0:
                    task.mark_stopped()
                    task.set_archived(True)
                    return True
            time.sleep(5)
        raise ValueError('Triggered Timeout!')
    else:
        return f"Can not find task {task}.\n\n"


if __name__ == '__main__':
    check_task_status(sys.argv[1])
