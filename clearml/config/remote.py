from .defs import TASK_ID_ENV_VAR

running_remotely_task_id = TASK_ID_ENV_VAR.get()


def override_current_task_id(task_id):
    """
    Overrides the current task id to simulate remote running with a specific task.

    Use for testing and debug only.

    :param task_id: The task's id to use as the remote task.
        Pass None to simulate local execution.
    """

    global running_remotely_task_id
    running_remotely_task_id = task_id
    # make sure we change the cached value as well.
    import clearml
    clearml.config._running_remotely_task_id = task_id
