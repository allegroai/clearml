from ....config import deferred_config


class TaskStopReason(object):
    stopped = "stopped"
    reset = "reset"
    status_changed = "status_changed"


class TaskStopSignal(object):
    enabled = deferred_config('development.support_stopping', False, transform=bool)

    _number_of_consecutive_reset_tests = 4

    def __init__(self, task):
        from ....backend_interface import Task
        assert isinstance(task, Task)
        self.task = task
        self._task_reset_state_counter = 0
        self._status_in_progress = str(Task.TaskStatusEnum.in_progress)
        self._status_created = str(Task.TaskStatusEnum.created)
        self._status_expected_statuses = (
                str(Task.TaskStatusEnum.created),
                str(Task.TaskStatusEnum.queued),
                str(Task.TaskStatusEnum.in_progress),
            )

    def test(self):
        # noinspection PyBroadException
        try:
            # we use internal status read, so we do not need to constantly pull the entire task object,
            # it might be large, and we want to also avoid the edit lock on it.
            status, message = self.task._get_status()
            # if we did not get a proper status, return and recheck later
            if status is None and message is None:
                return None

            status = str(status)
            message = str(message)

            if status == self._status_in_progress and "stopping" in message:
                # make sure we syn the entire task object
                self.task.reload()
                return TaskStopReason.stopped

            if status not in self._status_expected_statuses and "worker" not in message:
                # make sure we syn the entire task object
                self.task.reload()
                return TaskStopReason.status_changed

            if status == self._status_created:
                self._task_reset_state_counter += 1

                if self._task_reset_state_counter >= self._number_of_consecutive_reset_tests:
                    # make sure we syn the entire task object
                    self.task.reload()
                    return TaskStopReason.reset

                self.task.log.warning(
                    "Task {} was reset! if state is consistent we shall terminate.".format(self.task.id),
                )
            else:
                self._task_reset_state_counter = 0
        except Exception:
            return None
