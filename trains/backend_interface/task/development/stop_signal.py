from ....config import config
from ....backend_api.services import tasks


class TaskStopReason(object):
    stopped = "stopped"
    reset = "reset"
    status_changed = "status_changed"


class TaskStopSignal(object):
    enabled = bool(config.get('development.support_stopping', False))

    _number_of_consecutive_reset_tests = 4

    # _unexpected_statuses = (
    #     tasks.TaskStatusEnum.closed,
    #     tasks.TaskStatusEnum.stopped,
    #     tasks.TaskStatusEnum.failed,
    #     tasks.TaskStatusEnum.published,
    #     tasks.TaskStatusEnum.completed,
    # )

    def __init__(self, task):
        from ....backend_interface import Task
        assert isinstance(task, Task)
        self.task = task
        self._task_reset_state_counter = 0

    def test(self):
        # noinspection PyBroadException
        try:
            # we use internal status read, so we do not need to constantly pull the entire task object,
            # it might be large, and we want to also avoid the edit lock on it.
            status, message = self.task._get_status()
            status = str(status)
            message = str(message)

            if status == str(tasks.TaskStatusEnum.in_progress) and "stopping" in message:
                # make sure we syn the entire task object
                self.task.reload()
                return TaskStopReason.stopped

            _expected_statuses = (
                str(tasks.TaskStatusEnum.created),
                str(tasks.TaskStatusEnum.queued),
                str(tasks.TaskStatusEnum.in_progress),
            )

            if status not in _expected_statuses and "worker" not in message:
                # make sure we syn the entire task object
                self.task.reload()
                return TaskStopReason.status_changed

            if status == str(tasks.TaskStatusEnum.created):
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
