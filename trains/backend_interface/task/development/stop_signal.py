from ....config import config
from ....backend_interface import Task, TaskStatusEnum


class TaskStopReason(object):
    stopped = "stopped"
    reset = "reset"
    status_changed = "status_changed"


class TaskStopSignal(object):
    enabled = bool(config.get('development.support_stopping', False))

    _number_of_consecutive_reset_tests = 4

    _unexpected_statuses = (
        TaskStatusEnum.closed,
        TaskStatusEnum.stopped,
        TaskStatusEnum.failed,
        TaskStatusEnum.published,
    )

    def __init__(self, task):
        assert isinstance(task, Task)
        self.task = task
        self._task_reset_state_counter = 0

    def test(self):
        status = self.task.status
        message = self.task.data.status_message

        if status == TaskStatusEnum.in_progress and "stopping" in message:
            return TaskStopReason.stopped

        if status in self._unexpected_statuses and "worker" not in message:
            return TaskStopReason.status_changed

        if status == TaskStatusEnum.created:
            self._task_reset_state_counter += 1

            if self._task_reset_state_counter >= self._number_of_consecutive_reset_tests:
                return TaskStopReason.reset

            self.task.get_logger().warning(
                "Task {} was reset! if state is consistent we shall terminate.".format(self.task.id),
            )
        else:
            self._task_reset_state_counter = 0
