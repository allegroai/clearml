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
            status = self.task.status
            message = self.task.data.status_message

            if status == tasks.TaskStatusEnum.in_progress and "stopping" in message:
                return TaskStopReason.stopped

            _expected_statuses = (
                tasks.TaskStatusEnum.created,
                tasks.TaskStatusEnum.queued,
                tasks.TaskStatusEnum.in_progress,
            )

            if status not in _expected_statuses and "worker" not in message:
                return TaskStopReason.status_changed

            if status == tasks.TaskStatusEnum.created:
                self._task_reset_state_counter += 1

                if self._task_reset_state_counter >= self._number_of_consecutive_reset_tests:
                    return TaskStopReason.reset

                self.task.get_logger().warning(
                    "Task {} was reset! if state is consistent we shall terminate.".format(self.task.id),
                )
            else:
                self._task_reset_state_counter = 0
        except Exception:
            return None
