# ClearML - Example of manual single value scalars reporting
#
from clearml import Task


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="examples", task_name="Scalar reporting (Single Value)")

    # Get the task logger,
    # You can also call Task.current_task().get_logger() from anywhere in your code.
    logger = task.get_logger()

    # report scalars
    logger.report_single_value(name="metric_A", value=125)
    logger.report_single_value(name="metric_B", value=305.95)
    logger.report_single_value(name="metric_C", value=486)

    # force flush reports
    # If flush is not called, reports are flushed in the background every couple of seconds,
    # and at the end of the process execution
    logger.flush(wait=True)

    # get scalars
    # Getting one metric
    metric_B = task.get_reported_single_value('metric_B')
    print('metric_B is', metric_B)

    # Getting all metrics at once
    metric_all = task.get_reported_single_values()
    print('All metrics', metric_all)


if __name__ == "__main__":
    main()
