# ClearML - Example of manual graphs and statistics reporting
#
import numpy as np

from clearml import Task, Logger


def report_plots(logger, iteration=0):
    # type: (Logger, int) -> ()
    """
    reporting plots to plots section
    :param logger: The task.logger to use for sending the plots
    :param iteration: The iteration number of the current reports
    """

    # report 3d surface
    surface = np.random.randint(10, size=(10, 10))
    logger.report_surface(
        "example_surface",
        "series1",
        iteration=iteration,
        matrix=surface,
        xaxis="title X",
        yaxis="title Y",
        zaxis="title Z",
    )

    # report 3d scatter plot
    scatter3d = np.random.randint(10, size=(10, 3))
    logger.report_scatter3d(
        "example_scatter_3d",
        "series_xyz",
        iteration=iteration,
        scatter=scatter3d,
        xaxis="title x",
        yaxis="title y",
        zaxis="title z",
    )


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="examples", task_name="3D plot reporting")

    print('reporting 3D plot graphs')

    # Get the task logger,
    # You can also call Task.current_task().get_logger() from anywhere in your code.
    logger = task.get_logger()

    # report graphs
    report_plots(logger)

    # force flush reports
    # If flush is not called, reports are flushed in the background every couple of seconds,
    # and at the end of the process execution
    logger.flush()

    print('We are done reporting, have a great day :)')


if __name__ == "__main__":
    main()
