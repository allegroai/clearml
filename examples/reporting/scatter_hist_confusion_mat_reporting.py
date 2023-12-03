# ClearML - Example of manual graphs and  statistics reporting
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

    # report a single histogram
    histogram = np.random.randint(10, size=10)
    logger.report_histogram(
        "single_histogram",
        "random histogram",
        iteration=iteration,
        values=histogram,
        xaxis="title x",
        yaxis="title y",
    )

    # report a two histograms on the same graph (plot)
    histogram1 = np.random.randint(13, size=10)
    histogram2 = histogram * 0.75
    logger.report_histogram(
        "two_histogram",
        "series 1",
        iteration=iteration,
        values=histogram1,
        xaxis="title x",
        yaxis="title y",
    )
    logger.report_histogram(
        "two_histogram",
        "series 2",
        iteration=iteration,
        values=histogram2,
        xaxis="title x",
        yaxis="title y",
    )

    # report confusion matrix
    confusion = np.random.randint(10, size=(10, 10))
    logger.report_confusion_matrix(
        "example_confusion",
        "ignored",
        iteration=iteration,
        matrix=confusion,
        xaxis="title X",
        yaxis="title Y",
    )

    # report confusion matrix with 0,0 is at the top left
    logger.report_confusion_matrix(
        "example_confusion_0_0_at_top",
        "ignored",
        iteration=iteration,
        matrix=confusion,
        xaxis="title X",
        yaxis="title Y",
        yaxis_reversed=True,
    )

    scatter2d = np.hstack(
        (np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1)))
    )
    # report 2d scatter plot with lines
    logger.report_scatter2d(
        "example_scatter",
        "series_xy",
        iteration=iteration,
        scatter=scatter2d,
        xaxis="title x",
        yaxis="title y",
    )

    scatter2d = np.hstack(
        (np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1)))
    )
    # report 2d scatter plot with markers
    logger.report_scatter2d(
        "example_scatter",
        "series_markers",
        iteration=iteration,
        scatter=scatter2d,
        xaxis="title x",
        yaxis="title y",
        mode='markers'
    )

    scatter2d = np.hstack(
        (np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1)))
    )
    # report 2d scatter plot with markers
    logger.report_scatter2d(
        "example_scatter",
        "series_lines+markers",
        iteration=iteration,
        scatter=scatter2d,
        xaxis="title x",
        yaxis="title y",
        mode='lines+markers'
    )


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="examples", task_name="2D plots reporting")

    print('reporting some graphs')

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
