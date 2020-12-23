# ClearML - Example of manual graphs and  statistics reporting
#

import pandas as pd

from clearml import Task, Logger


def report_table(logger, iteration=0):
    # type: (Logger, int) -> ()
    """
    reporting tables to the plots section
    :param logger: The task.logger to use for sending the plots
    :param iteration: The iteration number of the current reports
    """
    # report tables

    # Report table - DataFrame with index
    df = pd.DataFrame(
        {
            "num_legs": [2, 4, 8, 0],
            "num_wings": [2, 0, 0, 0],
            "num_specimen_seen": [10, 2, 1, 8],
        },
        index=["falcon", "dog", "spider", "fish"],
    )
    df.index.name = "id"
    logger.report_table("table pd", "PD with index", iteration=iteration, table_plot=df)

    # Report table - CSV from path
    csv_url = "https://raw.githubusercontent.com/plotly/datasets/master/Mining-BTC-180.csv"
    logger.report_table("table - csv", "remote csv", iteration=iteration, url=csv_url)

    # Report table - Created with list of lists
    logger.report_table("table - list of lists",
                        "List of lists",
                        iteration=iteration,
                        table_plot=[
                            ["Last Name", "First Name", "Age"],
                            ["Doe", "John", "25"],
                            ["Smith", "Peter", "30"],
                        ])


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="examples", task_name="table reporting")

    print('reporting pandas tables and python lists as tables into the plots section')

    # Get the task logger,
    # You can also call Task.current_task().get_logger() from anywhere in your code.
    logger = task.get_logger()

    # report graphs
    report_table(logger)

    # force flush reports
    # If flush is not called, reports are flushed in the background every couple of seconds,
    # and at the end of the process execution
    logger.flush()

    print('We are done reporting, have a great day :)')


if __name__ == "__main__":
    main()
