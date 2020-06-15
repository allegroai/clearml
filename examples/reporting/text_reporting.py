# TRAINS - Example of manual graphs and  statistics reporting
#
import logging
import sys

from trains import Task, Logger


def report_logs(logger):
    # type: (Logger) -> ()
    """
    reporting text to logs section
    :param logger: The task.logger to use for sending the text
    """
    # standard python logging
    logging.info("This is an info message")

    # this is a loguru test example
    try:
        from loguru import logger as loguru_logger # noqa

        loguru_logger.info("That's it, beautiful and simple logging! (using ANSI colors)")
    except ImportError:
        print('loguru not installed, skipping loguru test')

    # report text
    logger.report_text("hello, this is plain text")


def main():
    # Create the experiment Task
    task = Task.init(project_name="examples", task_name="text reporting")

    print('reporting text logs')

    # report regular console print
    print('This is standard output test')

    # report stderr
    print('This is standard error test', file=sys.stderr)

    # Get the task logger,
    # You can also call Task.current_task().get_logger() from anywhere in your code.
    logger = task.get_logger()

    # report text based logs
    report_logs(logger)

    # force flush reports
    # If flush is not called, reports are flushed in the background every couple of seconds,
    # and at the end of the process execution
    logger.flush()

    print('We are done reporting, have a great day :)')


if __name__ == "__main__":
    main()
