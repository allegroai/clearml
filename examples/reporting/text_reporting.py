# ClearML - Example of manual graphs and  statistics reporting
#
from __future__ import print_function

import logging
import sys

import six

from clearml import Logger, Task


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
        from loguru import logger as loguru_logger  # noqa

        loguru_logger.info(
            "That's it, beautiful and simple logging! (using ANSI colors)"
        )
    except ImportError:
        print("loguru not installed, skipping loguru test")

    # report text
    logger.report_text("hello, this is plain text")


def report_debug_text(logger):
    # type: (Logger) -> ()
    """
    reporting text to debug sample section
    :param logger: The task.logger to use for sending the sample
    """
    text_to_send = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Suspendisse ac justo ut dolor scelerisque posuere. 
Donec hendrerit, purus viverra congue maximus, neque orci vehicula elit, pulvinar elementum diam lorem ut arcu. 
Sed convallis ipsum justo. Duis faucibus consectetur cursus. Morbi eleifend nisl vel maximus dapibus. 
Vestibulum commodo justo eget tellus interdum dapibus. Curabitur pulvinar nibh vitae orci laoreet, id sodales justo ultrices. 
Etiam mollis dui et viverra ultrices. Vestibulum vitae molestie libero, quis lobortis risus. Morbi venenatis quis odio nec efficitur. 
Vestibulum dictum ipsum at viverra ultrices. Aliquam sed ante massa. Quisque convallis libero in orci fermentum tincidunt.
"""
    logger.report_media(
        title="text title",
        series="text series",
        iteration=1,
        stream=six.StringIO(text_to_send),
        file_extension=".txt",
    )


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="examples", task_name="text reporting")

    print("reporting text logs")

    # report regular console print
    print("This is standard output test")

    # report stderr
    print("This is standard error test", file=sys.stderr)

    # Get the task logger,
    # You can also call Task.current_task().get_logger() from anywhere in your code.
    logger = task.get_logger()

    # report text based logs
    report_logs(logger)

    # report text as debug example
    report_debug_text(logger)

    # force flush reports
    # If flush is not called, reports are flushed in the background every couple of seconds,
    # and at the end of the process execution
    logger.flush()

    print("We are done reporting, have a great day :)")


if __name__ == "__main__":
    main()
