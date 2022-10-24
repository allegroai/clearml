# ClearML - Example of manual graphs and  statistics reporting
#
import os

import numpy as np
from PIL import Image

from clearml import Task, Logger


def report_debug_images(logger, iteration=0):
    # type: (Logger, int) -> ()
    """
    reporting images to debug samples section
    :param logger: The task.logger to use for sending the plots
    :param iteration: The iteration number of the current reports
    """

    # report image as float image
    m = np.eye(256, 256, dtype=float)
    logger.report_image("image", "image float", iteration=iteration, image=m)

    # report image as uint8
    m = np.eye(256, 256, dtype=np.uint8) * 255
    logger.report_image("image", "image uint8", iteration=iteration, image=m)

    # report image as uint8 RGB
    m = np.concatenate((np.atleast_3d(m), np.zeros((256, 256, 2), dtype=np.uint8)), axis=2)
    logger.report_image("image", "image color red", iteration=iteration, image=m)

    # report PIL Image object
    image_open = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_samples", "picasso.jpg"))
    logger.report_image("image", "image PIL", iteration=iteration, image=image_open)

    # Image can be uploaded via 'report_media' too.
    logger.report_media(
        "image",
        "image with report media",
        iteration=iteration,
        local_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_samples", "picasso.jpg"),
        file_extension="jpg",
    )


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="examples", task_name="Image reporting")

    print('reporting a few debug images')

    # Get the task logger,
    # You can also call Task.current_task().get_logger() from anywhere in your code.
    logger = task.get_logger()

    # report debug images
    report_debug_images(logger)

    # force flush reports
    # If flush is not called, reports are flushed in the background every couple of seconds,
    # and at the end of the process execution
    logger.flush()

    print('We are done reporting, have a great day :)')


if __name__ == "__main__":
    main()
