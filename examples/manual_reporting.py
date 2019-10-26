# TRAINS - Example of manual graphs and  statistics reporting
#
import os
from PIL import Image
import numpy as np
import logging
from trains import Task


task = Task.init(project_name="examples", task_name="Manual reporting")

# standard python logging
logging.info("This is an info message")

# this is loguru test example
try:
    from loguru import logger
    logger.info("That's it, beautiful and simple logging! (using ANSI colors)")
except ImportError:
    pass

# get TRAINS logger object for any metrics / reports
logger = Task.current_task().get_logger()

# log text
logger.report_text("hello")

# report scalar values
logger.report_scalar("example_scalar", "series A", iteration=0, value=100)
logger.report_scalar("example_scalar", "series A", iteration=1, value=200)

# report histogram
histogram = np.random.randint(10, size=10)
logger.report_histogram("example_histogram", "random histogram", iteration=1, values=histogram,
                        xaxis="title x", yaxis="title y")

# report confusion matrix
confusion = np.random.randint(10, size=(10, 10))
logger.report_matrix("example_confusion", "ignored", iteration=1, matrix=confusion, xaxis="title X", yaxis="title Y")

# report 3d surface
logger.report_surface("example_surface", "series1", iteration=1, matrix=confusion,
                      xaxis="title X", yaxis="title Y", zaxis="title Z")

# report 2d scatter plot
scatter2d = np.hstack((np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1))))
logger.report_scatter2d("example_scatter", "series_xy", iteration=1, scatter=scatter2d,
                        xaxis="title x", yaxis="title y")

# report 3d scatter plot
scatter3d = np.random.randint(10, size=(10, 3))
logger.report_scatter3d("example_scatter_3d", "series_xyz", iteration=1, scatter=scatter3d,
                        xaxis="title x", yaxis="title y", zaxis="title z")

# reporting images
m = np.eye(256, 256, dtype=np.float)
logger.report_image("test case", "image float", iteration=1, image=m)
m = np.eye(256, 256, dtype=np.uint8)*255
logger.report_image("test case", "image uint8", iteration=1, image=m)
m = np.concatenate((np.atleast_3d(m), np.zeros((256, 256, 2), dtype=np.uint8)), axis=2)
logger.report_image("test case", "image color red", iteration=1, image=m)
image_open = Image.open(os.path.join("samples", "picasso.jpg"))
logger.report_image("test case", "image PIL", iteration=1, image=image_open)
# flush reports (otherwise it will be flushed in the background, every couple of seconds)
logger.flush()
