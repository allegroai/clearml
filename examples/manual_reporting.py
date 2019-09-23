# TRAINS - Example of manual graphs and  statistics reporting
#
import numpy as np
import logging
from trains import Task


task = Task.init(project_name='examples', task_name='Manual reporting')

# standard python logging
logging.info('This is an info message')

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
logger.report_histogram("example_histogram", "random histogram", iteration=1, values=histogram)

# report confusion matrix
confusion = np.random.randint(10, size=(10, 10))
logger.report_matrix("example_confusion", "ignored", iteration=1, matrix=confusion)

# report 3d surface
logger.report_surface("example_surface", "series1", iteration=1, matrix=confusion,
                      xtitle='title X', ytitle='title Y', ztitle='title Z')

# report 2d scatter plot
scatter2d = np.hstack((np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1))))
logger.report_scatter2d("example_scatter", "series_xy", iteration=1, scatter=scatter2d)

# report 3d scatter plot
scatter3d = np.random.randint(10, size=(10, 3))
logger.report_scatter3d("example_scatter_3d", "series_xyz", iteration=1, scatter=scatter3d)

# reporting images
m = np.eye(256, 256, dtype=np.float)
logger.report_image("test case", "image float", iteration=1, matrix=m)
m = np.eye(256, 256, dtype=np.uint8)*255
logger.report_image("test case", "image uint8", iteration=1, matrix=m)
m = np.concatenate((np.atleast_3d(m), np.zeros((256, 256, 2), dtype=np.uint8)), axis=2)
logger.report_image("test case", "image color red", iteration=1, matrix=m)

# flush reports (otherwise it will be flushed in the background, every couple of seconds)
logger.flush()
