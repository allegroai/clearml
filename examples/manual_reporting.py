# TRAINS - Example of manual graphs and  statistics reporting
#
import numpy as np
import logging
from trains import Task


task = Task.init(project_name='examples', task_name='Manual reporting')

# example python logger
logging.getLogger().setLevel('DEBUG')
logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')

# get TRAINS logger object for any metrics / reports
logger = task.get_logger()

# log text
logger.console("hello")

# report scalar values
logger.report_scalar("example_scalar", "series A", iteration=0, value=100)
logger.report_scalar("example_scalar", "series A", iteration=1, value=200)

# report histogram
histogram = np.random.randint(10, size=10)
logger.report_vector("example_histogram", "random histogram", iteration=1, values=histogram)

# report confusion matrix
confusion = np.random.randint(10, size=(10, 10))
logger.report_matrix("example_confusion", "ignored", iteration=1, matrix=confusion)

# report 2d scatter plot
scatter2d = np.hstack((np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1))))
logger.report_scatter2d("example_scatter", "series_xy", iteration=1, scatter=scatter2d)

# report 3d scatter plot
scatter3d = np.random.randint(10, size=(10, 3))
logger.report_scatter3d("example_scatter_3d", "series_xyz", iteration=1, scatter=scatter3d)

# report image
m = np.eye(256, 256, dtype=np.uint8)*255
logger.report_image_and_upload("test case", "image uint", iteration=1, matrix=m)
m = np.eye(256, 256, dtype=np.float)
logger.report_image_and_upload("test case", "image float", iteration=1, matrix=m)

# flush reports (otherwise it will be flushed in the background, every couple of seconds)
logger.flush()
