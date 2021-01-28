# ClearML - Example of Matplotlib and Seaborn integration and reporting
#
import numpy as np
import matplotlib.pyplot as plt
from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
# Create a new task, disable automatic matplotlib connect
task = Task.init(
    project_name='examples',
    task_name='Manual Matplotlib example',
    auto_connect_frameworks={'matplotlib': False}
)

# Create plot and explicitly report as figure
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
task.logger.report_matplotlib_figure(
    title="Manual Reporting",
    series="Just a plot",
    iteration=0,
    figure=plt,
)

# Show the plot
plt.show()

# Create plot and explicitly report as an image
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
task.logger.report_matplotlib_figure(
    title="Manual Reporting",
    series="Plot as an image",
    iteration=0,
    figure=plt,
    report_image=True,
)


# Create an image plot and explicitly report (as an image)
m = np.eye(256, 256, dtype=np.uint8)
plt.imshow(m)
task.logger.report_matplotlib_figure(
    title="Manual Reporting",
    series="Image plot",
    iteration=0,
    figure=plt,
    report_image=True,  # Note this is required for image plots
)

# Show the plot
plt.show()

