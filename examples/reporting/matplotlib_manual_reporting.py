# ClearML - Example of Matplotlib and Seaborn integration and reporting
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
# Create a new task, disable automatic matplotlib connect
task = Task.init(
    project_name="examples",
    task_name="Manual Matplotlib example",
    auto_connect_frameworks={"matplotlib": False},
)

# Create plot and explicitly report as figure
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
task.logger.report_matplotlib_figure(
    title="Manual Reporting", series="Just a plot", iteration=0, figure=plt
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
# Show the plot
plt.show()

# Create image plot
m = np.eye(256, 256, dtype=np.uint8)
plt.imshow(m)
# Report plot
task.logger.report_matplotlib_figure(
    title="Manual Reporting",
    series="Image plot",
    iteration=0,
    figure=plt,
    report_interactive=False,
)
# Show plot
plt.show()

# Create Seaborn plot
sns.set(style="darkgrid")
# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")
# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal", hue="region", style="event", data=fmri)
# Report plot
task.logger.report_matplotlib_figure(
    title="Seaborn example",
    series="My Plot Series 4",
    iteration=10,
    figure=plt,
    report_interactive=False,
)
# Show plot
plt.show()

print("This is a Matplotlib & Seaborn example")
