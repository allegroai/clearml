# ClearML - Example of Matplotlib and Seaborn integration and reporting
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='Matplotlib example')

# Create a plot
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# Plot will be reported automatically
plt.show()

# Alternatively, in order to report the plot with a more meaningful title/series and iteration number
area = (40 * np.random.rand(N))**2
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
task.logger.report_matplotlib_figure(title="My Plot Title", series="My Plot Series", iteration=10, figure=plt)
plt.show()

# Create another plot - with a name
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color='black')
# Plot will be reported automatically
plt.show()

# Create image plot
m = np.eye(256, 256, dtype=np.uint8)
plt.imshow(m)
# Plot will be reported automatically
plt.show()

# Create image plot - with a name
m = np.eye(256, 256, dtype=np.uint8)
plt.imshow(m)
plt.title('Image Title')
# Plot will be reported automatically
plt.show()

sns.set(style="darkgrid")
# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")
# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri)
# Plot will be reported automatically
plt.show()

print('This is a Matplotlib & Seaborn example')
