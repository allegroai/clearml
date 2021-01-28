# ClearML - Example of Plotly integration and reporting
#
from clearml import Task
import plotly.express as px


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init('examples', 'plotly reporting')

print('reporting plotly figures')

# Iris dataset
df = px.data.iris()

# create complex plotly figure
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", marginal_y="rug", marginal_x="histogram")

# report the plotly figure
task.get_logger().report_plotly(title="iris", series="sepal", iteration=0, figure=fig)

print('done')
