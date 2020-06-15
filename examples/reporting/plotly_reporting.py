# TRAINS - Example of Plotly integration and reporting
#
from trains import Task
import plotly.express as px


task = Task.init('examples', 'plotly reporting')

print('reporting plotly figures')

# Iris dataset
df = px.data.iris()

# create complex plotly figure
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", marginal_y="rug", marginal_x="histogram")

# report the plotly figure
task.get_logger().report_plotly(title="iris", series="sepal", iteration=0, figure=fig)

print('done')
