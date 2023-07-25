# ClearML Explicit Logging

Using the **ClearML** [Logger](https://github.com/allegroai/clearml/blob/master/clearml/logger.py) module and other **ClearML** features, you can explicitly log any of the following:

* Report graphs and images
  * [Scalar metrics](#scalar-metrics)
  * [Histograms](#histograms)
  * [Line plots](#line-plots)
  * [2D scatter diagrams](#2d-scatter-diagrams)
  * [3D scatter diagrams](#3d-scatter-diagrams)
  * [Confusion matrices](#confusion-matrices)
  * [Surface diagrams](#surface-diagrams)
  * [Images](#images)
  
* Track hyperparameters and OS environment variables
  * Logging experiment parameter [dictionaries](#logging-experiment-parameter-dictionaries)
  * Specifying [environment variables](#specifying-environment-variables-to-track) to track

* Message logging
  * [Reporting text without formatting](#reporting-text-without-formatting) 
  
Additionally, the **ClearML** Logger module provides methods that allow you to do the following:
    
  * Get the [current logger]()
  * Overrride the ClearML configuration file with a [default upload destination]() for images and files
  
## Graphs and Images

### Scalar Metrics

Use to report scalar metrics by iteration as a line plot.

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/scalar_reporting.py)) with the following method.

**Method**:

```python
def report_scalar(self, title, series, value, iteration)
```

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>iteration
            </td>
            <td>integer
            </td>
            <td>The iteration number (x-axis).
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>series
            </td>
            <td>string
            </td>
            <td>The data series label.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>title
            </td>
            <td>string
            </td>
            <td>The plot title.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>value
            </td>
            <td>float
            </td>
            <td>The scalar metric data value (y-axis).
            </td>
            <td>Yes
            </td>
        </tr>
    </tbody>
</table>

### Histograms

Use to report any data by iteration as a histogram.

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/scatter_hist_confusion_mat_reporting.py)) with the following method.

**Method**:

```python
def report_histogram(self, title, series, values, iteration, labels=None, xlabels=None)
```

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>iteration
            </td>
            <td>integer
            </td>
            <td>The iteration number (x-axis).
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>series
            </td>
            <td>string
            </td>
            <td>The data series label.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>title
            </td>
            <td>string
            </td>
            <td>The plot title.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>values
            </td>
            <td>Either:
                <ul>
                    <li>list of float
                    </li>
                    <li>numpy array
                    </li>
                </ul>
            </td>
            <td>The histogram data values (y-axis).
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>labels
            </td>
            <td>list of strings
            </td>
            <td>Labels for each bar group in the histogram. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>xlabels
            </td>
            <td>list of strings
            </td>
            <td>Labels for each bucket in the histogram. Each label in the <code>xlabels</code> list corresponds to a value in the <code>values</code> list (or numpy array). The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
    </tbody>
</table>


### Line Plots

Use to report any data by iteration as a single or multiple line plot.

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/scatter_hist_confusion_mat_reporting.py)) with the following method.

**Method**:

```python
def report_line_plot(self, title, series, iteration, xaxis, yaxis, mode='lines', reverse_xaxis=False, comment=None)
```

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>iteration
            </td>
            <td>integer
            </td>
            <td>The iteration number (x-axis).
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>series
            </td>
            <td>LineSeriesInfo
            </td>
            <td>One (single line plot) or more (multiple line plot) series of data.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>title
            </td>
            <td>string
            </td>
            <td>The plot title.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>comment
            </td>
            <td>string
            </td>
            <td>Any text (e.g., subtitle or other comment) which displays under the plot title. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>mode
            </td>
            <td>string
            </td>
            <td>Type of line plot. The values are:
                <ul>
                    <li><code>lines</code> - lines connecting data points (default)
                    </li>
                    <li><code>markers</code> - markers for each data point
                    </li>
                    <li><code>lines+markers</code> - lines and markers
                    </li>
                </ul>
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>reverse_xaxis
            </td>
            <td>boolean
            </td>
            <td>Indicates whether to display the x-axis values in ascending or descending order. The values are:
                <ul>
                    <li><code>True</code> - descending order
                    </li>
                    <li><code>False</code> - ascending order (default)
                    </li>
                </ul>
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>xaxis
            </td>
            <td>string
            </td>
            <td>x-axis title.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>yaxis
            </td>
            <td>string
            </td>
            <td>y-axis title.
            </td>
            <td>No
            </td>
        </tr>
    </tbody>
</table>


### 2D Scatter Diagrams

Use to report any vector data as a 2D scatter diagram.

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/scatter_hist_confusion_mat_reporting.py)) with the following method.

**Method**:

```python
def report_scatter2d(self, title, series, scatter, iteration, xaxis=None, yaxis=None, labels=None, mode='lines', comment=None)
```

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>iteration
            </td>
            <td>integer
            </td>
            <td>The iteration number (x-axis).
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>scatter
            </td>
            <td>Either:
                <ul>
                    <li>list of (pairs of x, y)
                    </li>
                    <li>ndarray
                    </li>
                </ul>
            </td>
            <td>The scatter data. For example, a list of pairs in the form: <code>[(x1,y1),(x2,y2),(x3,y3),...]</code>.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>series
            </td>
            <td>string
            </td>
            <td>The data series label.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>title
            </td>
            <td>string
            </td>
            <td>The plot title.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>comment
            </td>
            <td>string
            </td>
            <td>Any text (e.g., subtitle or other comment) which displays under the plot title. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>labels
            </td>
            <td>list of strings
            </td>
            <td>Label text per data point in the scatter diagram. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>mode
            </td>
            <td>string
            </td>
            <td>Type of 2D scatter diagram. The values are:
                <ul>
                    <li><code>lines</code> - lines connecting data points (default)
                    </li>
                    <li><code>markers</code> - markers for each data point
                    </li>
                    <li><code>lines+markers</code> - lines and markers
                    </li>
                </ul>
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>xaxis
            </td>
            <td>string
            </td>
            <td>x-axis title. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>yaxis
            </td>
            <td>string
            </td>
            <td>y-axis title. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
    </tbody>
</table>


### 3D Scatter Diagrams

Use to report any array data as a 3D scatter diagram.

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/3d_plots_reporting.py)) with the following method.

**Method**:

```python
def report_scatter3d(self, title, series, scatter, iteration, labels=None, mode='markers', fill=False, comment=None)
```

**Argument**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>iteration
            </td>
            <td>integer
            </td>
            <td>The iteration number (x-axis).
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>scatter
            </td>
            <td>Either:
                <ul>
                    <li>list of (pairs of x, y, z)
                    </li>
                    <li>ndarray
                    </li>
                    <li>list of series [[(x1,y1,z1)...]]
                    </li>
                </ul>
            </td>
            <td>The scatter data. For example, a list of series in the form: 
            <code>[[(x1,y1,z1),(x2,y2,z2),...],[(x3,y3,z3),(x4,y4,z4),...],[(x5,y5,z5),(x6,y6,z6),...]]</code>
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>series
            </td>
            <td>string
            </td>
            <td>The data series label.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>title
            </td>
            <td>string
            </td>
            <td>The plot title.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>comment
            </td>
            <td>string
            </td>
            <td>Any text (e.g., subtitle or other comment) which displays under the plot title. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>fill
            </td>
            <td>boolean
            </td>
            <td>Indicates whether to fill the area under the scatter diagram. The values are:
                <ul>
                    <li><code>True</code> - fill
                    </li>
                    <li><code>False</code> - do not fill (default)
                    </li>
                </ul>
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>labels
            </td>
            <td>list of strings
            </td>
            <td>Label text per data point in the scatter. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>mode
            </td>
            <td>string
            </td>
            <td>Type of 3D scatter diagram. The values are:
                <ul>
                    <li><code>lines</code>
                    </li>
                    <li><code>markers</code>
                    </li>
                    <li><code>lines+markers</code>
                    </li>
                </ul>
            The default values is <code>lines</code>.
            </td>
            <td>No
            </td>
        </tr>
    </tbody>
</table>


### Confusion Matrices

Use to report a heat-map matrix as a confusion matrix. You can also plot a heat-map as a [surface diagram](#surface-diagrams).

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/scatter_hist_confusion_mat_reporting.py)) with the following method.

**Method**:

```python
def report_confusion_matrix(self, title, series, matrix, iteration, xlabels=None, ylabels=None, comment=None)
```

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>iteration
            </td>
            <td>integer
            </td>
            <td>The iteration number (x-axis).
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>matrix
            </td>
            <td>ndarray
            </td>
            <td>A heat-map matrix.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>series
            </td>
            <td>string
            </td>
            <td>The data series label.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>title
            </td>
            <td>string
            </td>
            <td>The plot title.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>xlabels
            </td>
            <td>list of strings
            </td>
            <td>Label per column of the matrix. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>ylabels
            </td>
            <td>list of strings
            </td>
            <td>Label per row of the matrix. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
    </tbody>
</table>


### Surface Diagrams

Use to plot a heat-map matrix as a surface diagram. You can also plot a heat-map as a [confusion matrix](#confusion-matrices).

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/3d_plots_reporting.py)) with the following method.

**Method**:

```python
def report_surface(self, title, series, matrix, iteration, xlabels=None, ylabels=None, xaxis=None, yaxis=None, camera=None, comment=None)
```

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>iteration
            </td>
            <td>integer
            </td>
            <td>The iteration number (x-axis).
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>matrix
            </td>
            <td>ndarray
            </td>
            <td>A heat-map matrix.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>series
            </td>
            <td>string
            </td>
            <td>The data series label.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>title
            </td>
            <td>string
            </td>
            <td>The plot title.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>camera
            </td>
            <td>tuple
            </td>
            <td>The position of the camera as (x, y, x), if applicable.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>comment
            </td>
            <td>string
            </td>
            <td>Any text (e.g., subtitle or other comment) which displays under the plot title. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>xlabels
            </td>
            <td>list of strings
            </td>
            <td>Label per column of the matrix. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>xaxis
            </td>
            <td>string
            </td>
            <td>x-axis title.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>ylabels
            </td>
            <td>list of strings
            </td>
            <td>Label per row of the matrix. The default value is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>yaxis
            </td>
            <td>string
            </td>
            <td>y-axis title.
            </td>
            <td>No
            </td>
        </tr>
    </tbody>
</table>

### Images

Use to report an image and upload its contents to the bucket specified in the **ClearML** configuration file,
or a [default upload destination](#set-default-upload-destination), if you set a default. 

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/manual_reporting.py)) with the following method.

**Method**:

```python
def report_image(self, title, series, iteration, local_path=None, matrix=None, max_image_history=None, delete_after_upload=False)
```

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>iteration
            </td>
            <td>integer
            </td>
            <td>The iteration number.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>series
            </td>
            <td>string
            </td>
            <td>The label of the series.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>title
            </td>
            <td>string
            </td>
            <td>The title of the image.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>delete_after_upload
            </td>
            <td>boolean
            </td>
            <td>Indicates whether to delete the local copy of the file after uploading it. The values are:
                <ul>
                    <li><code>True</code> - delete
                    </li>
                    <li><code>False</code> - do not delete (default)
                    </li>
                </ul>
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>matrix
            </td>
            <td>ndarray
            </td>
            <td>A 3D numpy.ndarray object containing image data (RGB). If <code>path</code> is not specified, then <code>matrix</code> is required. The default values is <code>None</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>max_image_history
            </td>
            <td>integer
            </td>
            <td>The maximum number of images to store per metric / variant combination. For an unlimited number of images to store, specify a negative number.
            The default value which is set in the global configuration is <code>5</code>.
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>local_path
            </td>
            <td>string
            </td>
            <td>The path of the image file. If <code>matrix</code> is not specified, then <code>path</code> is required. The default values is <code>None</code>. 
            </td>
            <td>No
            </td>
        </tr>
    </tbody>
</table>

## Hyperparameters and Environment Variables

### Logging Experiment Parameter Dictionaries

In order for **ClearML** to log a dictionary of parameters, use the `Task.connect` method.

For example, to log the hyperparameters <code>learning_rate</code>, <code>batch_size</code>, <code>display_step</code>, <code>model_path</code>, <code>n_hidden_1</code>, and <code>n_hidden_2</code>:  

```python
# Create a dictionary of parameters
parameters_dict = { 'learning_rate': 0.001, 'batch_size': 100, 'display_step': 1, 
    'model_path': "/tmp/model.ckpt", 'n_hidden_1': 256, 'n_hidden_2': 256 }
    
# Connect the dictionary to your ClearML Task    
parameters_dict = Task.current_task().connect(parameters_dict)
```

### Specifying Environment Variables to Track

By setting the `CLEARML_LOG_ENVIRONMENT` environment variable, make **ClearML** log either:

* All environment variables

        export CLEARML_LOG_ENVIRONMENT="*"

* Specific environment variables

    For example, log `PWD` and `PYTHONPATH`

        export CLEARML_LOG_ENVIRONMENT="PWD,PYTHONPATH"

* No environment variables

        export CLEARML_LOG_ENVIRONMENT=

## Logging Messages

Use the methods in this section to log various types of messages. The method name describes the type of message.

### Debugging Messages

**Method**:

```python
def debug(self, msg, *args, **kwargs)
```

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/text_reporting.py)) with the following method.

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
        <tr>
            <td>msg
            </td>
            <td>string 
            </td>
            <td>The text to log.
            </td>
            <td>Yes
            </td>
        </tr>
    </tbody>
</table>

### Informational Messages

**Method**:

```python
def info(self, msg, *args, **kwargs)
```

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/text_reporting.py)) with the following method.

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
        <tr>
            <td>msg
            </td>
            <td>string 
            </td>
            <td>The text to log.
            </td>
            <td>Yes
            </td>
        </tr>
    </tbody>
</table>

### Warnings

**Method**:

```python
def warn(self, msg, *args, **kwargs)
```

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/text_reporting.py)) with the following method.

**Arguments**:<a name="log_arguments"></a>

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>msg
            </td>
            <td>string 
            </td>
            <td>The text to log.
            </td>
            <td>Yes
            </td>
        </tr>
    </tbody>
</table>

### General Errors

**Method**:

```python
def error(self, msg, *args, **kwargs)
```

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/text_reporting.py)) with the following method.

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
        <tr>
            <td>msg
            </td>
            <td>string 
            </td>
            <td>The text to log.
            </td>
            <td>Yes
            </td>
        </tr>
    </tbody>
</table>

### Critical Errors

**Method**:

```python
def critical(self, msg, *args, **kwargs)
```

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/text_reporting.py)) with the following method.

**Arguments**:
    
<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
        <tr>
            <td>msg
            </td>
            <td>string 
            </td>
            <td>The text to log.
            </td>
            <td>Yes
            </td>
        </tr>
    </tbody>
</table>
        
### Fatal Errors

**Method**:

```python
def fatal(self, msg, *args, **kwargs)
```

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/text_reporting.py)) with the following method.

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
        <tr>
            <td>msg
            </td>
            <td>string 
            </td>
            <td>The text to log.
            </td>
            <td>Yes
            </td>
        </tr>
    </tbody>
</table>

### Console and Logger Messages

**Method**:

```python
def console(self, msg, level=logging.INFO, omit_console=False, *args, **kwargs)
```

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/text_reporting.py)) with the following method.

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
        <tr>
            <td>msg
            </td>
            <td>string 
            </td>
            <td>The text to log.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>level
            </td>
            <td>string
            </td>
            <td>The log level. The values are:
                <ul>
                    <li><code>logging.DEBUG</code>
                    </li>
                    <li><code>logging.INFO</code>
                    </li>
                    <li><code>logging.WARNING</code>
                    </li>
                    <li><code>logging.ERROR</code>
                    </li>
                    <li><code>logging.FATAL</code>
                    </li>
                    <li><code>logging.CRITICAL</code>
                    </li>
                </ul>
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>omit_console
            </td>
            <td>boolean
            </td>
            <td>Indicates whether to send the message to the log only. The values are:
                <ul>
                    <li><code>True</code> - send the message to the log only
                    </li>
                    <li><code>False</code> - send the message to the console and the log
                    </li>
                </ul>
            </td>
            <td>No
            </td>
        </tr>
    </tbody>
</table>

### Reporting Text Without Formatting 

**Method**:

```python
def report_text(self, msg, level=logging.INFO, print_console=False, *args, **_)
```

First [get the current logger](#get-the-current-logger) and then use it (see an [example script](https://github.com/allegroai/clearml/blob/master/examples/reporting/text_reporting.py)) with the following method.

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>msg
            </td>
            <td>string 
            </td>
            <td>The text to log.
            </td>
            <td>Yes
            </td>
        </tr>
        <tr>
            <td>level
            </td>
            <td>string
            </td>
            <td>The log level. The values are:
                <ul>
                    <li><code>logging.DEBUG</code>
                    </li>
                    <li><code>logging.INFO</code>
                    </li>
                    <li><code>logging.WARNING</code>
                    </li>
                    <li><code>logging.ERROR</code>
                    </li>
                    <li><code>logging.FATAL</code>
                    </li>
                    <li><code>logging.CRITICAL</code>
                    </li>
                </ul>
            </td>
            <td>No
            </td>
        </tr>
        <tr>
            <td>print_console
            </td>
            <td>boolean
            </td>
            <td>Indicates whether to log to the console, in addition to the log. The values are:
                <ul>
                    <li><code>True</code> - print to the console and log
                    </li>
                    <li><code>False</code> - print to the log, only (default)
                    </li>
                </ul>
            </td>
            <td>No
            </td>
        </tr>
    </tbody>
</table>

## Logger Object and Storage Methods

### Get the Current Logger

Use to return a reference to the current logger object.

**Method**:

```python
def current_logger(cls)
```

**Arguments**:

None.

### Set Default Upload Destination

Use to specify the default destination storage location used for uploading images.
Images are uploaded and a link to the image is reported.

Credentials for the storage location are in the global configuration file (for example, on Linux, <code>~/clearml.conf</code>). 

**Method**:

```python
def set_default_upload_destination(self, uri)
```

**Arguments**:

<table width="100%">
    <thead>
        <tr>
            <th width="15%">Parameter
            </th>
            <th width="25%">Type
            </th>
            <th width="55%">Description
            </th>
            <th width="5%">Mandatory
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>uri
            </td>
            <td>string
            </td>
            <td>The destination storage location, for example <code>s3://bucket/directory/</code> or <code>file:///tmp/debug/</code>.
            </td>
            <td>Yes
            </td>
        </tr>
    </tbody>
</table>

