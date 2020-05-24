import logging
import math
import warnings
from typing import Any, Sequence, Union, List, Optional, Tuple

import numpy as np
import six

try:
    import pandas as pd
except ImportError:
    pd = None
from PIL import Image
from pathlib2 import Path

from .backend_interface.logger import StdStreamPatch, LogFlusher
from .backend_interface.task import Task as _Task
from .backend_interface.task.development.worker import DevWorker
from .backend_interface.task.log import TaskHandler
from .backend_interface.util import mutually_exclusive
from .config import running_remotely, get_cache_dir, config
from .debugging.log import LoggerRoot
from .errors import UsageError
from .storage.helper import StorageHelper
from .utilities.plotly_reporter import SeriesInfo

# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning, module=__name__)


class Logger(object):
    """
    The ``Logger`` class is the Trains console log and metric statistics interface, and contains methods for explicit
    reporting.

    Explicit reporting extends Trains automagical capturing of inputs and output. Explicit reporting
    methods include scalar plots, line plots, histograms, confusion matrices, 2D and 3D scatter
    diagrams, text logging, tables, and image uploading and reporting.

    In the **Trains Web-App (UI)**, ``Logger`` output appears in the **RESULTS** tab, **LOG**, **SCALARS**,
    **PLOTS**, and **DEBUG IMAGES** sub-tabs. When you compare experiments, ``Logger`` output appears in the
    comparisons.

    .. warning::

       Do not construct Logger objects directly.

    You must get a Logger object before calling any of the other ``Logger`` class methods by calling
    :meth:`.Task.get_logger` or :meth:`Logger.current_logger`.


    """
    SeriesInfo = SeriesInfo
    _tensorboard_logging_auto_group_scalars = False
    _tensorboard_single_series_per_graph = config.get('metrics.tensorboard_single_series_per_graph', False)

    def __init__(self, private_task):
        """
        .. warning::
            **Do not construct Logger manually!**
            Please use :meth:`Logger.get_current`
        """
        assert isinstance(private_task, _Task), \
            'Logger object cannot be instantiated externally, use Logger.current_logger()'
        super(Logger, self).__init__()
        self._task = private_task
        self._default_upload_destination = None
        self._flusher = None
        self._report_worker = None
        self._task_handler = None
        self._graph_titles = {}

        StdStreamPatch.patch_std_streams(self)

    @classmethod
    def current_logger(cls):
        # type: () -> Logger
        """
        Get the Logger object for the main execution Task, the current running Task, if one exists. If no Logger object
        exists, this method creates one and returns it. Therefore, you can call this method from anywhere
        in the code.

        .. code-block:: py

           logger = Logger.current_logger()

        :return: The Logger object (a singleton) for the current running Task.
        """
        from .task import Task
        task = Task.current_task()
        if not task:
            return None
        return task.get_logger()

    def report_text(self, msg, level=logging.INFO, print_console=True, *args, **_):
        # type: (str, int, bool, Any, Any) -> None
        """
        For explicit reporting, print text to the log. Optionally, print a log level and print to the console.

        For example:

        .. code-block:: py

           logger.report_text('log some text', level=logging.DEBUG, print_console=False)

        You can view the reported text in the **Trains Web-App (UI)**, **RESULTS** tab, **LOG** sub-tab.

        :param str msg: The text to log.
        :param int level: The log level from the Python ``logging`` package. The default value is ``logging.INFO``.
        :param bool print_console: In addition to the log, print to the console?

            The values are:

            - ``True`` - Print to the console. (Default)
            - ``False`` - Do not print to the console.
        """
        return self._console(msg, level, not print_console, *args, **_)

    def report_scalar(self, title, series, value, iteration):
        # type: (str, str, float, int) -> None
        """
        For explicit reporting, plot a scalar series.

        For example, plot a scalar series:

        .. code-block:: py

           scalar_series = [random.randint(0,10) for i in range(10)]
           logger.report_scalar(title='scalar metrics','series', value=scalar_series[iteration], iteration=0)

        You can view the scalar plots in the **Trains Web-App (UI)**, **RESULTS** tab, **SCALARS** sub-tab.

        :param str title: The title of the plot. Plot more than one scalar series on the same plot by using
            the same ``title`` for each call to this method.
        :param str series: The title of the series.
        :param float value: The value to plot per iteration.
        :param int iteration: The iteration number. Iterations are on the x-axis.
        """

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series)
        return self._task.reporter.report_scalar(title=title, series=series, value=float(value), iter=iteration)

    def report_vector(
            self,
            title,  # type: str
            series,  # type: str
            values,  # type: Sequence[Union[int, float]]
            iteration,  # type: int
            labels=None,  # type: Optional[List[str]]
            xlabels=None,  # type: Optional[List[str]]
            xaxis=None,  # type: Optional[str]
            yaxis=None,  # type: Optional[str]
            mode=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, plot a vector as (default stacked) histogram.

        For example:

        .. code-block:: py

           vector_series = np.random.randint(10, size=10).reshape(2,5)
           logger.report_vector(title='vector example', series='vector series', values=vector_series, iteration=0,
                labels=['A','B'], xaxis='X axis label', yaxis='Y axis label')

        You can view the vectors plots in the **Trains Web-App (UI)**, **RESULTS** tab, **PLOTS** sub-tab.

        :param str title: The title of the plot.
        :param str series: The title of the series.
        :param list(float) values: The series values. A list of floats, or an N-dimensional Numpy array containing
            data for each histogram bar.
        :type values: list(float), numpy.ndarray
        :param int iteration: The iteration number. Each ``iteration`` creates another plot.
        :param list(str) labels: Labels for each bar group, creating a plot legend labeling each series. (Optional)
        :param list(str) xlabels: Labels per entry in each bucket in the histogram (vector), creating a set of labels
            for each histogram bar on the x-axis. (Optional)
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param str mode: Multiple histograms mode, stack / group / relative. Default is 'group'.
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """
        self._touch_title_series(title, series)
        return self.report_histogram(title, series, values, iteration, labels=labels, xlabels=xlabels,
                                     xaxis=xaxis, yaxis=yaxis, mode=mode, extra_layout=extra_layout)

    def report_histogram(
            self,
            title,  # type: str
            series,  # type: str
            values,  # type: Sequence[Union[int, float]]
            iteration,  # type: int
            labels=None,  # type: Optional[List[str]]
            xlabels=None,  # type: Optional[List[str]]
            xaxis=None,  # type: Optional[str]
            yaxis=None,  # type: Optional[str]
            mode=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, plot a (default grouped) histogram.
        Notice this function will not calculate the histogram,
        it assumes the histogram was already calculated in `values`

        For example:

        .. code-block:: py

           vector_series = np.random.randint(10, size=10).reshape(2,5)
           logger.report_histogram(title='histogram example', series='histogram series',
                values=vector_series, iteration=0, labels=['A','B'], xaxis='X axis label', yaxis='Y axis label')

        You can view the reported histograms in the **Trains Web-App (UI)**, **RESULTS** tab, **PLOTS** sub-tab.

        :param str title: The title of the plot.
        :param str series: The title of the series.
        :param list(float) values: The series values. A list of floats, or an N-dimensional Numpy array containing
            data for each histogram bar.
        :type values: list(float), numpy.ndarray
        :param int iteration: The iteration number. Each ``iteration`` creates another plot.
        :param list(str) labels: Labels for each bar group, creating a plot legend labeling each series. (Optional)
        :param list(str) xlabels: Labels per entry in each bucket in the histogram (vector), creating a set of labels
            for each histogram bar on the x-axis. (Optional)
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param str mode: Multiple histograms mode, stack / group / relative. Default is 'group'.
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """

        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series)
        return self._task.reporter.report_histogram(
            title=title,
            series=series,
            histogram=values,
            iter=iteration,
            labels=labels,
            xlabels=xlabels,
            xtitle=xaxis,
            ytitle=yaxis,
            mode=mode or 'group',
            layout_config=extra_layout,
        )

    def report_table(
            self,
            title,  # type: str
            series,  # type: str
            iteration,  # type: int
            table_plot=None,  # type: Optional[pd.DataFrame]
            csv=None,  # type: Optional[str]
            url=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit report, report a table plot.

        One and only one of the following parameters must be provided.

        - ``table_plot`` - Pandas DataFrame
        - ``csv`` - CSV file
        - ``url`` - URL to CSV file

        For example:

        .. code-block:: py

           df = pd.DataFrame({'num_legs': [2, 4, 8, 0],
                   'num_wings': [2, 0, 0, 0],
                   'num_specimen_seen': [10, 2, 1, 8]},
                   index=['falcon', 'dog', 'spider', 'fish'])

           logger.report_table(title='table example',series='pandas DataFrame',iteration=0,table_plot=df)

        You can view the reported tables in the **Trains Web-App (UI)**, **RESULTS** tab, **PLOTS** sub-tab.

        :param str title: The title of the table.
        :param str series: The title of the series.
        :param int iteration: The iteration number.
        :param table_plot: The output table plot object
        :type table_plot: pandas.DataFrame
        :param csv: path to local csv file
        :type csv: str
        :param url: A URL to the location of csv file.
        :type url: str
        :param extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        :type extra_layout: dict
        """
        mutually_exclusive(
            UsageError, _check_none=True,
            table_plot=table_plot, csv=csv, url=url
        )
        table = table_plot
        if url or csv:
            if not pd:
                raise UsageError(
                    "pandas is required in order to support reporting tables using CSV or a URL, "
                    "please install the pandas python package"
                )
            if url:
                table = pd.read_csv(url)
            elif csv:
                table = pd.read_csv(csv)

        def replace(dst, *srcs):
            for src in srcs:
                reporter_table.replace(src, dst, inplace=True)

        reporter_table = table.fillna(str(np.nan))
        replace("NaN", np.nan, math.nan)
        replace("Inf", np.inf, math.inf)
        replace("-Inf", -np.inf, np.NINF, -math.inf)

        return self._task.reporter.report_table(
            title=title,
            series=series,
            table=reporter_table,
            iteration=iteration,
            layout_config=extra_layout,
        )

    def report_line_plot(
            self,
            title,  # type: str
            series,  # type: str
            iteration,  # type: int
            xaxis,  # type: str
            yaxis,  # type: str
            mode='lines',  # type: str
            reverse_xaxis=False,  # type: bool
            comment=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, plot one or more series as lines.

        :param str title: The title of the plot.
        :param list(LineSeriesInfo) series: All the series data, one list element for each line
            in the plot.
        :param int iteration: The iteration number.
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param str mode: The type of line plot.

            The values are:

            - ``lines`` (default)
            - ``markers``
            - ``lines+markers``

        :param bool reverse_xaxis: Reverse the x-axis?

            The values are:

            - ``True`` - The x-axis is high to low  (reversed).
            - ``False`` - The x-axis is low to high  (not reversed). (Default)

        :param str comment: A comment displayed with the plot, underneath the title.
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """

        series = [self.SeriesInfo(**s) if isinstance(s, dict) else s for s in series]

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series[0].name if series else '')
        return self._task.reporter.report_line_plot(
            title=title,
            series=series,
            iter=iteration,
            xtitle=xaxis,
            ytitle=yaxis,
            mode=mode,
            reverse_xaxis=reverse_xaxis,
            comment=comment,
            layout_config=extra_layout,
        )

    def report_scatter2d(
            self,
            title,  # type: str
            series,  # type: str
            scatter,  # type: Union[Sequence[Tuple[float, float]], np.ndarray]
            iteration,  # type: int
            xaxis=None,  # type: Optional[str]
            yaxis=None,  # type: Optional[str]
            labels=None,  # type: Optional[List[str]]
            mode='lines',  # type: str
            comment=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, report a 2d scatter plot.

        For example:

        .. code-block:: py

           scatter2d = np.hstack((np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1))))
           logger.report_scatter2d(title="example_scatter", series="series", iteration=0, scatter=scatter2d,
                xaxis="title x", yaxis="title y")

        Plot multiple 2D scatter series on the same plot by passing the same ``title`` and ``iteration`` values
        to this method:

        .. code-block:: py

           scatter2d_1 = np.hstack((np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1))))
           logger.report_scatter2d(title="example_scatter", series="series_1", iteration=1, scatter=scatter2d_1,
                xaxis="title x", yaxis="title y")

           scatter2d_2 = np.hstack((np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1))))
           logger.report_scatter2d("example_scatter", "series_2", iteration=1, scatter=scatter2d_2,
                xaxis="title x", yaxis="title y")

        :param str title: The title of the plot.
        :param str series: The title of the series.
        :param list scatter: The scatter data. numpy.ndarray or list of (pairs of x,y) scatter:
        :param int iteration: The iteration number. To set an initial iteration, for example to continue a previously
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param list(str) labels: Labels per point in the data assigned to the ``scatter`` parameter. The labels must be
            in the same order as the data.
        :param str mode: The type of scatter plot.

            The values are:

            - ``lines``
            - ``markers``
            - ``lines+markers``

        :param str comment: A comment displayed with the plot, underneath the title.
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """

        if not isinstance(scatter, np.ndarray):
            if not isinstance(scatter, list):
                scatter = list(scatter)
            scatter = np.array(scatter)

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series)
        return self._task.reporter.report_2d_scatter(
            title=title,
            series=series,
            data=scatter.astype(np.float32),
            iter=iteration,
            mode=mode,
            xtitle=xaxis,
            ytitle=yaxis,
            labels=labels,
            comment=comment,
            layout_config=extra_layout,
        )

    def report_scatter3d(
            self,
            title,  # type: str
            series,  # type: str
            scatter,  # type: Union[Sequence[Tuple[float, float, float]], np.ndarray]
            iteration,  # type: int
            xaxis=None,  # type: Optional[str]
            yaxis=None,  # type: Optional[str]
            zaxis=None,  # type: Optional[str]
            labels=None,  # type: Optional[List[str]]
            mode='markers',  # type: str
            fill=False,  # type: bool
            comment=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, plot a 3d scatter graph (with markers).

        :param str title: The title of the plot.
        :param str series: The title of the series.
        :param Union[numpy.ndarray, list] scatter: The scatter data.
            list of (pairs of x,y,z), list of series [[(x1,y1,z1)...]], or numpy.ndarray
        :param int iteration: The iteration number.
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param str zaxis: The z-axis title. (Optional)
        :param list(str) labels: Labels per point in the data assigned to the ``scatter`` parameter. The labels must be
            in the same order as the data.
        :param str mode: The type of scatter plot.

            The values are:

            - ``lines``
            - ``markers``
            - ``lines+markers``

        For example:

        .. code-block:: py

           scatter3d = np.random.randint(10, size=(10, 3))
           logger.report_scatter3d(title="example_scatter_3d", series="series_xyz", iteration=1, scatter=scatter3d,
                xaxis="title x", yaxis="title y", zaxis="title z")

        :param bool fill: Fill the area under the curve?

            The values are:

            - ``True`` - Fill
            - ``False`` - Do not fill (Default)

        :param str comment: A comment displayed with the plot, underneath the title.
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """
        # check if multiple series
        multi_series = (
            isinstance(scatter, list)
            and (
                isinstance(scatter[0], np.ndarray)
                or (
                    scatter[0]
                    and isinstance(scatter[0], list)
                    and isinstance(scatter[0][0], list)
                )
            )
        )

        if not multi_series:
            if not isinstance(scatter, np.ndarray):
                if not isinstance(scatter, list):
                    scatter = list(scatter)
                scatter = np.array(scatter)
            try:
                scatter = scatter.astype(np.float32)
            except ValueError:
                pass

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series)
        return self._task.reporter.report_3d_scatter(
            title=title,
            series=series,
            data=scatter,
            iter=iteration,
            labels=labels,
            mode=mode,
            fill=fill,
            comment=comment,
            xtitle=xaxis,
            ytitle=yaxis,
            ztitle=zaxis,
            layout_config=extra_layout,
        )

    def report_confusion_matrix(
            self,
            title,  # type: str
            series,  # type: str
            matrix,  # type: np.ndarray
            iteration,  # type: int
            xaxis=None,  # type: Optional[str]
            yaxis=None,  # type: Optional[str]
            xlabels=None,  # type: Optional[List[str]]
            ylabels=None,  # type: Optional[List[str]]
            comment=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, plot a heat-map matrix.

        For example:

        .. code-block:: py

           confusion = np.random.randint(10, size=(10, 10))
           logger.report_confusion_matrix("example confusion matrix", "ignored", iteration=1, matrix=confusion,
                xaxis="title X", yaxis="title Y")

        :param str title: The title of the plot.
        :param str series: The title of the series.
        :param numpy.ndarray matrix: A heat-map matrix (example: confusion matrix)
        :param int iteration: The iteration number.
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param list(str) xlabels: Labels for each column of the matrix. (Optional)
        :param list(str) ylabels: Labels for each row of the matrix. (Optional)
        :param str comment: A comment displayed with the plot, underneath the title.
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series)
        return self._task.reporter.report_value_matrix(
            title=title,
            series=series,
            data=matrix.astype(np.float32),
            iter=iteration,
            xtitle=xaxis,
            ytitle=yaxis,
            xlabels=xlabels,
            ylabels=ylabels,
            comment=comment,
            layout_config=extra_layout,
        )

    def report_matrix(
            self,
            title,  # type: str
            series,  # type: str
            matrix,  # type: np.ndarray
            iteration,  # type: int
            xaxis=None,  # type: Optional[str]
            yaxis=None,  # type: Optional[str]
            xlabels=None,  # type: Optional[List[str]]
            ylabels=None,  # type: Optional[List[str]]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, plot a confusion matrix.

        .. note::
            This method is the same as :meth:`Logger.report_confusion_matrix`.

        :param str title: The title of the plot.
        :param str series: The title of the series.
        :param numpy.ndarray matrix: A heat-map matrix (example: confusion matrix)
        :param int iteration: The iteration number.
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param list(str) xlabels: Labels for each column of the matrix. (Optional)
        :param list(str) ylabels: Labels for each row of the matrix. (Optional)
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """
        self._touch_title_series(title, series)
        return self.report_confusion_matrix(title, series, matrix, iteration,
                                            xaxis=xaxis, yaxis=yaxis, xlabels=xlabels, ylabels=ylabels,
                                            extra_layout=extra_layout)

    def report_surface(
            self,
            title,  # type: str
            series,  # type: str
            matrix,  # type: np.ndarray
            iteration,  # type: int
            xaxis=None,  # type: Optional[str]
            yaxis=None,  # type: Optional[str]
            zaxis=None,  # type: Optional[str]
            xlabels=None,  # type: Optional[List[str]]
            ylabels=None,  # type: Optional[List[str]]
            camera=None,  # type: Optional[Sequence[float]]
            comment=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, report a 3d surface plot.

        .. note::
           This method plots the same data as :meth:`Logger.report_confusion_matrix`, but presents the
           data as a surface diagram not a confusion matrix.

        .. code-block:: py

           surface_matrix = np.random.randint(10, size=(10, 10))
           logger.report_surface("example surface", "series", iteration=0, matrix=surface_matrix,
                xaxis="title X", yaxis="title Y", zaxis="title Z")

        :param str title: The title of the plot.
        :param str series: The title of the series.
        :param numpy.ndarray matrix: A heat-map matrix (example: confusion matrix)
        :param int iteration: The iteration number.
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param str zaxis: The z-axis title. (Optional)
        :param list(str) xlabels: Labels for each column of the matrix. (Optional)
        :param list(str) ylabels: Labels for each row of the matrix. (Optional)
        :param list(float) camera: X,Y,Z coordinates indicating the camera position. The default value is ``(1,1,1)``.
        :param str comment: A comment displayed with the plot, underneath the title.
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series)
        return self._task.reporter.report_value_surface(
            title=title,
            series=series,
            data=matrix.astype(np.float32),
            iter=iteration,
            xlabels=xlabels,
            ylabels=ylabels,
            xtitle=xaxis,
            ytitle=yaxis,
            ztitle=zaxis,
            camera=camera,
            comment=comment,
            layout_config=extra_layout,
        )

    def report_image(
            self,
            title,  # type: str
            series,  # type: str
            iteration,  # type: int
            local_path=None,  # type: Optional[str]
            image=None,  # type: Optional[Union[np.ndarray, Image.Image]]
            matrix=None,  # type: Optional[np.ndarray]
            max_image_history=None,  # type: Optional[int]
            delete_after_upload=False,  # type: bool
            url=None  # type: Optional[str]
    ):
        """
        For explicit reporting, report an image and upload its contents.

        This method uploads the image to a preconfigured bucket (see :meth:`Logger.setup_upload`) with a key (filename)
        describing the task ID, title, series and iteration.

        For example:

        .. code-block:: py

           matrix = np.eye(256, 256, dtype=np.uint8)*255
           matrix = np.concatenate((np.atleast_3d(matrix), np.zeros((256, 256, 2), dtype=np.uint8)), axis=2)
           logger.report_image("test case", "image color red", iteration=1, image=m)

           image_open = Image.open(os.path.join("<image_path>", "<image_filename>"))
           logger.report_image("test case", "image PIL", iteration=1, image=image_open)

        One and only one of the following parameters must be provided.

        - :paramref:`~.Logger.report_image.local_path`
        - :paramref:`~.Logger.report_image.url`
        - :paramref:`~.Logger.report_image.image`
        - :paramref:`~.Logger.report_image.matrix`

        :param str title: The title of the image.
        :param str series: The title of the series of this image.
        :param int iteration: The iteration number.
        :param str local_path: A path to an image file.
        :param str url: A URL for the location of a pre-uploaded image.
        :param image: Image data (RGB).
        :type image: numpy.ndarray, PIL.Image.Image
        :param numpy.ndarray matrix: Image data (RGB).

            .. note::
               The ``matrix`` paramater is deprecated. Use the ``image`` parameters.
        :type matrix: 3D numpy.ndarray
        :param int max_image_history: The maximum number of images to store per metric/variant combination.
            For an unlimited number, use a negative value. The default value is set in global configuration
            (default=``5``).
        :param bool delete_after_upload: After the upload, delete the local copy of the image?

            The values are:

            - ``True`` - Delete after upload.
            - ``False`` - Do not delete after upload. (Default)
        """
        mutually_exclusive(
            UsageError, _check_none=True,
            local_path=local_path or None, url=url or None, image=image, matrix=matrix
        )
        if matrix is not None:
            warnings.warn("'matrix' variable is deprecated; use 'image' instead.", DeprecationWarning)
        if image is None:
            image = matrix
        if image is not None and not isinstance(image, (np.ndarray, Image.Image)):
            raise ValueError("Supported 'image' types are: numpy.ndarray or PIL.Image")

        # if task was not started, we have to start it
        self._start_task_if_needed()

        self._touch_title_series(title, series)

        if url:
            self._task.reporter.report_image(
                title=title,
                series=series,
                src=url,
                iter=iteration,
            )

        else:
            upload_uri = self.get_default_upload_destination()
            if not upload_uri:
                upload_uri = Path(get_cache_dir()) / 'debug_images'
                upload_uri.mkdir(parents=True, exist_ok=True)
                # Verify that we can upload to this destination
                upload_uri = str(upload_uri)
                storage = StorageHelper.get(upload_uri)
                upload_uri = storage.verify_upload(folder_uri=upload_uri)

            if isinstance(image, Image.Image):
                image = np.array(image)

            self._task.reporter.report_image_and_upload(
                title=title,
                series=series,
                path=local_path,
                image=image,
                iter=iteration,
                upload_uri=upload_uri,
                max_image_history=max_image_history,
                delete_after_upload=delete_after_upload,
            )

    def report_media(
            self,
            title,  # type: str
            series,  # type: str
            iteration,  # type: int
            local_path=None,  # type: Optional[str]
            stream=None,  # type: Optional[six.BytesIO]
            file_extension=None,  # type: Optional[str]
            max_history=None,  # type: Optional[int]
            delete_after_upload=False,  # type: bool
            url=None  # type: Optional[str]
    ):
        """
        Report an image and upload its contents.

        Image is uploaded to a preconfigured bucket (see setup_upload()) with a key (filename)
        describing the task ID, title, series and iteration.

        .. note::
            :paramref:`~.Logger.report_image.local_path`, :paramref:`~.Logger.report_image.url`,
            :paramref:`~.Logger.report_image.image` and :paramref:`~.Logger.report_image.matrix`
            are mutually exclusive, and at least one must be provided.

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param int iteration: Iteration number
        :param str local_path: A path to an image file.
        :param stream: BytesIO stream to upload (must provide file extension if used)
        :param str url: A URL to the location of a pre-uploaded image.
        :param file_extension: file extension to use when stream is passed
        :param int max_history: maximum number of media files to store per metric/variant combination
            use negative value for unlimited. default is set in global configuration (default=5)
        :param bool delete_after_upload: if True, one the file was uploaded the local copy will be deleted
        """
        mutually_exclusive(
            UsageError, _check_none=True,
            local_path=local_path or None, url=url or None, stream=stream,
        )
        if stream is not None and not file_extension:
            raise ValueError("No file extension provided for stream media upload")

        # if task was not started, we have to start it
        self._start_task_if_needed()

        self._touch_title_series(title, series)

        if url:
            self._task.reporter.report_media(
                title=title,
                series=series,
                src=url,
                iter=iteration,
            )

        else:
            upload_uri = self.get_default_upload_destination()
            if not upload_uri:
                upload_uri = Path(get_cache_dir()) / 'debug_images'
                upload_uri.mkdir(parents=True, exist_ok=True)
                # Verify that we can upload to this destination
                upload_uri = str(upload_uri)
                storage = StorageHelper.get(upload_uri)
                upload_uri = storage.verify_upload(folder_uri=upload_uri)

            self._task.reporter.report_media_and_upload(
                title=title,
                series=series,
                path=local_path,
                stream=stream,
                iter=iteration,
                upload_uri=upload_uri,
                max_history=max_history,
                delete_after_upload=delete_after_upload,
                file_extension=file_extension,
            )

    def set_default_upload_destination(self, uri):
        # type: (str) -> None
        """
        Set the destination storage URI (for example, S3, Google Cloud Storage, a file path) for uploading debug images.

        The images are uploaded separately. A link to each image is reported.

        .. note::
           Credentials for the destination storage are specified in the  Trains configuration file,
           ``~/trains.conf``.

        :param str uri: example: 's3://bucket/directory/' or 'file:///tmp/debug/'

        :return: bool

            The values are:

            - ``True`` - The destination scheme is supported (for example, ``s3://``, ``file://``, or ``gc://``).
            - ``False`` - The destination scheme is not supported.
        """

        # Create the storage helper
        storage = StorageHelper.get(uri)

        # Verify that we can upload to this destination
        uri = storage.verify_upload(folder_uri=uri)

        self._default_upload_destination = uri

    def get_default_upload_destination(self):
        # type: () -> str
        """
        Get the destination storage URI (for example, S3, Google Cloud Storage, a file path) for uploading debug images
        (see :meth:`Logger.set_default_upload_destination`).

        :return: The default upload destination URI.

            For example, ``s3://bucket/directory/`` or ``file:///tmp/debug/``.
        """
        return self._default_upload_destination or self._task._get_default_report_storage_uri()

    def flush(self):
        # type: () -> bool
        """
        Flush cached reports and console outputs to backend.

        :return: bool

            The values are:

            - ``True`` - Successfully flushed the cache.
            - ``False`` - Failed.
        """
        self._flush_stdout_handler()
        if self._task:
            return self._task.flush()
        return False

    def get_flush_period(self):
        # type: () -> Optional[float]
        """
        Get the Logger flush period.

        :return: The logger flush period in seconds.
        """
        if self._flusher:
            return self._flusher.period
        return None

    def set_flush_period(self, period):
        # type: (float) -> None
        """
        Set the logger flush period.

        :param float period: The period to flush the logger in seconds. To set no periodic flush,
            specify ``None`` or ``0``.
        """
        if self._task.is_main_task() and DevWorker.report_stdout and DevWorker.report_period and \
                not running_remotely() and period is not None:
            period = min(period or DevWorker.report_period, DevWorker.report_period)

        if not period:
            if self._flusher:
                self._flusher.exit()
                self._flusher = None
        elif self._flusher:
            self._flusher.set_period(period)
        else:
            self._flusher = LogFlusher(self, period)
            self._flusher.start()

    def report_image_and_upload(
            self,
            title,  # type: str
            series,  # type: str
            iteration,  # type: int
            path=None,  # type: Optional[str]
            matrix=None,  # type: # type: Optional[Union[np.ndarray, Image.Image]]
            max_image_history=None,  # type: Optional[int]
            delete_after_upload=False  # type: bool
    ):
        """
        .. deprecated:: 0.13.0
            Use :meth:`Logger.report_image` instead
        """
        self.report_image(title=title, series=series, iteration=iteration, local_path=path, image=matrix,
                          max_image_history=max_image_history, delete_after_upload=delete_after_upload)

    @classmethod
    def tensorboard_auto_group_scalars(cls, group_scalars=False):
        # type: (bool) -> None
        """
        Group together TensorBoard scalars that do not have a title, or assign a title/series with the same tag.

        :param group_scalars: Group TensorBoard scalars without a title?

            The values are:

            - ``True`` - Scalars without specific titles are grouped together in the "Scalars" plot, preserving
              backward compatibility with Trains automagical behavior.
            - ``False`` - TensorBoard scalars without titles get a title/series with the same tag. (Default)
        :type group_scalars: bool
        """
        cls._tensorboard_logging_auto_group_scalars = group_scalars

    @classmethod
    def tensorboard_single_series_per_graph(cls, single_series=False):
        # type: (bool) -> None
        """
        Group TensorBoard scalar series together or in separate plots.

        :param single_series: Group TensorBoard scalar series together?

            The values are:

            - ``True`` - Generate a separate plot for each TensorBoard scalar series.
            - ``False`` - Group the TensorBoard scalar series together in the same plot. (Default)

        :type single_series: bool
        """
        cls._tensorboard_logging_single_series_per_graphs = single_series

    @classmethod
    def _remove_std_logger(cls):
        StdStreamPatch.remove_std_logger()

    def _console(self, msg, level=logging.INFO, omit_console=False, *args, **kwargs):
        # type: (str, int, bool, Any, Any) -> None
        """
        print text to log (same as print to console, and also prints to console)

        :param str msg: text to print to the console (always send to the backend and displayed in console)
        :param level: logging level, default: logging.INFO
        :type level: Logging Level
        :param bool omit_console: Omit the console output, and only send the ``msg`` value to the log?

            - ``True`` - Omit the console output.
            - ``False`` - Print the console output. (Default)

        """
        try:
            level = int(level)
        except (TypeError, ValueError):
            self._task.log.log(level=logging.ERROR,
                               msg='Logger failed casting log level "%s" to integer' % str(level))
            level = logging.INFO

        if not running_remotely():
            # noinspection PyBroadException
            try:
                record = self._task.log.makeRecord(
                    "console", level=level, fn='', lno=0, func='', msg=msg, args=args, exc_info=None
                )
                # find the task handler that matches our task
                if not self._task_handler:
                    self._task_handler = [h for h in LoggerRoot.get_base_logger().handlers
                                          if isinstance(h, TaskHandler) and h.task_id == self._task.id][0]
                self._task_handler.emit(record)
            except Exception:
                # avoid infinite loop, output directly to stderr
                try:
                    # make sure we are writing to the original stdout
                    StdStreamPatch.stderr_original_write(
                        'trains.Logger failed sending log [level {}]: "{}"\n'.format(level, msg))
                except Exception:
                    pass

        if not omit_console:
            # if we are here and we grabbed the stdout, we need to print the real thing
            if DevWorker.report_stdout and not running_remotely():
                # noinspection PyBroadException
                try:
                    # make sure we are writing to the original stdout
                    StdStreamPatch.stdout_original_write(str(msg) + '\n')
                except Exception:
                    pass
            else:
                print(str(msg))

        # if task was not started, we have to start it
        self._start_task_if_needed()

    def _report_image_plot_and_upload(
            self,
            title,  # type: str
            series,  # type: str
            iteration,  # type: int
            path=None,  # type: Optional[str]
            matrix=None,  # type: Optional[np.ndarray]
            max_image_history=None,  # type: Optional[int]
            delete_after_upload=False  # type: bool
    ):
        """
        Report an image, upload its contents, and present in plots section using plotly

        Image is uploaded to a preconfigured bucket (see :meth:`Logger.setup_upload`) with a key (filename)
        describing the task ID, title, series and iteration.

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param iteration: Iteration number
        :type iteration: int
        :param path: A path to an image file. Required unless matrix is provided.
        :type path: str
        :param matrix: A 3D numpy.ndarray object containing image data (RGB). Required unless filename is provided.
        :type matrix: str
        :param max_image_history: maximum number of image to store per metric/variant combination \
        use negative value for unlimited. default is set in global configuration (default=5)
        :type max_image_history: int
        :param delete_after_upload: if True, one the file was uploaded the local copy will be deleted
        :type delete_after_upload: boolean
        """

        # if task was not started, we have to start it
        self._start_task_if_needed()
        upload_uri = self.get_default_upload_destination()
        if not upload_uri:
            upload_uri = Path(get_cache_dir()) / 'debug_images'
            upload_uri.mkdir(parents=True, exist_ok=True)
            # Verify that we can upload to this destination
            upload_uri = str(upload_uri)
            storage = StorageHelper.get(upload_uri)
            upload_uri = storage.verify_upload(folder_uri=upload_uri)

        self._task.reporter.report_image_plot_and_upload(
            title=title,
            series=series,
            path=path,
            matrix=matrix,
            iter=iteration,
            upload_uri=upload_uri,
            max_image_history=max_image_history,
            delete_after_upload=delete_after_upload,
        )

    def _report_file_and_upload(
            self,
            title,  # type: str
            series,  # type: str
            iteration,  # type: int
            path=None,  # type: Optional[str]
            max_file_history=None,  # type: Optional[int]
            delete_after_upload=False  # type: bool
    ):
        """
        Upload a file and report it as link in the debug images section.

        File is uploaded to a preconfigured storage (see :meth:`Logger.setup_upload`) with a key (filename)
        describing the task ID, title, series and iteration.

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param iteration: Iteration number
        :type iteration: int
        :param path: A path to file to be uploaded
        :type path: str
        :param max_file_history: maximum number of files to store per metric/variant combination \
        use negative value for unlimited. default is set in global configuration (default=5)
        :type max_file_history: int
        :param delete_after_upload: if True, one the file was uploaded the local copy will be deleted
        :type delete_after_upload: boolean
        """

        # if task was not started, we have to start it
        self._start_task_if_needed()
        upload_uri = self.get_default_upload_destination()
        if not upload_uri:
            upload_uri = Path(get_cache_dir()) / 'debug_images'
            upload_uri.mkdir(parents=True, exist_ok=True)
            # Verify that we can upload to this destination
            upload_uri = str(upload_uri)
            storage = StorageHelper.get(upload_uri)
            upload_uri = storage.verify_upload(folder_uri=upload_uri)

        self._task.reporter.report_image_and_upload(
            title=title,
            series=series,
            path=path,
            image=None,
            iter=iteration,
            upload_uri=upload_uri,
            max_image_history=max_file_history,
            delete_after_upload=delete_after_upload,
        )

    def _start_task_if_needed(self):
        # deprecated
        pass

    def _flush_stdout_handler(self):
        if self._task_handler and DevWorker.report_stdout:
            self._task_handler.flush()

    def _close_stdout_handler(self, wait=True):
        # detach the sys stdout/stderr
        StdStreamPatch.remove_std_logger(self)

        if self._task_handler and DevWorker.report_stdout:
            t = self._task_handler
            self._task_handler = None
            t.close(wait)

    def _touch_title_series(self, title, series):
        # type: (str, str) -> None
        if title not in self._graph_titles:
            self._graph_titles[title] = set()
        self._graph_titles[title].add(series)

    def _get_used_title_series(self):
        return self._graph_titles

    @classmethod
    def _get_tensorboard_auto_group_scalars(cls):
        """
        :return: return True if we preserve Tensorboard backward compatibility behaviour,
            i.e. Scalars without specific title will be under the "Scalars" graph
            default is False: Tensorboard scalars without title will have title/series with the same tag
        """
        return cls._tensorboard_logging_auto_group_scalars

    @classmethod
    def _get_tensorboard_single_series_per_graph(cls):
        """
        :return: return True if we generate a separate graph (plot) for each Tensorboard scalar series
            default is False: Tensorboard scalar series will be grouped according to their title
        """
        return cls._tensorboard_single_series_per_graph
