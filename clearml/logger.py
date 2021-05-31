import logging
import math
import warnings
from typing import Any, Sequence, Union, List, Optional, Tuple, Dict, TYPE_CHECKING

import numpy as np
import six
from PIL import Image
from pathlib2 import Path

from .debugging.log import LoggerRoot

try:
    import pandas as pd
except ImportError:
    pd = None

from .backend_interface.logger import StdStreamPatch
from .backend_interface.task import Task as _Task
from .backend_interface.task.log import TaskHandler
from .backend_interface.util import mutually_exclusive
from .config import running_remotely, get_cache_dir, config, DEBUG_SIMULATE_REMOTE_TASK
from .errors import UsageError
from .storage.helper import StorageHelper
from .utilities.plotly_reporter import SeriesInfo

# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning, module=__name__)


if TYPE_CHECKING:
    from matplotlib.figure import Figure as MatplotlibFigure  # noqa
    from matplotlib import pyplot  # noqa


class Logger(object):
    """
    The ``Logger`` class is the ClearML console log and metric statistics interface, and contains methods for explicit
    reporting.

    Explicit reporting extends ClearML automagical capturing of inputs and output. Explicit reporting
    methods include scalar plots, line plots, histograms, confusion matrices, 2D and 3D scatter
    diagrams, text logging, tables, and image uploading and reporting.

    In the **ClearML Web-App (UI)**, ``Logger`` output appears in the **RESULTS** tab, **CONSOLE**, **SCALARS**,
    **PLOTS**, and **DEBUG SAMPLES** sub-tabs. When you compare experiments, ``Logger`` output appears in the
    comparisons.

    .. warning::

       Do not construct Logger objects directly.

    You must get a Logger object before calling any of the other ``Logger`` class methods by calling
    :meth:`.Task.get_logger` or :meth:`Logger.current_logger`.


    """
    SeriesInfo = SeriesInfo
    _tensorboard_logging_auto_group_scalars = False
    _tensorboard_single_series_per_graph = config.get('metrics.tensorboard_single_series_per_graph', False)

    def __init__(self, private_task, connect_stdout=True, connect_stderr=True, connect_logging=False):
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
        self._graph_titles = {}
        self._tensorboard_series_force_prefix = None
        self._task_handler = TaskHandler(task=self._task, capacity=100) \
            if private_task.is_main_task() or (connect_stdout or connect_stderr or connect_logging) else None
        self._connect_std_streams = connect_stdout or connect_stderr
        self._connect_logging = connect_logging

        # Make sure urllib is never in debug/info,
        disable_urllib3_info = config.get('log.disable_urllib3_info', True)
        if disable_urllib3_info and logging.getLogger('urllib3').isEnabledFor(logging.INFO):
            logging.getLogger('urllib3').setLevel(logging.WARNING)

        StdStreamPatch.patch_std_streams(self, connect_stdout=connect_stdout, connect_stderr=connect_stderr)

        if self._connect_logging:
            StdStreamPatch.patch_logging_formatter(self)
        elif not self._connect_std_streams:
            # make sure that at least the main clearml logger is connect
            base_logger = LoggerRoot.get_base_logger()
            if base_logger and base_logger.handlers:
                StdStreamPatch.patch_logging_formatter(self, base_logger.handlers[0])

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

        You can view the reported text in the **ClearML Web-App (UI)**, **RESULTS** tab, **CONSOLE** sub-tab.

        :param str msg: The text to log.
        :param int level: The log level from the Python ``logging`` package. The default value is ``logging.INFO``.
        :param bool print_console: In addition to the log, print to the console

            The values are:

            - ``True`` - Print to the console. (default)
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

        You can view the scalar plots in the **ClearML Web-App (UI)**, **RESULTS** tab, **SCALARS** sub-tab.

        :param str title: The title (metric) of the plot. Plot more than one scalar series on the same plot by using
            the same ``title`` for each call to this method.
        :param str series: The series name (variant) of the reported scalar.
        :param float value: The value to plot per iteration.
        :param int iteration: The iteration number. Iterations are on the x-axis.
        """

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series)
        # noinspection PyProtectedMember
        return self._task._reporter.report_scalar(title=title, series=series, value=float(value), iter=iteration)

    def report_vector(
            self,
            title,  # type: str
            series,  # type: str
            values,  # type: Sequence[Union[int, float]]
            iteration=None,  # type: Optional[int]
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

        You can view the vectors plots in the **ClearML Web-App (UI)**, **RESULTS** tab, **PLOTS** sub-tab.

        :param str title: The title (metric) of the plot.
        :param str series: The series name (variant) of the reported histogram.
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
        return self.report_histogram(title, series, values, iteration or 0, labels=labels, xlabels=xlabels,
                                     xaxis=xaxis, yaxis=yaxis, mode=mode, extra_layout=extra_layout)

    def report_histogram(
            self,
            title,  # type: str
            series,  # type: str
            values,  # type: Sequence[Union[int, float]]
            iteration=None,  # type: Optional[int]
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

        You can view the reported histograms in the **ClearML Web-App (UI)**, **RESULTS** tab, **PLOTS** sub-tab.

        :param str title: The title (metric) of the plot.
        :param str series: The series name (variant) of the reported histogram.
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
        # noinspection PyProtectedMember
        return self._task._reporter.report_histogram(
            title=title,
            series=series,
            histogram=values,
            iter=iteration or 0,
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
            iteration=None,  # type: Optional[int]
            table_plot=None,  # type: Optional[pd.DataFrame, Sequence[Sequence]]
            csv=None,  # type: Optional[str]
            url=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit report, report a table plot.

        One and only one of the following parameters must be provided.

        - ``table_plot`` - Pandas DataFrame or Table as list of rows (list)
        - ``csv`` - CSV file
        - ``url`` - URL to CSV file

        For example:

        .. code-block:: py

           df = pd.DataFrame({'num_legs': [2, 4, 8, 0],
                   'num_wings': [2, 0, 0, 0],
                   'num_specimen_seen': [10, 2, 1, 8]},
                   index=['falcon', 'dog', 'spider', 'fish'])

           logger.report_table(title='table example',series='pandas DataFrame',iteration=0,table_plot=df)

        You can view the reported tables in the **ClearML Web-App (UI)**, **RESULTS** tab, **PLOTS** sub-tab.

        :param str title: The title (metric) of the table.
        :param str series: The series name (variant) of the reported table.
        :param int iteration: The iteration number.
        :param table_plot: The output table plot object
        :type table_plot: pandas.DataFrame or Table as list of rows (list)
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
                table = pd.read_csv(url, index_col=[0])
            elif csv:
                table = pd.read_csv(csv, index_col=[0])

        def replace(dst, *srcs):
            for src in srcs:
                reporter_table.replace(src, dst, inplace=True)

        if isinstance(table, (list, tuple)):
            reporter_table = table
        else:
            reporter_table = table.fillna(str(np.nan))
            replace("NaN", np.nan, math.nan if six.PY3 else float("nan"))
            replace("Inf", np.inf, math.inf if six.PY3 else float("inf"))
            replace("-Inf", -np.inf, np.NINF, -math.inf if six.PY3 else -float("inf"))
        # noinspection PyProtectedMember
        return self._task._reporter.report_table(
            title=title,
            series=series,
            table=reporter_table,
            iteration=iteration or 0,
            layout_config=extra_layout,
        )

    def report_line_plot(
            self,
            title,  # type: str
            series,  # type: Sequence[SeriesInfo]
            xaxis,  # type: str
            yaxis,  # type: str
            mode='lines',  # type: str
            iteration=None,  # type: Optional[int]
            reverse_xaxis=False,  # type: bool
            comment=None,  # type: Optional[str]
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, plot one or more series as lines.

        :param str title: The title (metric) of the plot.
        :param list series: All the series data, one list element for each line in the plot.
        :param int iteration: The iteration number.
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param str mode: The type of line plot.

            The values are:

            - ``lines`` (default)
            - ``markers``
            - ``lines+markers``

        :param bool reverse_xaxis: Reverse the x-axis

            The values are:

            - ``True`` - The x-axis is high to low  (reversed).
            - ``False`` - The x-axis is low to high  (not reversed). (default)

        :param str comment: A comment displayed with the plot, underneath the title.
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """

        # noinspection PyArgumentList
        series = [self.SeriesInfo(**s) if isinstance(s, dict) else s for s in series]

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series[0].name if series else '')
        # noinspection PyProtectedMember
        return self._task._reporter.report_line_plot(
            title=title,
            series=series,
            iter=iteration or 0,
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
            iteration=None,  # type: Optional[int]
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

        :param str title: The title (metric) of the plot.
        :param str series: The series name (variant) of the reported scatter plot.
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
        # noinspection PyProtectedMember
        return self._task._reporter.report_2d_scatter(
            title=title,
            series=series,
            data=scatter,
            iter=iteration or 0,
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
            iteration=None,  # type: Optional[int]
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

        :param str title: The title (metric) of the plot.
        :param str series: The series name (variant) of the reported scatter plot.
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

        :param bool fill: Fill the area under the curve

            The values are:

            - ``True`` - Fill
            - ``False`` - Do not fill (default)

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
        # noinspection PyProtectedMember
        return self._task._reporter.report_3d_scatter(
            title=title,
            series=series,
            data=scatter,
            iter=iteration or 0,
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
            iteration=None,  # type: Optional[int]
            xaxis=None,  # type: Optional[str]
            yaxis=None,  # type: Optional[str]
            xlabels=None,  # type: Optional[List[str]]
            ylabels=None,  # type: Optional[List[str]]
            yaxis_reversed=False,  # type: bool
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

        :param str title: The title (metric) of the plot.
        :param str series: The series name (variant) of the reported confusion matrix.
        :param numpy.ndarray matrix: A heat-map matrix (example: confusion matrix)
        :param int iteration: The iteration number.
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param list(str) xlabels: Labels for each column of the matrix. (Optional)
        :param list(str) ylabels: Labels for each row of the matrix. (Optional)
        :param bool yaxis_reversed: If False 0,0 is at the bottom left corner. If True 0,0 is at the Top left corner
        :param str comment: A comment displayed with the plot, underneath the title.
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series)
        # noinspection PyProtectedMember
        return self._task._reporter.report_value_matrix(
            title=title,
            series=series,
            data=matrix.astype(np.float32),
            iter=iteration or 0,
            xtitle=xaxis,
            ytitle=yaxis,
            xlabels=xlabels,
            ylabels=ylabels,
            yaxis_reversed=yaxis_reversed,
            comment=comment,
            layout_config=extra_layout,
        )

    def report_matrix(
            self,
            title,  # type: str
            series,  # type: str
            matrix,  # type: np.ndarray
            iteration=None,  # type: Optional[int]
            xaxis=None,  # type: Optional[str]
            yaxis=None,  # type: Optional[str]
            xlabels=None,  # type: Optional[List[str]]
            ylabels=None,  # type: Optional[List[str]]
            yaxis_reversed=False,  # type: bool
            extra_layout=None,  # type: Optional[dict]
    ):
        """
        For explicit reporting, plot a confusion matrix.

        .. note::
            This method is the same as :meth:`Logger.report_confusion_matrix`.

        :param str title: The title (metric) of the plot.
        :param str series: The series name (variant) of the reported confusion matrix.
        :param numpy.ndarray matrix: A heat-map matrix (example: confusion matrix)
        :param int iteration: The iteration number.
        :param str xaxis: The x-axis title. (Optional)
        :param str yaxis: The y-axis title. (Optional)
        :param list(str) xlabels: Labels for each column of the matrix. (Optional)
        :param list(str) ylabels: Labels for each row of the matrix. (Optional)
        :param bool yaxis_reversed: If False 0,0 is at the bottom left corner. If True 0,0 is at the Top left corner
        :param dict extra_layout: optional dictionary for layout configuration, passed directly to plotly
            example: extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}
        """
        self._touch_title_series(title, series)
        return self.report_confusion_matrix(title, series, matrix, iteration or 0,
                                            xaxis=xaxis, yaxis=yaxis, xlabels=xlabels, ylabels=ylabels,
                                            yaxis_reversed=yaxis_reversed,
                                            extra_layout=extra_layout)

    def report_surface(
            self,
            title,  # type: str
            series,  # type: str
            matrix,  # type: np.ndarray
            iteration=None,  # type: Optional[int]
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

        :param str title: The title (metric) of the plot.
        :param str series: The series name (variant) of the reported surface.
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
        # noinspection PyProtectedMember
        return self._task._reporter.report_value_surface(
            title=title,
            series=series,
            data=matrix.astype(np.float32),
            iter=iteration or 0,
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
            iteration=None,  # type: Optional[int]
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

        - ``local_path``
        - ``url``
        - ``image``
        - ``matrix``

        :param str title: The title (metric) of the image.
        :param str series: The series name (variant) of the reported image.
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
        :param bool delete_after_upload: After the upload, delete the local copy of the image

            The values are:

            - ``True`` - Delete after upload.
            - ``False`` - Do not delete after upload. (default)
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
            # noinspection PyProtectedMember
            self._task._reporter.report_image(
                title=title,
                series=series,
                src=url,
                iter=iteration or 0,
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
                image = np.array(image)  # noqa
            # noinspection PyProtectedMember
            self._task._reporter.report_image_and_upload(
                title=title,
                series=series,
                path=local_path,
                image=image,
                iter=iteration or 0,
                upload_uri=upload_uri,
                max_image_history=max_image_history,
                delete_after_upload=delete_after_upload,
            )

    def report_media(
            self,
            title,  # type: str
            series,  # type: str
            iteration=None,  # type: Optional[int]
            local_path=None,  # type: Optional[str]
            stream=None,  # type: Optional[Union[six.BytesIO, six.StringIO]]
            file_extension=None,  # type: Optional[str]
            max_history=None,  # type: Optional[int]
            delete_after_upload=False,  # type: bool
            url=None  # type: Optional[str]
    ):
        """
        Report media upload its contents, including images, audio, and video.

        Media is uploaded to a preconfigured bucket (see setup_upload()) with a key (filename)
        describing the task ID, title, series and iteration.

        One and only one of the following parameters must be provided

        - ``local_path``
        - ``stream``
        - ``url``

        If you use ``stream`` for a BytesIO stream to upload, ``file_extension`` must be provided.

        :param str title: The title (metric) of the media.
        :param str series: The series name (variant) of the reported media.
        :param int iteration: The iteration number.
        :param str local_path: A path to an media file.
        :param stream: BytesIO stream to upload. If provided, ``file_extension`` must also be provided.
        :param str url: A URL to the location of a pre-uploaded media.
        :param file_extension: A file extension to use when ``stream`` is passed.
        :param int max_history: The maximum number of media files to store per metric/variant combination
            use negative value for unlimited. default is set in global configuration (default=5)
        :param bool delete_after_upload: After the file is uploaded, delete the local copy

            - ``True`` - Delete
            - ``False`` - Do not delete

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
            # noinspection PyProtectedMember
            self._task._reporter.report_media(
                title=title,
                series=series,
                src=url,
                iter=iteration or 0,
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
            # noinspection PyProtectedMember
            self._task._reporter.report_media_and_upload(
                title=title,
                series=series,
                path=local_path,
                stream=stream,
                iter=iteration or 0,
                upload_uri=upload_uri,
                max_history=max_history,
                delete_after_upload=delete_after_upload,
                file_extension=file_extension,
            )

    def report_plotly(
            self,
            title,  # type: str
            series,  # type: str
            figure,  # type: Union[Dict, "Figure"]  # noqa: F821
            iteration=None,  # type: Optional[int]
    ):
        """
        Report a ``Plotly`` figure (plot) directly

        ``Plotly`` figure can be a ``plotly.graph_objs._figure.Figure`` or a dictionary as defined by ``plotly.js``

        :param str title: The title (metric) of the plot.
        :param str series: The series name (variant) of the reported plot.
        :param int iteration: The iteration number.
        :param dict figure: A ``plotly`` Figure object or a ``poltly`` dictionary
        """
        # if task was not started, we have to start it
        self._start_task_if_needed()

        self._touch_title_series(title, series)

        plot = figure if isinstance(figure, dict) else figure.to_plotly_json()
        # noinspection PyBroadException
        try:
            plot['layout']['title'] = series
        except Exception:
            pass
        # noinspection PyProtectedMember
        self._task._reporter.report_plot(
            title=title,
            series=series,
            plot=plot,
            iter=iteration or 0,
        )

    def report_matplotlib_figure(
            self,
            title,  # type: str
            series,  # type: str
            figure,  # type: Union[MatplotlibFigure, pyplot]
            iteration=None,  # type: Optional[int]
            report_image=False,  # type: bool
            report_interactive=True,  # type: bool
    ):
        """
        Report a ``matplotlib`` figure / plot directly

        ``matplotlib.figure.Figure`` / ``matplotlib.pyplot``

        :param str title: The title (metric) of the plot.
        :param str series: The series name (variant) of the reported plot.
        :param int iteration: The iteration number.
        :param MatplotlibFigure figure: A ``matplotlib`` Figure object
        :param report_image: Default False. If True the plot will be uploaded as a debug sample (png image),
            and will appear under the debug samples tab (instead of the Plots tab).
        :param report_interactive: If True (default) it will try to convert the matplotlib into interactive
            plot in the UI. If False the matplotlib is saved as is and will
            be non-interactive (with the exception of zooming in/out)
        """
        # if task was not started, we have to start it
        self._start_task_if_needed()

        # noinspection PyProtectedMember
        self._task._reporter.report_matplotlib(
            title=title,
            series=series,
            figure=figure,
            iter=iteration or 0,
            logger=self,
            force_save_as_image=False if report_interactive and not report_image
            else ('png' if report_image else True),
        )

    def set_default_upload_destination(self, uri):
        # type: (str) -> None
        """
        Set the destination storage URI (for example, S3, Google Cloud Storage, a file path) for uploading debug images.

        The images are uploaded separately. A link to each image is reported.

        .. note::
           Credentials for the destination storage are specified in the  ClearML configuration file,
           ``~/clearml.conf``.

        :param str uri: example: 's3://bucket/directory/' or 'file:///tmp/debug/'

        :return: True, if the destination scheme is supported (for example, ``s3://``, ``file://``, or ``gc://``).
            False, if not supported.

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

            For example: ``s3://bucket/directory/``, or ``file:///tmp/debug/``.
        """
        # noinspection PyProtectedMember
        return self._default_upload_destination or self._task._get_default_report_storage_uri()

    def flush(self):
        # type: () -> bool
        """
        Flush cached reports and console outputs to backend.

        :return: True, if successfully flushed the cache. False, if failed.
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

        Deprecated - Use ``sdk.development.worker.report_period_sec`` to externally control the flush period.

        :param float period: The period to flush the logger in seconds. To set no periodic flush,
            specify ``None`` or ``0``.
        """
        pass

    def report_image_and_upload(
            self,
            title,  # type: str
            series,  # type: str
            iteration=None,  # type: Optional[int]
            path=None,  # type: Optional[str]
            matrix=None,  # type: Optional[Union[np.ndarray, Image.Image]]
            max_image_history=None,  # type: Optional[int]
            delete_after_upload=False  # type: bool
    ):
        """
        .. deprecated:: 0.13.0
            Use :meth:`Logger.report_image` instead
        """
        self.report_image(title=title, series=series, iteration=iteration or 0, local_path=path, image=matrix,
                          max_image_history=max_image_history, delete_after_upload=delete_after_upload)

    def capture_logging(self):
        # type: () -> "_LoggingContext"
        """
        Return context capturing all the logs (via logging) reported under the context

        :return: a ContextManager
        """
        class _LoggingContext(object):
            def __init__(self, a_logger):
                self.logger = a_logger

            def __enter__(self, *_, **__):
                if not self.logger:
                    return
                StdStreamPatch.patch_logging_formatter(self.logger)

            def __exit__(self, *_, **__):
                if not self.logger:
                    return
                StdStreamPatch.remove_patch_logging_formatter()

        # Do nothing if we already have full logging support
        return _LoggingContext(None if self._connect_logging else self)

    @classmethod
    def tensorboard_auto_group_scalars(cls, group_scalars=False):
        # type: (bool) -> None
        """
        Group together TensorBoard scalars that do not have a title, or assign a title/series with the same tag.

        :param group_scalars: Group TensorBoard scalars without a title

            The values are:

            - ``True`` - Scalars without specific titles are grouped together in the "Scalars" plot, preserving
              backward compatibility with ClearML automagical behavior.
            - ``False`` - TensorBoard scalars without titles get a title/series with the same tag. (default)
        :type group_scalars: bool
        """
        cls._tensorboard_logging_auto_group_scalars = group_scalars

    @classmethod
    def tensorboard_single_series_per_graph(cls, single_series=False):
        # type: (bool) -> None
        """
        Deprecated, this is now controlled from the UI!
        Group TensorBoard scalar series together or in separate plots.

        :param single_series: Group TensorBoard scalar series together

            The values are:

            - ``True`` - Generate a separate plot for each TensorBoard scalar series.
            - ``False`` - Group the TensorBoard scalar series together in the same plot. (default)

        :type single_series: bool
        """
        cls._tensorboard_single_series_per_graph = single_series

    @classmethod
    def matplotlib_force_report_non_interactive(cls, force):
        # type: (bool) -> None
        """
        If True all matplotlib are always converted to non interactive static plots (images), appearing in under
        the Plots section. If False (default), matplotlib figures are converted into interactive web UI plotly
        figures, in case figure conversion fails, it defaults to non-interactive plots.

        :param force: If True all matplotlib figures are converted automatically to non-interactive plots.
        """
        from clearml.backend_interface.metrics import Reporter
        Reporter.matplotlib_force_report_non_interactive(force=force)

    @classmethod
    def _remove_std_logger(cls):
        # noinspection PyBroadException
        try:
            StdStreamPatch.remove_std_logger()
        except Exception:
            return False
        return True

    def _console(self, msg, level=logging.INFO, omit_console=False, *args, **_):
        # type: (str, int, bool, Any, Any) -> None
        """
        print text to log (same as print to console, and also prints to console)

        :param str msg: text to print to the console (always send to the backend and displayed in console)
        :param level: logging level, default: logging.INFO
        :type level: Logging Level
        :param bool omit_console: Omit the console output, and only send the ``msg`` value to the log

            - ``True`` - Omit the console output.
            - ``False`` - Print the console output. (default)

        """
        try:
            level = int(level)
        except (TypeError, ValueError):
            self._task.log.log(level=logging.ERROR,
                               msg='Logger failed casting log level "%s" to integer' % str(level))
            level = logging.INFO

        # noinspection PyProtectedMember
        if not self._skip_console_log() or not self._task._is_remote_main_task():
            if self._task_handler:
                # noinspection PyBroadException
                try:
                    record = self._task.log.makeRecord(
                        "console", level=level, fn='', lno=0, func='', msg=msg, args=args, exc_info=None
                    )
                    # find the task handler that matches our task
                    self._task_handler.emit(record)
                except Exception:
                    # avoid infinite loop, output directly to stderr
                    # noinspection PyBroadException
                    try:
                        # make sure we are writing to the original stdout
                        StdStreamPatch.stderr_original_write(
                            'clearml.Logger failed sending log [level {}]: "{}"\n'.format(level, msg))
                    except Exception:
                        pass
            else:
                # noinspection PyProtectedMember
                self._task._reporter.report_console(message=msg, level=level)

        if not omit_console:
            # if we are here and we grabbed the stdout, we need to print the real thing
            if self._connect_std_streams and not self._skip_console_log():
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
            iteration=None,  # type: Optional[int]
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
        :type matrix: np.array
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

        # noinspection PyProtectedMember
        self._task._reporter.report_image_plot_and_upload(
            title=title,
            series=series,
            path=path,
            matrix=matrix,
            iter=iteration or 0,
            upload_uri=upload_uri,
            max_image_history=max_image_history,
            delete_after_upload=delete_after_upload,
        )

    def _report_file_and_upload(
            self,
            title,  # type: str
            series,  # type: str
            iteration=None,  # type: Optional[int]
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
        # noinspection PyProtectedMember
        self._task._reporter.report_image_and_upload(
            title=title,
            series=series,
            path=path,
            image=None,
            iter=iteration or 0,
            upload_uri=upload_uri,
            max_image_history=max_file_history,
            delete_after_upload=delete_after_upload,
        )

    def _start_task_if_needed(self):
        # deprecated
        pass

    def _flush_stdout_handler(self):
        if self._task_handler:
            self._task_handler.flush()

    def _close_stdout_handler(self, wait=True):
        # detach the sys stdout/stderr
        if self._connect_std_streams:
            StdStreamPatch.remove_std_logger(self)

        if self._task_handler:
            t = self._task_handler
            self._task_handler = None
            t.close(wait)

    def _touch_title_series(self, title, series):
        # type: (str, str) -> None
        if title not in self._graph_titles:
            self._graph_titles[title] = set()
        self._graph_titles[title].add(series)

    def _get_used_title_series(self):
        # type: () -> dict
        return self._graph_titles

    def _get_tensorboard_series_prefix(self):
        # type: () -> Optional[str]
        """
        :return str: return a string prefix to put in front of every report combing from tensorboard
        """
        return self._tensorboard_series_force_prefix

    def _set_tensorboard_series_prefix(self, prefix):
        # type: (Optional[str]) -> ()
        """
        :param str prefix: Set a string prefix to put in front of every report combing from tensorboard
        """
        self._tensorboard_series_force_prefix = str(prefix) if prefix else None

    @classmethod
    def _get_tensorboard_auto_group_scalars(cls):
        # type: () -> bool
        """
        :return: True, if we preserve Tensorboard backward compatibility behaviour,
            i.e., scalars without specific title will be under the "Scalars" graph
            default is False: Tensorboard scalars without title will have title/series with the same tag
        """
        return cls._tensorboard_logging_auto_group_scalars

    @classmethod
    def _get_tensorboard_single_series_per_graph(cls):
        # type: () -> bool
        """
        :return: True, if we generate a separate graph (plot) for each Tensorboard scalar series
            default is False: Tensorboard scalar series will be grouped according to their title
        """
        return cls._tensorboard_single_series_per_graph

    @classmethod
    def _skip_console_log(cls):
        return bool(running_remotely() and not DEBUG_SIMULATE_REMOTE_TASK.get())
