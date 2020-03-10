import logging
import warnings

import numpy as np
import six

try:
    import pandas as pd
except ImportError:
    pd = None
from PIL import Image
from pathlib2 import Path

from .backend_api.services import tasks
from .backend_interface.logger import StdStreamPatch, LogFlusher
from .backend_interface.task import Task as _Task
from .backend_interface.task.development.worker import DevWorker
from .backend_interface.task.log import TaskHandler
from .backend_interface.util import mutually_exclusive
from .config import running_remotely, get_cache_dir, config
from .debugging.log import LoggerRoot
from .errors import UsageError
from .storage import StorageHelper
from .utilities.plotly_reporter import SeriesInfo

# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning, module=__name__)


class Logger(object):
    """
    Console log and metric statistics interface.

    This is how we send graphs/plots/text to the system, later we can compare the performance of different tasks.

    **Usage:** :func:`Logger.current_logger` or :func:`Task.get_logger`
    """
    SeriesInfo = SeriesInfo
    _tensorboard_logging_auto_group_scalars = False
    _tensorboard_single_series_per_graph = config.get('metrics.tensorboard_single_series_per_graph', False)

    def __init__(self, private_task):
        """
        **Do not construct Logger manually!**

        please use :func:`Logger.current_logger`
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
        Return a logger object for the current task. Can be called from anywhere in the code

        :return: Singleton Logger object for the current running task
        """
        from .task import Task
        task = Task.current_task()
        if not task:
            return None
        return task.get_logger()

    def report_text(self, msg, level=logging.INFO, print_console=True, *args, **_):
        """
        print text to log and optionally also prints to console

        :param str msg: text to print to the console (always send to the backend and displayed in console)
        :param int level: logging level, default: logging.INFO
        :param bool print_console: If True we also print 'msg' to console
        """
        return self._console(msg, level, not print_console, *args, **_)

    def report_scalar(self, title, series, value, iteration):
        """
        Report a scalar value

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param float value: Reported value
        :param int iteration: Iteration number
        """

        # if task was not started, we have to start it
        self._start_task_if_needed()
        self._touch_title_series(title, series)
        return self._task.reporter.report_scalar(title=title, series=series, value=float(value), iter=iteration)

    def report_vector(self, title, series, values, iteration, labels=None, xlabels=None,
                      xaxis=None, yaxis=None):
        """
        Report a histogram plot

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param list(float) values: Reported values (or numpy array)
        :param int iteration: Iteration number
        :param list(str) labels: optional, labels for each bar group.
        :param list(str) xlabels: optional label per entry in the vector (bucket in the histogram)
        :param str xaxis: optional x-axis title
        :param str yaxis: optional y-axis title
        """
        self._touch_title_series(title, series)
        return self.report_histogram(title, series, values, iteration, labels=labels, xlabels=xlabels,
                                     xaxis=xaxis, yaxis=yaxis)

    def report_histogram(self, title, series, values, iteration, labels=None, xlabels=None,
                         xaxis=None, yaxis=None):
        """
        Report a histogram plot

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param list(float) values: Reported values (or numpy array)
        :param int iteration: Iteration number
        :param list(str) labels: optional, labels for each bar group.
        :param list(str) xlabels: optional label per entry in the vector (bucket in the histogram)
        :param str xaxis: optional x-axis title
        :param str yaxis: optional y-axis title
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
        )

    def report_table(self, title, series, iteration, table_plot=None, csv=None, url=None):
        """
        Report a table plot.

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param iteration: Iteration number
        :type iteration: int
        :param table_plot: The output table plot object
        :type table_plot: pandas.DataFrame
        :param csv: path to local csv file
        :type csv: str
        :param url: A URL to the location of csv file.
        :type url: str

        .. note::
            :paramref:`~.Logger.report_table.table_plot`, :paramref:`~.Logger.report_table.csv`
            and :paramref:`~.Logger.report_table.url' are mutually exclusive, and at least one must be provided.
        """
        mutually_exclusive(
            UsageError, _check_none=True,
            table_plot=table_plot, csv=csv, url=url
        )
        table = table_plot
        if url or csv:
            if not pd:
                raise UsageError(
                    "pandas is required in order to support reporting tables using CSV or a URL, please install the pandas python package"
                )
            if url:
                table = pd.read_csv(url)
            elif csv:
                table = pd.read_csv(csv)

        return self._task.reporter.report_table(
            title=title,
            series=series,
            table=table,
            iteration=iteration
        )

    def report_line_plot(self, title, series, iteration, xaxis, yaxis, mode='lines',
                         reverse_xaxis=False, comment=None):
        """
        Report a (possibly multiple) line plot.

        :param str title: Title (AKA metric)
        :param list(LineSeriesInfo) series: All the series' data, one for each line in the plot.
        :param int iteration: Iteration number
        :param str xaxis: optional x-axis title
        :param str yaxis: optional y-axis title
        :param str mode: scatter plot with 'lines'/'markers'/'lines+markers'
        :param bool reverse_xaxis: If true X axis will be displayed from high to low (reversed)
        :param str comment: comment underneath the title
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
        )

    def report_scatter2d(self, title, series, scatter, iteration, xaxis=None, yaxis=None, labels=None,
                         mode='lines', comment=None):
        """
        Report a 2d scatter graph (with lines)

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param np.ndarray scatter: A scattered data: list of (pairs of x,y) (or numpy array)
        :param int iteration: Iteration number
        :param str xaxis: optional x-axis title
        :param str yaxis: optional y-axis title
        :param list(str) labels: label (text) per point in the scatter (in the same order)
        :param str mode: scatter plot with 'lines'/'markers'/'lines+markers'
        :param str comment: comment underneath the title
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
        )

    def report_scatter3d(self, title, series, scatter, iteration, xaxis=None, yaxis=None, zaxis=None,
                         labels=None, mode='markers', fill=False, comment=None):
        """
        Report a 3d scatter graph (with markers)

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param Union[np.ndarray, list] scatter: A scattered data: list of (pairs of x,y,z) (or numpy array)
            or list of series [[(x1,y1,z1)...]]
        :param int iteration: Iteration number
        :param str xaxis: optional x-axis title
        :param str yaxis: optional y-axis title
        :param str zaxis: optional z-axis title
        :param list(str) labels: label (text) per point in the scatter (in the same order)
        :param str mode: scatter plot with 'lines'/'markers'/'lines+markers'
        :param bool fill: fill area under the curve
        :param str comment: comment underneath the title
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
        )

    def report_confusion_matrix(self, title, series, matrix, iteration, xaxis=None, yaxis=None,
                                xlabels=None, ylabels=None, comment=None):
        """
        Report a heat-map matrix

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param np.ndarray matrix: A heat-map matrix (example: confusion matrix)
        :param int iteration: Iteration number
        :param str xaxis: optional x-axis title
        :param str yaxis: optional y-axis title
        :param list(str) xlabels: optional label per column of the matrix
        :param list(str) ylabels: optional label per row of the matrix
        :param str comment: comment underneath the title
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
        )

    def report_matrix(self, title, series, matrix, iteration, xaxis=None, yaxis=None, xlabels=None, ylabels=None):
        """
        Same as report_confusion_matrix
        Report a heat-map matrix

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param np.ndarray matrix: A heat-map matrix (example: confusion matrix)
        :param int iteration: Iteration number
        :param str xaxis: optional x-axis title
        :param str yaxis: optional y-axis title
        :param list(str) xlabels: optional label per column of the matrix
        :param list(str) ylabels: optional label per row of the matrix
        """
        self._touch_title_series(title, series)
        return self.report_confusion_matrix(title, series, matrix, iteration,
                                            xaxis=xaxis, yaxis=yaxis, xlabels=xlabels, ylabels=ylabels)

    def report_surface(self, title, series, matrix, iteration, xaxis=None, yaxis=None, zaxis=None,
                       xlabels=None, ylabels=None, camera=None, comment=None):
        """
        Report a 3d surface (same data as heat-map matrix, only presented differently)

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param np.ndarray matrix: A heat-map matrix (example: confusion matrix)
        :param int iteration: Iteration number
        :param str xaxis: optional x-axis title
        :param str yaxis: optional y-axis title
        :param str zaxis: optional z-axis title
        :param list(str) xlabels: optional label per column of the matrix
        :param list(str) ylabels: optional label per row of the matrix
        :param list(float) camera: X,Y,Z camera position. def: (1,1,1)
        :param str comment: comment underneath the title
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
        )

    def report_image(self, title, series, iteration, local_path=None, image=None, matrix=None, max_image_history=None,
                     delete_after_upload=False, url=None):
        """
        Report an image and upload its contents.

        Image is uploaded to a preconfigured bucket (see setup_upload()) with a key (filename)
        describing the task ID, title, series and iteration.

        .. note::
            :paramref:`~.Logger.report_image.local_path`, :paramref:`~.Logger.report_image.url`, :paramref:`~.Logger.report_image.image` and :paramref:`~.Logger.report_image.matrix`
            are mutually exclusive, and at least one must be provided.

        :param str title: Title (AKA metric)
        :param str series: Series (AKA variant)
        :param int iteration: Iteration number
        :param str local_path: A path to an image file.
        :param str url: A URL to the location of a pre-uploaded image.
        :param np.ndarray or PIL.Image.Image image: Could be a PIL.Image.Image object or a 3D numpy.ndarray
                object containing image data (RGB).
        :param np.ndarray matrix: A 3D numpy.ndarray object containing image data (RGB).
                This is deprecated, use image variable instead.
        :param int max_image_history: maximum number of image to store per metric/variant combination
            use negative value for unlimited. default is set in global configuration (default=5)
        :param bool delete_after_upload: if True, one the file was uploaded the local copy will be deleted
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

    def set_default_upload_destination(self, uri):
        """
        Set the uri to upload all the debug images to.

        Images are uploaded separately to the destination storage (e.g. s3,gc,file) and then
        a link to the uploaded image is sent in the report
        Notice: credentials for the upload destination will be pooled from the
        global configuration file (i.e. ~/trains.conf)

        :param str uri: example: 's3://bucket/directory/' or 'file:///tmp/debug/'
        :return: True if destination scheme is supported (i.e. s3:// file:// gc:// etc...)
        """

        # Create the storage helper
        storage = StorageHelper.get(uri)

        # Verify that we can upload to this destination
        uri = storage.verify_upload(folder_uri=uri)

        self._default_upload_destination = uri

    def get_default_upload_destination(self):
        """
        Get the uri to upload all the debug images to.

        Images are uploaded separately to the destination storage (e.g. s3,gc,file) and then
        a link to the uploaded image is sent in the report
        Notice: credentials for the upload destination will be pooled from the
        global configuration file (i.e. ~/trains.conf)

        :return: Uri (str)  example: 's3://bucket/directory/' or 'file:///tmp/debug/' etc...
        """
        return self._default_upload_destination or self._task._get_default_report_storage_uri()

    def flush(self):
        """
        Flush cached reports and console outputs to backend.

        :return: True if successful
        """
        self._flush_stdout_handler()
        if self._task:
            return self._task.flush()
        return False

    def get_flush_period(self):
        """
        :return: logger flush period in seconds
        """
        if self._flusher:
            return self._flusher.period
        return None

    def set_flush_period(self, period):
        """
        Set the period of the logger flush.

        :param float period: The period to flush the logger in seconds. If None or 0,
            There will be no periodic flush.
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

    def report_image_and_upload(self, title, series, iteration, path=None, matrix=None, max_image_history=None,
                                delete_after_upload=False):
        """
        Deprecated: Backwards compatibility, please use report_image instead
        """
        self.report_image(title=title, series=series, iteration=iteration, local_path=path, image=matrix,
                          max_image_history=max_image_history, delete_after_upload=delete_after_upload)

    @classmethod
    def tensorboard_auto_group_scalars(cls, group_scalars=False):
        """
        If `group_scalars` set to True, we preserve backward compatible Tensorboard auto-magic behaviour,
        i.e. Scalars without specific title will be grouped under the "Scalars" graph.
        Default is False: Tensorboard scalars without title will have title/series with the same tag
        """
        cls._tensorboard_logging_auto_group_scalars = group_scalars

    @classmethod
    def tensorboard_single_series_per_graph(cls, single_series=False):
        """
        If `single_series` set to True, we generate a separate graph (plot) for each Tensorboard scalar series
        Default is False: Tensorboard scalar series will be grouped according to their title
        """
        cls._tensorboard_logging_single_series_per_graphs = single_series

    @classmethod
    def _remove_std_logger(cls):
        StdStreamPatch.remove_std_logger()

    def _console(self, msg, level=logging.INFO, omit_console=False, *args, **kwargs):
        """
        print text to log (same as print to console, and also prints to console)

        :param msg: text to print to the console (always send to the backend and displayed in console)
        :param level: logging level, default: logging.INFO
        :param omit_console: If True we only send 'msg' to log (no console print)
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
                LoggerRoot.get_base_logger().warning(msg='Logger failed sending log: [level %s]: "%s"'
                                                         % (str(level), str(msg)))

        if not omit_console:
            # if we are here and we grabbed the stdout, we need to print the real thing
            if DevWorker.report_stdout and not running_remotely():
                # noinspection PyBroadException
                try:
                    # make sure we are writing to the original stdout
                    StdStreamPatch.stdout_original_write(str(msg)+'\n')
                except Exception:
                    pass
            else:
                print(str(msg))

        # if task was not started, we have to start it
        self._start_task_if_needed()

    def _report_image_plot_and_upload(self, title, series, iteration, path=None, matrix=None, max_image_history=None,
                                      delete_after_upload=False):
        """
        Report an image, upload its contents, and present in plots section using plotly

        Image is uploaded to a preconfigured bucket (see setup_upload()) with a key (filename)
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

    def _report_file_and_upload(self, title, series, iteration, path=None, max_file_history=None,
                                delete_after_upload=False):
        """
        Upload a file and report it as link in the debug images section.

        File is uploaded to a preconfigured storage (see setup_upload()) with a key (filename)
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
        # do not refresh the task status read from cached variable _status
        if str(self._task._status) == str(tasks.TaskStatusEnum.created):
            self._task.mark_started()

        self._task._dev_mode_task_start()

    def _flush_stdout_handler(self):
        if self._task_handler and DevWorker.report_stdout:
            self._task_handler.flush()

    def _flush_wait_stdout_handler(self):
        if self._task_handler and DevWorker.report_stdout:
            self._task_handler.flush()
            self._task_handler.wait_for_flush()

    def _touch_title_series(self, title, series):
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
