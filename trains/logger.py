import logging
import re
import sys
import threading
from functools import wraps

import numpy as np
from pathlib2 import Path

from .debugging.log import LoggerRoot
from .backend_interface.task.development.worker import DevWorker
from .backend_interface.task.log import TaskHandler
from .storage import StorageHelper
from .utilities.plotly_reporter import SeriesInfo
from .backend_api.services import tasks
from .backend_interface.task import Task as _Task
from .config import running_remotely, get_cache_dir


def _safe_names(func):
    """
    Validate the form of title and series parameters.

    This decorator assert that a method receives 'title' and 'series' as its
    first positional arguments, and that their values have only legal characters.

    '\', '/' and ':' will be replaced automatically by '_'
    Whitespace chars will be replaced automatically by ' '
    """
    _replacements = {
        '_': re.compile(r"[/\\:]"),
        ' ': re.compile(r"[\s]"),
    }

    def _make_safe(value):
        for repl, regex in _replacements.items():
            value = regex.sub(repl, value)
        return value

    @wraps(func)
    def fixed_names(self, title, series, *args, **kwargs):
        title = _make_safe(title)
        series = _make_safe(series)

        func(self, title, series, *args, **kwargs)

    return fixed_names


class Logger(object):
    """
    Console log and metric statistics interface.

    This is how we send graphs/plots/text to the system, later we can compare the performance of different tasks.

    **Usage: Task.get_logger()**
    """
    SeriesInfo = SeriesInfo
    _stdout_proxy = None
    _stderr_proxy = None
    _stdout_original_write = None

    def __init__(self, private_task):
        """
        **Do not construct Logger manually!**

        please use Task.get_logger()
        """
        assert isinstance(private_task, _Task), \
            'Logger object cannot be instantiated externally, use Task.get_logger()'
        super(Logger, self).__init__()
        self._task = private_task
        self._default_upload_destination = None
        self._flusher = None
        self._report_worker = None
        self._task_handler = None

        if DevWorker.report_stdout and not PrintPatchLogger.patched and not running_remotely():
            Logger._stdout_proxy = PrintPatchLogger(sys.stdout, self, level=logging.INFO)
            Logger._stderr_proxy = PrintPatchLogger(sys.stderr, self, level=logging.ERROR)
            self._task_handler = TaskHandler(self._task.session, self._task.id, capacity=100)
            # noinspection PyBroadException
            try:
                if Logger._stdout_original_write is None:
                    Logger._stdout_original_write = sys.stdout.write
                # this will only work in python 3, guard it with try/catch
                if not hasattr(sys.stdout, '_original_write'):
                    sys.stdout._original_write = sys.stdout.write
                sys.stdout.write = stdout__patched__write__
                if not hasattr(sys.stderr, '_original_write'):
                    sys.stderr._original_write = sys.stderr.write
                sys.stderr.write = stderr__patched__write__
            except Exception:
                pass
            sys.stdout = Logger._stdout_proxy
            sys.stderr = Logger._stderr_proxy
            # patch the base streams of sys (this way colorama will keep its ANSI colors)
            # noinspection PyBroadException
            try:
                sys.__stderr__ = sys.stderr
            except Exception:
                pass
            # noinspection PyBroadException
            try:
                sys.__stdout__ = sys.stdout
            except Exception:
                pass

            # now check if we have loguru and make it re-register the handlers
            # because it sores internally the stream.write function, which we cant patch
            # noinspection PyBroadException
            try:
                from loguru import logger
                register_stderr = None
                register_stdout = None
                for k, v in logger._handlers.items():
                    if v._name == '<stderr>':
                        register_stderr = k
                    elif v._name == '<stdout>':
                        register_stderr = k
                if register_stderr is not None:
                    logger.remove(register_stderr)
                    logger.add(sys.stderr)
                if register_stdout is not None:
                    logger.remove(register_stdout)
                    logger.add(sys.stdout)
            except Exception:
                pass

        elif DevWorker.report_stdout and not running_remotely():
            self._task_handler = TaskHandler(self._task.session, self._task.id, capacity=100)
            if Logger._stdout_proxy:
                Logger._stdout_proxy.connect(self)
            if Logger._stderr_proxy:
                Logger._stderr_proxy.connect(self)

    def console(self, msg, level=logging.INFO, omit_console=False, *args, **kwargs):
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

        # noinspection PyBroadException
        try:
            record = self._task.log.makeRecord(
                "console", level=level, fn='', lno=0, func='', msg=msg, args=args, exc_info=None
            )
            # find the task handler
            if not self._task_handler:
                self._task_handler = [h for h in LoggerRoot.get_base_logger().handlers if isinstance(h, TaskHandler)][0]
            self._task_handler.emit(record)
        except Exception:
            self._task.log.log(level=logging.ERROR,
                               msg='Logger failed sending log: [level %s]: "%s"' % (str(level), str(msg)))

        if not omit_console:
            # if we are here and we grabbed the stdout, we need to print the real thing
            if DevWorker.report_stdout:
                # noinspection PyBroadException
                try:
                    # make sure we are writing to the original stdout
                    Logger._stdout_original_write(str(msg)+'\n')
                except Exception:
                    pass
            else:
                print(str(msg))

        # if task was not started, we have to start it
        self._start_task_if_needed()

    def report_text(self, msg, level=logging.INFO, print_console=False, *args, **_):
        return self.console(msg, level, not print_console, *args, **_)

    def debug(self, msg, *args, **kwargs):
        """ Print information to the log. This is the same as console(msg, logging.DEBUG) """
        self._task.log.log(msg=msg, level=logging.DEBUG, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """ Print information to the log. This is the same as console(msg, logging.INFO) """
        self._task.log.log(msg=msg, level=logging.INFO, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        """ Print a warning to the log. This is the same as console(msg, logging.WARNING) """
        self._task.log.log(msg=msg, level=logging.WARNING, *args, **kwargs)

    warning = warn

    def error(self, msg, *args, **kwargs):
        """ Print an error to the log. This is the same as console(msg, logging.ERROR) """
        self._task.log.log(msg=msg, level=logging.ERROR, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        """ Print a fatal error to the log. This is the same as console(msg, logging.FATAL) """
        self._task.log.log(msg=msg, level=logging.FATAL, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """ Print a critical error to the log. This is the same as console(msg, logging.CRITICAL) """
        self._task.log.log(msg=msg, level=logging.CRITICAL, *args, **kwargs)

    def report_scalar(self, title, series, value, iteration):
        """
        Report a scalar value

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param value: Reported value
        :type value: float
        :param iteration: Iteration number
        :type value: int
        """

        # if task was not started, we have to start it
        self._start_task_if_needed()

        return self._task.reporter.report_scalar(title=title, series=series, value=float(value), iter=iteration)

    def report_vector(self, title, series, values, iteration, labels=None, xlabels=None):
        """
        Report a histogram plot

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param values: Reported values (or numpy array)
        :type values: [float]
        :param iteration: Iteration number
        :type iteration: int
        :param labels: optional, labels for each bar group.
        :type labels: list of strings.
        :param xlabels: optional label per entry in the vector (bucket in the histogram)
        :type xlabels: list of strings.
        """
        return self.report_histogram(title, series, values, iteration, labels=labels, xlabels=xlabels)

    def report_histogram(self, title, series, values, iteration, labels=None, xlabels=None):
        """
        Report a histogram plot

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param values: Reported values (or numpy array)
        :type values: [float]
        :param iteration: Iteration number
        :type iteration: int
        :param labels: optional, labels for each bar group.
        :type labels: list of strings.
        :param xlabels: optional label per entry in the vector (bucket in the histogram)
        :type xlabels: list of strings.
        """

        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # if task was not started, we have to start it
        self._start_task_if_needed()

        return self._task.reporter.report_histogram(
            title=title,
            series=series,
            histogram=values,
            iter=iteration,
            labels=labels,
            xlabels=xlabels,
        )

    def report_line_plot(self, title, series, iteration, xaxis, yaxis, mode='lines', reverse_xaxis=False, comment=None):
        """
        Report a (possibly multiple) line plot.

        :param title: Title (AKA metric)
        :type title: str
        :param series: All the series' data, one for each line in the plot.
        :type series: An iterable of LineSeriesInfo.
        :param iteration: Iteration number
        :type iteration: int
        :param xaxis: optional x-axis title
        :param yaxis: optional y-axis title
        :param mode: scatter plot with 'lines'/'markers'/'lines+markers'
        :type mode: str
        :param reverse_xaxis: If true X axis will be displayed from high to low (reversed)
        :type reverse_xaxis: bool
        :param comment: comment underneath the title
        :type comment: str
        """

        series = [self.SeriesInfo(**s) if isinstance(s, dict) else s for s in series]

        # if task was not started, we have to start it
        self._start_task_if_needed()

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

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param scatter: A scattered data: list of (pairs of x,y) (or numpy array)
        :type scatter: ndarray or list
        :param iteration: Iteration number
        :type iteration: int
        :param xaxis: optional x-axis title
        :param yaxis: optional y-axis title
        :param labels: label (text) per point in the scatter (in the same order)
        :param mode: scatter plot with 'lines'/'markers'/'lines+markers'
        :type mode: str
        :param comment: comment underneath the title
        :type comment: str
        """

        if not isinstance(scatter, np.ndarray):
            if not isinstance(scatter, list):
                scatter = list(scatter)
            scatter = np.array(scatter)

        # if task was not started, we have to start it
        self._start_task_if_needed()

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

    def report_scatter3d(self, title, series, scatter, iteration, labels=None, mode='markers',
                         fill=False, comment=None):
        """
        Report a 3d scatter graph (with markers)

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param scatter: A scattered data: list of (pairs of x,y,z) (or numpy array) or list of series [[(x1,y1,z1)...]]
        :type scatter: ndarray or list
        :param iteration: Iteration number
        :type iteration: int
        :param labels: label (text) per point in the scatter (in the same order)
        :param mode: scatter plot with 'lines'/'markers'/'lines+markers'
        :param fill: fill area under the curve
        :param comment: comment underneath the title
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

        return self._task.reporter.report_3d_scatter(
            title=title,
            series=series,
            data=scatter,
            iter=iteration,
            labels=labels,
            mode=mode,
            fill=fill,
            comment=comment,
        )

    def report_confusion_matrix(self, title, series, matrix, iteration, xlabels=None, ylabels=None, comment=None):
        """
        Report a heat-map matrix

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param matrix: A heat-map matrix (example: confusion matrix)
        :type matrix: ndarray
        :param iteration: Iteration number
        :type iteration: int
        :param xlabels: optional label per column of the matrix
        :param ylabels: optional label per row of the matrix
        :param comment: comment underneath the title
        """

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        # if task was not started, we have to start it
        self._start_task_if_needed()

        return self._task.reporter.report_value_matrix(
            title=title,
            series=series,
            data=matrix.astype(np.float32),
            iter=iteration,
            xlabels=xlabels,
            ylabels=ylabels,
            comment=comment,
        )

    def report_matrix(self, title, series, matrix, iteration, xlabels=None, ylabels=None):
        """
        Same as report_confusion_matrix
        Report a heat-map matrix

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param matrix: A heat-map matrix (example: confusion matrix)
        :type matrix: ndarray
        :param iteration: Iteration number
        :type iteration: int
        :param xlabels: optional label per column of the matrix
        :param ylabels: optional label per row of the matrix
        """
        return self.report_confusion_matrix(title, series, matrix, iteration, xlabels=xlabels, ylabels=ylabels)

    def report_surface(self, title, series, matrix, iteration, xlabels=None, ylabels=None,
                       xtitle=None, ytitle=None, camera=None, comment=None):
        """
        Report a 3d surface (same data as heat-map matrix, only presented differently)

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param matrix: A heat-map matrix (example: confusion matrix)
        :type matrix: ndarray
        :param iteration: Iteration number
        :type iteration: int
        :param xlabels: optional label per column of the matrix
        :param ylabels: optional label per row of the matrix
        :param xtitle: optional x-axis title
        :param ytitle: optional y-axis title
        :param camera: X,Y,Z camera position. def: (1,1,1)
        :param comment: comment underneath the title
        """

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        # if task was not started, we have to start it
        self._start_task_if_needed()

        return self._task.reporter.report_value_surface(
            title=title,
            series=series,
            data=matrix.astype(np.float32),
            iter=iteration,
            xlabels=xlabels,
            ylabels=ylabels,
            xtitle=xtitle,
            ytitle=ytitle,
            camera=camera,
            comment=comment,
        )

    @_safe_names
    def report_image(self, title, series, src, iteration):
        """
        Report an image, and register the 'src' as url content.

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param src: Image source URI. This URI will be used by the webapp and workers when trying to obtain the image \
        for presentation of processing. Currently only http(s), file and s3 schemes are supported.
        :type src: str
        :param iteration: Iteration number
        :type iteration: int
        """

        # if task was not started, we have to start it
        self._start_task_if_needed()

        self._task.reporter.report_image(
            title=title,
            series=series,
            src=src,
            iter=iteration,
        )

    @_safe_names
    def report_image_and_upload(self, title, series, iteration, path=None, matrix=None, max_image_history=None,
                                delete_after_upload=False):
        """
        Report an image and upload its contents.

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
        upload_uri = self._default_upload_destination or self._task._get_default_report_storage_uri()
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
            matrix=matrix,
            iter=iteration,
            upload_uri=upload_uri,
            max_image_history=max_image_history,
            delete_after_upload=delete_after_upload,
        )

    def report_image_plot_and_upload(self, title, series, iteration, path=None, matrix=None, max_image_history=None,
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
        upload_uri = self._default_upload_destination or self._task._get_default_report_storage_uri()
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

    def set_default_upload_destination(self, uri):
        """
        Set the uri to upload all the debug images to.

        Images are uploaded separately to the destination storage (e.g. s3,gc,file) and then
        a link to the uploaded image is sent in the report
        Notice: credentials for the upload destination will be pooled from the
        global configuration file (i.e. ~/trains.conf)

        :param uri: example: 's3://bucket/directory/' or 'file:///tmp/debug/'
        :return: True if destination scheme is supported (i.e. s3:// file:// gc:// etc...)
        """

        # Create the storage helper
        storage = StorageHelper.get(uri)

        # Verify that we can upload to this destination
        uri = storage.verify_upload(folder_uri=uri)

        self._default_upload_destination = uri

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
        if self._flusher:
            return self._flusher.period
        return None

    def set_flush_period(self, period):
        """
        Set the period of the logger flush.

        :param period: The period to flush the logger in seconds. If None or 0,
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
            self._flusher = _Flusher(self, period)
            self._flusher.start()

    @classmethod
    def _remove_std_logger(self):
        if isinstance(sys.stdout, PrintPatchLogger):
            # noinspection PyBroadException
            try:
                sys.stdout.connect(None)
            except Exception:
                pass
        if isinstance(sys.stderr, PrintPatchLogger):
            # noinspection PyBroadException
            try:
                sys.stderr.connect(None)
            except Exception:
                pass

    def _start_task_if_needed(self):
        if self._task._status == tasks.TaskStatusEnum.created:
            self._task.mark_started()

        self._task._dev_mode_task_start()

    def _flush_stdout_handler(self):
        if self._task_handler and DevWorker.report_stdout:
            self._task_handler.flush()


def stdout__patched__write__(*args, **kwargs):
    if Logger._stdout_proxy:
        return Logger._stdout_proxy.write(*args, **kwargs)
    return sys.stdout._original_write(*args, **kwargs)


def stderr__patched__write__(*args, **kwargs):
    if Logger._stderr_proxy:
        return Logger._stderr_proxy.write(*args, **kwargs)
    return sys.stderr._original_write(*args, **kwargs)


class PrintPatchLogger(object):
    """
    Allowed patching a stream into the logger.
    Used for capturing and logging stdin and stderr when running in development mode pseudo worker.
    """
    patched = False
    lock = threading.Lock()
    recursion_protect_lock = threading.RLock()

    def __init__(self, stream, logger=None, level=logging.INFO):
        PrintPatchLogger.patched = True
        self._terminal = stream
        self._log = logger
        self._log_level = level
        self._cur_line = ''

    def write(self, message):
        # make sure that we do not end up in infinite loop (i.e. log.console ends up calling us)
        if self._log and not PrintPatchLogger.recursion_protect_lock._is_owned():
            try:
                self.lock.acquire()
                with PrintPatchLogger.recursion_protect_lock:
                    if hasattr(self._terminal, '_original_write'):
                        self._terminal._original_write(message)
                    else:
                        self._terminal.write(message)

                do_flush = '\n' in message
                do_cr = '\r' in message
                self._cur_line += message
                if (not do_flush and not do_cr) or not message:
                    return
                last_lf = self._cur_line.rindex('\n' if do_flush else '\r')
                next_line = self._cur_line[last_lf + 1:]
                cur_line = self._cur_line[:last_lf + 1].rstrip()
                self._cur_line = next_line
            finally:
                self.lock.release()

            if cur_line:
                with PrintPatchLogger.recursion_protect_lock:
                    # noinspection PyBroadException
                    try:
                        if self._log:
                            self._log.console(cur_line, level=self._log_level, omit_console=True)
                    except Exception:
                        # what can we do, nothing
                        pass
        else:
            if hasattr(self._terminal, '_original_write'):
                self._terminal._original_write(message)
            else:
                self._terminal.write(message)

    def connect(self, logger):
        self._cur_line = ''
        self._log = logger

    def __getattr__(self, attr):
        if attr in ['_log', '_terminal', '_log_level', '_cur_line']:
            return self.__dict__.get(attr)
        return getattr(self._terminal, attr)

    def __setattr__(self, key, value):
        if key in ['_log', '_terminal', '_log_level', '_cur_line']:
            self.__dict__[key] = value
        else:
            return setattr(self._terminal, key, value)


class _Flusher(threading.Thread):
    def __init__(self, logger, period, **kwargs):
        super(_Flusher, self).__init__(**kwargs)
        self.daemon = True

        self._period = period
        self._logger = logger
        self._exit_event = threading.Event()

    @property
    def period(self):
        return self._period

    def run(self):
        self._logger.flush()
        # store original wait period
        while True:
            period = self._period
            while not self._exit_event.wait(period or 1.0):
                self._logger.flush()
            # check if period is negative or None we should exit
            if self._period is None or self._period < 0:
                break
            # check if period was changed, we should restart
            self._exit_event.clear()

    def exit(self):
        self._period = None
        self._exit_event.set()

    def set_period(self, period):
        self._period = period
        # make sure we exit the previous wait
        self._exit_event.set()
