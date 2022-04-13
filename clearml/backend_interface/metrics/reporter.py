import datetime
import json
import logging
import math
import os
from time import sleep, time

import numpy as np
import six
from six.moves.queue import Empty

from .events import (
    ScalarEvent, VectorEvent, ImageEvent, PlotEvent, ImageEventNoUpload,
    UploadEvent, MediaEvent, ConsoleEvent, )
from ..base import InterfaceBase
from ..setupuploadmixin import SetupUploadMixin
from ...config import config
from ...utilities.async_manager import AsyncManagerMixin
from ...utilities.plotly_reporter import (
    create_2d_histogram_plot, create_value_matrix, create_3d_surface,
    create_2d_scatter_series, create_3d_scatter_series, create_line_plot, plotly_scatter3d_layout_dict,
    create_image_plot, create_plotly_table, )
from ...utilities.process.mp import BackgroundMonitor, ForkSemaphore, ForkEvent, ForkQueue
from ...utilities.py3_interop import AbstractContextManager
from ...utilities.process.mp import SafeQueue as PrQueue, SafeEvent

try:
    from collections.abc import Iterable  # noqa
except ImportError:
    from collections import Iterable


class BackgroundReportService(BackgroundMonitor, AsyncManagerMixin):
    __daemon_live_check_timeout = 10.0

    def __init__(self, task, async_enable, metrics, flush_frequency, flush_threshold):
        super(BackgroundReportService, self).__init__(
            task=task, wait_period=flush_frequency)
        self._flush_threshold = flush_threshold
        self._flush_event = ForkEvent()
        self._empty_state_event = ForkEvent()
        self._queue = ForkQueue()
        self._queue_size = 0
        self._res_waiting = ForkSemaphore()
        self._metrics = metrics
        self._storage_uri = None
        self._async_enable = async_enable
        self._is_thread_mode_in_subprocess_flag = None

    def set_storage_uri(self, uri):
        self._storage_uri = uri

    def set_subprocess_mode(self):
        if isinstance(self._queue, ForkQueue):
            self._write()
            self._queue = PrQueue()
        if not isinstance(self._event, SafeEvent):
            self._event = SafeEvent()
        if not isinstance(self._empty_state_event, SafeEvent):
            self._empty_state_event = SafeEvent()
        super(BackgroundReportService, self).set_subprocess_mode()

    def stop(self):
        if isinstance(self._queue, PrQueue):
            self._queue.close(self._flush_event)
        if not self.is_subprocess_mode() or self.is_subprocess_alive():
            self._flush_event.set()
        super(BackgroundReportService, self).stop()

    def wait(self, timeout=None):
        if not self._done_ev:
            return
        if not self.is_subprocess_mode() or self.is_subprocess_mode_and_parent_process():
            tic = time()
            while self.is_alive() and (not timeout or time()-tic < timeout):
                if self._done_ev.wait(timeout=1.0):
                    break

    def flush(self):
        while isinstance(self._queue, PrQueue) and self._queue.is_pending():
            sleep(0.1)
        self._queue_size = 0
        # stop background process?!
        if not self.is_subprocess_mode() or self.is_subprocess_alive():
            self._flush_event.set()

    def wait_for_events(self, timeout=None):
        if self._is_subprocess_mode_and_not_parent_process() and self.get_at_exit_state():
            return

        # noinspection PyProtectedMember
        if self._is_subprocess_mode_and_not_parent_process():
            while self._queue and not self._queue.empty():
                sleep(0.1)
            return

        self._empty_state_event.clear()
        if isinstance(self._empty_state_event, ForkEvent):
            self._flush_event.set()
            tic = time()
            while self._thread and self._thread.is_alive() and (not timeout or time()-tic < timeout):
                if self._empty_state_event.wait(timeout=1.0):
                    break
                if self._event.wait(0) or self._done_ev.wait(0):
                    break
                # if enough time passed and the flush event was not cleared,
                # there is no daemon thread running, we should leave
                if time()-tic > self.__daemon_live_check_timeout and self._flush_event.wait(0):
                    break
        elif isinstance(self._empty_state_event, SafeEvent):
            tic = time()
            while self.is_subprocess_alive() and (not timeout or time()-tic < timeout):
                if self._empty_state_event.wait(timeout=1.0):
                    break

        return

    def add_event(self, ev):
        if not self._queue:
            return
        # check that we did not loose the reporter sub-process
        if self.is_subprocess_mode() and not self._fast_is_subprocess_alive():
            # we lost the reporting subprocess, let's switch to thread mode
            # gel all data, work on local queue:
            self._write()
            # replace queue:
            self._queue = ForkQueue()
            self._queue_size = 0
            self._event = ForkEvent()
            self._done_ev = ForkEvent()
            self._start_ev = ForkEvent()
            self._flush_event = ForkEvent()
            self._empty_state_event = ForkEvent()
            self._res_waiting = ForkSemaphore()
            # set thread mode
            self._subprocess = False
            self._is_thread_mode_in_subprocess_flag = None
            # start background thread
            self._thread = None
            self._start()
            logging.getLogger('clearml.reporter').warning(
                'Event reporting sub-process lost, switching to thread based reporting')

        self._queue.put(ev)
        self._queue_size += 1
        if self._queue_size >= self._flush_threshold:
            self.flush()

    def daemon(self):
        self._is_thread_mode_in_subprocess_flag = self._is_thread_mode_and_not_main_process()

        while not self._event.wait(0):
            self._flush_event.wait(self._wait_timeout)
            self._flush_event.clear()
            # lock state
            self._res_waiting.acquire()
            self._write()
            # wait for all reports
            if self.get_num_results() > 0:
                self.wait_for_results()
            # set empty flag only if we are not waiting for exit signal
            if not self._event.wait(0):
                self._empty_state_event.set()
            # unlock state
            self._res_waiting.release()
        # make sure we flushed everything
        self._async_enable = False
        self._res_waiting.acquire()
        self._write()
        if self.get_num_results() > 0:
            self.wait_for_results()
        self._empty_state_event.set()
        self._res_waiting.release()

    def _write(self):
        if self._queue.empty():
            return
        # print('reporting %d events' % len(self._events))
        events = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get(block=False))
            except Empty:
                break
        if not events:
            return
        if self._is_thread_mode_in_subprocess_flag:
            for e in events:
                if isinstance(e, UploadEvent):
                    # noinspection PyProtectedMember
                    e._generate_file_name(force_pid_suffix=os.getpid())

        res = self._metrics.write_events(
            events, async_enable=self._async_enable, storage_uri=self._storage_uri)

        if self._async_enable:
            self._add_async_result(res)

    def send_all_events(self, wait=True):
        self._write()
        if wait and self.get_num_results() > 0:
            self.wait_for_results()

    def events_waiting(self):
        if not self._queue.empty():
            return True
        if not self.is_alive():
            return False
        try:
            return not self._res_waiting.get_value()
        except NotImplementedError:
            return self.get_num_results() > 0


class Reporter(InterfaceBase, AbstractContextManager, SetupUploadMixin, AsyncManagerMixin):
    """
    A simple metrics reporter class.
    This class caches reports and supports both a explicit flushing and context-based flushing. To ensure reports are
     sent to the backend, please use (assuming an instance of Reporter named 'reporter'):
     - use the context manager feature (which will automatically flush when exiting the context):
        with reporter:
            reporter.report...
            ...
     - explicitly call flush:
        reporter.report...
        ...
        reporter.flush()
    """

    def __init__(self, metrics, task, flush_threshold=10, async_enable=False, use_subprocess=False):
        """
        Create a reporter
        :param metrics: A Metrics manager instance that handles actual reporting, uploads etc.
        :type metrics: .backend_interface.metrics.Metrics
        :param task: Task object
        :param flush_threshold: Events flush threshold. This determines the threshold over which cached reported events
            are flushed and sent to the backend.
        :type flush_threshold: int
        """
        log = metrics.log.getChild('reporter')
        log.setLevel(log.level)
        if self.__class__.max_float_num_digits == -1:
            self.__class__.max_float_num_digits = config.get('metrics.plot_max_num_digits', None)

        super(Reporter, self).__init__(session=metrics.session, log=log)
        self._metrics = metrics
        self._bucket_config = None
        self._storage_uri = None
        self._async_enable = async_enable
        self._flush_frequency = 5.0
        self._max_iteration = 0
        self._report_service = BackgroundReportService(
            task=task, async_enable=async_enable, metrics=metrics,
            flush_frequency=self._flush_frequency, flush_threshold=flush_threshold)
        self._report_service.start()

    def _set_storage_uri(self, value):
        value = '/'.join(x for x in (value.rstrip('/'), self._metrics.storage_key_prefix) if x)
        self._storage_uri = value
        self._report_service.set_storage_uri(self._storage_uri)

    storage_uri = property(None, _set_storage_uri)
    max_float_num_digits = -1

    @property
    def async_enable(self):
        return self._async_enable

    @async_enable.setter
    def async_enable(self, value):
        self._async_enable = bool(value)

    @property
    def max_iteration(self):
        return self._max_iteration

    def _report(self, ev):
        if not self._report_service:
            return
        ev_iteration = ev.get_iteration()
        if ev_iteration is not None:
            # we have to manually add get_iteration_offset() because event hasn't reached the Metric manager
            self._max_iteration = max(self._max_iteration, ev_iteration + self._metrics.get_iteration_offset())
        self._report_service.add_event(ev)

    def flush(self):
        """
        Flush cached reports to backend.
        """
        if self._report_service:
            self._report_service.flush()

    def wait_for_events(self, timeout=None):
        if self._report_service:
            return self._report_service.wait_for_events(timeout=timeout)

    def stop(self):
        if not self._report_service:
            return
        report_service = self._report_service
        self._report_service = None
        if not report_service.is_subprocess_mode() or report_service.is_alive():
            report_service.stop()
            report_service.wait()
        else:
            report_service.send_all_events()

    def is_alive(self):
        return self._report_service and self._report_service.is_alive()

    def is_constructed(self):
        # noinspection PyProtectedMember
        return self._report_service and (self._report_service.is_alive() or self._report_service._thread is True)

    def get_num_results(self):
        return self._report_service.get_num_results()

    def events_waiting(self):
        return self._report_service.events_waiting()

    def wait_for_results(self, *args, **kwargs):
        return self._report_service.wait_for_results(*args, **kwargs)

    def report_scalar(self, title, series, value, iter):
        """
        Report a scalar value
        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param value: Reported value
        :type value: float
        :param iter: Iteration number
        :type iter: int
        """
        ev = ScalarEvent(metric=self._normalize_name(title), variant=self._normalize_name(series), value=value,
                         iter=iter)
        self._report(ev)

    def report_vector(self, title, series, values, iter):
        """
        Report a vector of values
        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param values: Reported values
        :type values: [float]
        :param iter: Iteration number
        :type iter: int
        """
        if not isinstance(values, Iterable):
            raise ValueError('values: expected an iterable')
        ev = VectorEvent(metric=self._normalize_name(title), variant=self._normalize_name(series), values=values,
                         iter=iter)
        self._report(ev)

    def report_matplotlib(self, title, series, figure, iter, force_save_as_image=False, logger=None):
        from clearml.binding.matplotlib_bind import PatchedMatplotlib
        PatchedMatplotlib.report_figure(
            title=title,
            series=series,
            figure=figure,
            iter=iter,
            force_save_as_image=force_save_as_image if not isinstance(force_save_as_image, str) else False,
            report_as_debug_sample=force_save_as_image if isinstance(force_save_as_image, str) else False,
            reporter=self,
            logger=logger,
        )

    def report_plot(self, title, series, plot, iter, round_digits=None, nan_as_null=True):
        """
        Report a Plotly chart
        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param plot: A JSON describing a plotly chart (see https://help.plot.ly/json-chart-schema/)
        :type plot: str or dict
        :param iter: Iteration number
        :param round_digits: number of digits after the dot to leave
        :type round_digits: int
        :param nan_as_null: If True, Convert NaN to None (null), otherwise use 'nan'
        :type nan_as_null: bool
        """
        inf_value = math.inf if six.PY3 else float("inf")

        def to_base_type(o):
            if isinstance(o, float):
                if o != o:
                    return None if nan_as_null else 'nan'
                elif o == inf_value:
                    return 'inf'
                elif o == -inf_value:
                    return '-inf'
                return round(o, ndigits=round_digits) if round_digits is not None else o
            elif isinstance(o, (datetime.date, datetime.datetime)):
                return o.isoformat()
            return o

        # noinspection PyBroadException
        try:
            # Special json encoder for numpy types
            def default(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(round(obj, ndigits=round_digits) if round_digits is not None else obj)
                elif isinstance(obj, np.ndarray):
                    return [to_base_type(a) for a in obj.tolist()]
                return to_base_type(obj)

        except Exception:
            default = None

        if round_digits is None:
            round_digits = self.max_float_num_digits

        if round_digits is False:
            round_digits = None

        if isinstance(plot, dict):
            if 'data' in plot:
                for d in plot['data']:
                    if not isinstance(d, dict):
                        continue
                    for k, v in d.items():
                        if isinstance(v, list):
                            d[k] = list(to_base_type(s) for s in v)
                        elif isinstance(v, tuple):
                            d[k] = tuple(to_base_type(s) for s in v)
                        else:
                            d[k] = to_base_type(v)
            plot = json.dumps(plot, default=default)
        elif not isinstance(plot, six.string_types):
            raise ValueError('Plot should be a string or a dict')

        ev = PlotEvent(metric=self._normalize_name(title), variant=self._normalize_name(series),
                       plot_str=plot, iter=iter)
        self._report(ev)

    def report_image(self, title, series, src, iter):
        """
        Report an image.
        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param src: Image source URI. This URI will be used by the webapp and workers when trying to obtain the image
            for presentation of processing. Currently only http(s), file and s3 schemes are supported.
        :type src: str
        :param iter: Iteration number
        :type iter: int
        """
        ev = ImageEventNoUpload(metric=self._normalize_name(title), variant=self._normalize_name(series), iter=iter,
                                src=src)
        self._report(ev)

    def report_media(self, title, series, src, iter):
        """
        Report a media link.
        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param src: Media source URI. This URI will be used by the webapp and workers when trying to obtain the image
            for presentation of processing. Currently only http(s), file and s3 schemes are supported.
        :type src: str
        :param iter: Iteration number
        :type iter: int
        """
        ev = ImageEventNoUpload(metric=self._normalize_name(title), variant=self._normalize_name(series), iter=iter,
                                src=src)
        self._report(ev)

    def report_image_and_upload(self, title, series, iter, path=None, image=None, upload_uri=None,
                                max_image_history=None, delete_after_upload=False):
        """
        Report an image and upload its contents. Image is uploaded to a preconfigured bucket (see setup_upload()) with
         a key (filename) describing the task ID, title, series and iteration.

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param iter: Iteration number
        :type iter: int
        :param path: A path to an image file. Required unless matrix is provided.
        :type path: str
        :param image: Image data. Required unless filename is provided.
        :type image: A PIL.Image.Image object or a 3D numpy.ndarray object
        :param upload_uri: Destination URL
        :param max_image_history: maximum number of image to store per metric/variant combination
        use negative value for unlimited. default is set in global configuration (default=5)
        :param delete_after_upload: if True, one the file was uploaded the local copy will be deleted
        :type delete_after_upload: boolean
        """
        if not self._storage_uri and not upload_uri:
            raise ValueError('Upload configuration is required (use setup_upload())')
        if len([x for x in (path, image) if x is not None]) != 1:
            raise ValueError('Expected only one of [filename, image]')
        kwargs = dict(metric=self._normalize_name(title), variant=self._normalize_name(series), iter=iter,
                      file_history_size=max_image_history)
        ev = ImageEvent(image_data=image, upload_uri=upload_uri, local_image_path=path,
                        delete_after_upload=delete_after_upload, **kwargs)
        self._report(ev)

    def report_media_and_upload(self, title, series, iter, path=None, stream=None, upload_uri=None,
                                file_extension=None, max_history=None, delete_after_upload=False):
        """
        Report a media file/stream and upload its contents.
        Media is uploaded to a preconfigured bucket
        (see setup_upload()) with a key (filename) describing the task ID, title, series and iteration.

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param iter: Iteration number
        :type iter: int
        :param path: A path to an image file. Required unless matrix is provided.
        :type path: str
        :param stream: File/String stream
        :param upload_uri: Destination URL
        :param file_extension: file extension to use when stream is passed
        :param max_history: maximum number of files to store per metric/variant combination
        use negative value for unlimited. default is set in global configuration (default=5)
        :param delete_after_upload: if True, one the file was uploaded the local copy will be deleted
        :type delete_after_upload: boolean
        """
        if not self._storage_uri and not upload_uri:
            raise ValueError('Upload configuration is required (use setup_upload())')
        if len([x for x in (path, stream) if x is not None]) != 1:
            raise ValueError('Expected only one of [filename, stream]')
        if isinstance(stream, six.string_types):
            stream = six.StringIO(stream)

        kwargs = dict(metric=self._normalize_name(title), variant=self._normalize_name(series), iter=iter,
                      file_history_size=max_history)
        ev = MediaEvent(stream=stream, upload_uri=upload_uri, local_image_path=path,
                        override_filename_ext=file_extension,
                        delete_after_upload=delete_after_upload, **kwargs)
        self._report(ev)

    def report_histogram(self, title, series, histogram, iter, labels=None, xlabels=None,
                         xtitle=None, ytitle=None, comment=None, mode='group', layout_config=None):
        """
        Report an histogram bar plot
        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param histogram: The histogram data.
            A row for each dataset(bar in a bar group). A column for each bucket.
        :type histogram: numpy array
        :param iter: Iteration number
        :type iter: int
        :param labels: The labels for each bar group.
        :type labels: list of strings.
        :param xlabels: The labels of the x axis.
        :type xlabels: List of strings.
        :param str xtitle: optional x-axis title
        :param str ytitle: optional y-axis title
        :param comment: comment underneath the title
        :type comment: str
        :param mode: multiple histograms mode. valid options are: stack / group / relative. Default is 'group'.
        :type mode: str
        :param layout_config: optional dictionary for layout configuration, passed directly to plotly
        :type layout_config: dict or None
        """
        assert mode in ('stack', 'group', 'relative')

        plotly_dict = create_2d_histogram_plot(
            np_row_wise=histogram,
            title=title,
            xtitle=xtitle,
            ytitle=ytitle,
            labels=labels,
            series=series,
            xlabels=xlabels,
            comment=comment,
            mode=mode,
            layout_config=layout_config,
        )

        return self.report_plot(
            title=self._normalize_name(title),
            series=self._normalize_name(series),
            plot=plotly_dict,
            iter=iter,
            nan_as_null=False,
        )

    def report_table(self, title, series, table, iteration, layout_config=None):
        """
        Report a table plot.

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param table: The table data
        :type table: pandas.DataFrame
        :param iteration: Iteration number
        :type iteration: int
        :param layout_config: optional dictionary for layout configuration, passed directly to plotly
        :type layout_config: dict or None
        """
        table_output = create_plotly_table(table, title, series, layout_config=layout_config)
        return self.report_plot(
            title=self._normalize_name(title),
            series=self._normalize_name(series),
            plot=table_output,
            iter=iteration,
            round_digits=False,
            nan_as_null=False,
        )

    def report_line_plot(self, title, series, iter, xtitle, ytitle, mode='lines', reverse_xaxis=False,
                         comment=None, layout_config=None):
        """
        Report a (possibly multiple) line plot.

        :param title: Title (AKA metric)
        :type title: str
        :param series: All the series' data, one for each line in the plot.
        :type series: An iterable of LineSeriesInfo.
        :param iter: Iteration number
        :type iter: int
        :param xtitle: x-axis title
        :type xtitle: str
        :param ytitle: y-axis title
        :type ytitle: str
        :param mode: 'lines' / 'markers' / 'lines+markers'
        :type mode: str
        :param reverse_xaxis: If true X axis will be displayed from high to low (reversed)
        :type reverse_xaxis: bool
        :param comment: comment underneath the title
        :type comment: str
        :param layout_config: optional dictionary for layout configuration, passed directly to plotly
        :type layout_config: dict or None
        """

        plotly_dict = create_line_plot(
            title=title,
            series=series,
            xtitle=xtitle,
            ytitle=ytitle,
            mode=mode,
            reverse_xaxis=reverse_xaxis,
            comment=comment,
            layout_config=layout_config,
        )

        return self.report_plot(
            title=self._normalize_name(title),
            series='',
            plot=plotly_dict,
            iter=iter,
            nan_as_null=False,
        )

    def report_2d_scatter(self, title, series, data, iter, mode='lines', xtitle=None, ytitle=None, labels=None,
                          comment=None, layout_config=None):
        """
        Report a 2d scatter graph (with lines)

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param data: A scattered data: pairs of x,y as rows in a numpy array
        :type data: ndarray
        :param iter: Iteration number
        :type iter: int
        :param mode: (type str) 'lines'/'markers'/'lines+markers'
        :param xtitle: optional x-axis title
        :param ytitle: optional y-axis title
        :param labels: label (text) per point in the scatter (in the same order)
        :param comment: comment underneath the title
        :type comment: str
        :param layout_config: optional dictionary for layout configuration, passed directly to plotly
        :type layout_config: dict or None
        """
        plotly_dict = create_2d_scatter_series(
            np_row_wise=data,
            title=title,
            series_name=series,
            mode=mode,
            xtitle=xtitle,
            ytitle=ytitle,
            labels=labels,
            comment=comment,
            layout_config=layout_config,
        )

        return self.report_plot(
            title=self._normalize_name(title),
            series=self._normalize_name(series),
            plot=plotly_dict,
            iter=iter,
            nan_as_null=False,
        )

    def report_3d_scatter(self, title, series, data, iter, labels=None, mode='lines', color=((217, 217, 217, 0.14),),
                          marker_size=5, line_width=0.8, xtitle=None, ytitle=None, ztitle=None, fill=None,
                          comment=None, layout_config=None):
        """
        Report a 3d scatter graph (with markers)

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param data: A scattered data: pairs of x,y,z as rows in a numpy array. or list of numpy arrays
        :type data: ndarray.
        :param iter: Iteration number
        :type iter: int
        :param labels: label (text) per point in the scatter (in the same order)
        :type labels: list(str)
        :param mode: (type str) 'lines'/'markers'/'lines+markers'
        :param color: list of RGBA colors [(217, 217, 217, 0.14),]
        :param marker_size: marker size in px
        :param line_width: line width in px
        :param xtitle: optional x-axis title
        :param ytitle: optional y-axis title
        :param ztitle: optional z-axis title
        :param fill: optional
        :param comment: comment underneath the title
        :param layout_config: optional dictionary for layout configuration, passed directly to plotly
        :type layout_config: dict or None
        """
        data_series = data if isinstance(data, list) else [data]

        def get_labels(a_i):
            if labels and isinstance(labels, list):
                try:
                    item = labels[a_i]
                except IndexError:
                    item = labels[-1]
                if isinstance(item, list):
                    return item
            return labels

        plotly_obj = plotly_scatter3d_layout_dict(
            title=title,
            xaxis_title=xtitle,
            yaxis_title=ytitle,
            zaxis_title=ztitle,
            comment=comment,
            layout_config=layout_config,
        )

        for i, values in enumerate(data_series):
            plotly_obj = create_3d_scatter_series(
                np_row_wise=values,
                title=title,
                series_name=series[i] if isinstance(series, list) else None,
                labels=get_labels(i),
                plotly_obj=plotly_obj,
                mode=mode,
                line_width=line_width,
                marker_size=marker_size,
                color=color,
                fill_axis=fill,
            )

        return self.report_plot(
            title=self._normalize_name(title),
            series=self._normalize_name(series) if not isinstance(series, list) else None,
            plot=plotly_obj,
            iter=iter,
            nan_as_null=False,
        )

    def report_value_matrix(self, title, series, data, iter, xtitle=None, ytitle=None, xlabels=None, ylabels=None,
                            yaxis_reversed=False, comment=None, layout_config=None):
        """
        Report a heat-map matrix

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param data: A heat-map matrix (example: confusion matrix)
        :type data: ndarray
        :param iter: Iteration number
        :type iter: int
        :param str xtitle: optional x-axis title
        :param str ytitle: optional y-axis title
        :param xlabels: optional label per column of the matrix
        :param ylabels: optional label per row of the matrix
        :param bool yaxis_reversed: If False 0,0 is at the bottom left corner. If True 0,0 is at the Top left corner
        :param comment: comment underneath the title
        :param layout_config: optional dictionary for layout configuration, passed directly to plotly
        :type layout_config: dict or None
        """

        plotly_dict = create_value_matrix(
            np_value_matrix=data,
            title=title,
            xlabels=xlabels,
            ylabels=ylabels,
            series=series,
            comment=comment,
            xtitle=xtitle,
            ytitle=ytitle,
            yaxis_reversed=yaxis_reversed,
            layout_config=layout_config,
        )

        return self.report_plot(
            title=self._normalize_name(title),
            series=self._normalize_name(series),
            plot=plotly_dict,
            iter=iter,
            nan_as_null=False,
        )

    def report_value_surface(self, title, series, data, iter, xlabels=None, ylabels=None,
                             xtitle=None, ytitle=None, ztitle=None, camera=None, comment=None, layout_config=None):
        """
        Report a 3d surface (same data as heat-map matrix, only presented differently)

        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param data: A heat-map matrix (example: confusion matrix)
        :type data: ndarray
        :param iter: Iteration number
        :type iter: int
        :param xlabels: optional label per column of the matrix
        :param ylabels: optional label per row of the matrix
        :param xtitle: optional x-axis title
        :param ytitle: optional y-axis title
        :param ztitle: optional z-axis title
        :param camera: X,Y,Z camera position. def: (1,1,1)
        :param comment: comment underneath the title
        :param layout_config: optional dictionary for layout configuration, passed directly to plotly
        :type layout_config: dict or None
        """

        plotly_dict = create_3d_surface(
            np_value_matrix=data,
            title=title + '/' + series,
            xlabels=xlabels,
            ylabels=ylabels,
            series=series,
            xtitle=xtitle,
            ytitle=ytitle,
            ztitle=ztitle,
            camera=camera,
            comment=comment,
            layout_config=layout_config,
        )

        return self.report_plot(
            title=self._normalize_name(title),
            series=self._normalize_name(series),
            plot=plotly_dict,
            iter=iter,
            nan_as_null=False,
        )

    def report_image_plot_and_upload(self, title, series, iter, path=None, matrix=None,
                                     upload_uri=None, max_image_history=None, delete_after_upload=False):
        """
        Report an image as plot and upload its contents.
        Image is uploaded to a preconfigured bucket (see setup_upload()) with a key (filename)
        describing the task ID, title, series and iteration.
        Then a plotly object is created and registered, this plotly objects points to the uploaded image
        :param title: Title (AKA metric)
        :type title: str
        :param series: Series (AKA variant)
        :type series: str
        :param iter: Iteration number
        :type iter: int
        :param path: A path to an image file. Required unless matrix is provided.
        :type path: str
        :param matrix: A 3D numpy.ndarray object containing image data (RGB). Required unless filename is provided.
        :type matrix: np.ndarray
        :param upload_uri: upload image destination (str)
        :type upload_uri: str
        :param max_image_history: maximum number of image to store per metric/variant combination
        use negative value for unlimited. default is set in global configuration (default=5)
        :param delete_after_upload: if True, one the file was uploaded the local copy will be deleted
        :type delete_after_upload: boolean
        """
        if not upload_uri and not self._storage_uri:
            raise ValueError('Upload configuration is required (use setup_upload())')
        if len([x for x in (path, matrix) if x is not None]) != 1:
            raise ValueError('Expected only one of [filename, matrix]')
        kwargs = dict(metric=self._normalize_name(title), variant=self._normalize_name(series), iter=iter,
                      file_history_size=max_image_history)

        if matrix is not None:
            width = matrix.shape[1]  # noqa
            height = matrix.shape[0]  # noqa
        else:
            # noinspection PyBroadException
            try:
                from PIL import Image
                width, height = Image.open(path).size
            except Exception:
                width = 640
                height = 480

        ev = UploadEvent(image_data=matrix, upload_uri=upload_uri, local_image_path=path,
                         delete_after_upload=delete_after_upload, **kwargs)
        _, url = ev.get_target_full_upload_uri(upload_uri or self._storage_uri, self._metrics.storage_key_prefix)

        # Hack: if the url doesn't start with http/s then the plotly will not be able to show it,
        # then we put the link under images not plots
        if not url.startswith('http') and not self._offline_mode:
            return self.report_image_and_upload(title=title, series=series, iter=iter, path=path, image=matrix,
                                                upload_uri=upload_uri, max_image_history=max_image_history)

        self._report(ev)
        plotly_dict = create_image_plot(
            image_src=url,
            title=title + '/' + series,
            width=640,
            height=int(640 * float(height or 480) / float(width or 640)),
        )

        return self.report_plot(
            title=self._normalize_name(title),
            series=self._normalize_name(series),
            plot=plotly_dict,
            iter=iter,
            nan_as_null=False,
        )

    def report_console(self, message, level=logging.INFO):
        """
        Report a scalar value
        :param message: message (AKA metric)
        :type message: str
        :param level: log level (int or string, log level)
        :type level: int
        """
        ev = ConsoleEvent(
            message=message,
            level=level,
            worker=self.session.worker,
        )
        self._report(ev)

    @classmethod
    def matplotlib_force_report_non_interactive(cls, force):
        from clearml.binding.matplotlib_bind import PatchedMatplotlib
        PatchedMatplotlib.force_report_as_image(force=force)

    @classmethod
    def _normalize_name(cls, name):
        return name

    def __exit__(self, exc_type, exc_val, exc_tb):
        # don't flush in case an exception was raised
        if not exc_type:
            self.flush()
