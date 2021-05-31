import base64
import os
import sys
import threading
from collections import defaultdict
from functools import partial
from io import BytesIO
from mimetypes import guess_extension
from tempfile import mkstemp
from typing import Any, Union

import numpy as np
import six
from PIL import Image

from ...debugging.log import LoggerRoot
from ..frameworks import _patched_call, WeightsFileHandler, _Empty
from ..import_bind import PostImportHookPatching
from ...config import running_remotely
from ...model import InputModel, OutputModel, Framework

try:
    from google.protobuf.json_format import MessageToDict  # noqa
except ImportError:
    MessageToDict = None

try:
    from PIL import GifImagePlugin  # noqa
except ImportError:
    pass


class TensorflowBinding(object):
    @classmethod
    def update_current_task(cls, task, patch_reporting=True, patch_model_io=True):
        if not task:
            IsTensorboardInit.clear_tensorboard_used()

        EventTrainsWriter.update_current_task(task)

        if patch_reporting:
            PatchSummaryToEventTransformer.update_current_task(task)
            PatchTensorFlowEager.update_current_task(task)

        if patch_model_io:
            PatchKerasModelIO.update_current_task(task)
            PatchTensorflowModelIO.update_current_task(task)
            PatchTensorflow2ModelIO.update_current_task(task)


class IsTensorboardInit(object):
    _tensorboard_initialized = False

    @classmethod
    def tensorboard_used(cls):
        return cls._tensorboard_initialized

    @classmethod
    def set_tensorboard_used(cls):
        cls._tensorboard_initialized = True

    @classmethod
    def clear_tensorboard_used(cls):
        cls._tensorboard_initialized = False

    @staticmethod
    def _patched_tb__init__(original_init, self, *args, **kwargs):
        IsTensorboardInit._tensorboard_initialized = True
        return original_init(self, *args, **kwargs)


# noinspection PyProtectedMember
class WeightsGradientHistHelper(object):
    def __init__(self, logger, report_freq=100, histogram_update_freq_multiplier=10, histogram_granularity=50):
        self._logger = logger
        self.report_freq = report_freq
        self._histogram_granularity = histogram_granularity
        self._histogram_update_freq_multiplier = histogram_update_freq_multiplier
        self._histogram_update_call_counter = 0
        self._hist_report_cache = {}
        self._hist_x_granularity = 50

    @staticmethod
    def _sample_histograms(_hist_iters, _histogram_granularity):
        # re-sample history based on distribution of samples across time (steps)
        ratio = ((_hist_iters[-1] - _hist_iters[_histogram_granularity]) /
                 (_hist_iters[_histogram_granularity - 1] - _hist_iters[0])) if \
            _hist_iters.size > _histogram_granularity else 0.
        cur_idx_below = np.arange(0, min(_hist_iters.size, _histogram_granularity - 1))
        np.random.shuffle(cur_idx_below)
        cur_idx_below = cur_idx_below[:int(_histogram_granularity * (1.0 - ratio / (1 + ratio)) + 0.5)]
        if ratio > 0.0:
            cur_idx_above = np.arange(_histogram_granularity - 1, _hist_iters.size)
            np.random.shuffle(cur_idx_above)
            cur_idx_above = cur_idx_above[:int(_histogram_granularity * ratio / (1 + ratio))]
        else:
            cur_idx_above = np.array([])
        _cur_idx = np.unique(np.sort(np.concatenate((cur_idx_below, cur_idx_above)).astype(np.int)))
        return _cur_idx

    def add_histogram(self, title, series, step, hist_data):
        # only collect histogram every specific interval
        self._histogram_update_call_counter += 1
        if self._histogram_update_call_counter % self.report_freq != 0 or \
                self._histogram_update_call_counter < self.report_freq - 1:
            return None

        if isinstance(hist_data, dict):
            pass
        elif isinstance(hist_data, np.ndarray) and len(hist_data.shape) == 2 and np.atleast_2d(hist_data).shape[1] == 3:
            # prepare the dictionary, assume numpy
            # hist_data['bucketLimit'] is the histogram bucket right side limit, meaning X axis
            # hist_data['bucket'] is the histogram height, meaning the Y axis
            # notice hist_data[:, 1] is the right side limit, for backwards compatibility we take the left side
            hist_data = {'bucketLimit': hist_data[:, 0].tolist(), 'bucket': hist_data[:, 2].tolist()}
        else:
            # assume we have to do the histogram on the data
            hist_data = np.histogram(hist_data, bins=32)
            hist_data = {'bucketLimit': hist_data[1].tolist(), 'bucket': hist_data[0].tolist()}

        self._add_histogram(title=title, series=series, step=step, hist_data=hist_data)

    def _add_histogram(self, title, series, step, hist_data):
        # only collect histogram every specific interval
        self._histogram_update_call_counter += 1
        if self._histogram_update_call_counter % self.report_freq != 0 or \
                self._histogram_update_call_counter < self.report_freq - 1:
            return None

        # generate forward matrix of the histograms
        # Y-axis (rows) is iteration (from 0 to current Step)
        # X-axis averaged bins (conformed sample 'bucketLimit')
        # Z-axis actual value (interpolated 'bucket')
        step = EventTrainsWriter._fix_step_counter(title, series, step)

        # get histograms from cache
        hist_list, hist_iters, minmax = self._hist_report_cache.get((title, series), ([], np.array([]), None))

        # resample data so we are always constrained in number of histogram we keep
        if hist_iters.size >= self._histogram_granularity ** 2:
            idx = self._sample_histograms(hist_iters, self._histogram_granularity)
            hist_iters = hist_iters[idx]
            hist_list = [hist_list[i] for i in idx]

        # check if current sample is not already here (actually happens some times)
        if step in hist_iters:
            return None

        # add current sample, if not already here
        hist_iters = np.append(hist_iters, step)
        # hist_data['bucketLimit'] is the histogram bucket right side limit, meaning X axis
        # hist_data['bucket'] is the histogram height, meaning the Y axis
        hist = np.array(list(zip(hist_data['bucketLimit'], hist_data['bucket'])), dtype=np.float32)
        hist = hist[~np.isinf(hist[:, 0]), :]
        hist_list.append(hist)
        # keep track of min/max values of histograms (for later re-binning)
        if minmax is None:
            minmax = hist[:, 0].min(), hist[:, 0].max()
        else:
            # noinspection PyUnresolvedReferences
            minmax = min(minmax[0], hist[:, 0].min()), max(minmax[1], hist[:, 0].max())

        # update the cache
        self._hist_report_cache[(title, series)] = hist_list, hist_iters, minmax

        # only report histogram every specific interval, but do report the first few, so you know there are histograms
        if hist_iters.size < 1 or (hist_iters.size >= self._histogram_update_freq_multiplier and
                                   hist_iters.size % self._histogram_update_freq_multiplier != 0):
            return None

        # resample histograms on a unified bin axis +- epsilon
        _epsilon = abs((minmax[1] - minmax[0])/float(self._hist_x_granularity))
        if _epsilon == 0:
            _epsilon = 0.01
        _minmax = minmax[0] - _epsilon, minmax[1] + _epsilon
        prev_xedge = np.arange(start=_minmax[0],
                               step=(_minmax[1] - _minmax[0]) / float(self._hist_x_granularity - 2), stop=_minmax[1])
        # uniformly select histograms and the last one
        cur_idx = self._sample_histograms(hist_iters, self._histogram_granularity)
        report_hist = np.zeros(shape=(len(cur_idx), prev_xedge.size), dtype=np.float32)
        for i, n in enumerate(cur_idx):
            h = hist_list[n]
            report_hist[i, :] = np.interp(prev_xedge, h[:, 0], h[:, 1], right=0, left=0)
        yedges = hist_iters[cur_idx]
        xedges = prev_xedge

        # if only a single line make, add another zero line, for the scatter plot to draw
        if report_hist.shape[0] < 2:
            report_hist = np.vstack((np.zeros_like(report_hist), report_hist))

        # create 3d line (scatter) of histograms
        skipx = max(1, int(xedges.size / 10))
        skipy = max(1, int(yedges.size / 10))
        xlabels = ['%.2f' % v if i % skipx == 0 else '' for i, v in enumerate(xedges[:-1])]
        ylabels = [str(int(v)) if i % skipy == 0 else '' for i, v in enumerate(yedges)]
        self._logger.report_surface(
            title=title,
            series=series,
            iteration=0,
            xaxis=' ',
            yaxis='iteration',
            xlabels=xlabels,
            ylabels=ylabels,
            matrix=report_hist,
            camera=(-0.1, +1.3, 1.4))


# noinspection PyMethodMayBeStatic,PyProtectedMember,SpellCheckingInspection
class EventTrainsWriter(object):
    """
    TF SummaryWriter implementation that converts the tensorboard's summary into
    ClearML events and reports the events (metrics) for an ClearML task (logger).
    """
    __main_task = None
    _add_lock = threading.RLock()
    _series_name_lookup = {}

    # store all the created tensorboard writers in the system
    # this allows us to as weather a certain tile/series already exist on some EventWriter
    # and if it does, then we add to the series name the last token from the logdir
    # (so we can differentiate between the two)
    # key, value: key=hash(title, graph), value=EventTrainsWriter._id
    _title_series_writers_lookup = {}
    _event_writers_id_to_logdir = {}

    # Protect against step (iteration) reuse, for example,
    # steps counter inside an epoch, but wrapping around when epoch ends
    # i.e. step = 0..100 then epoch ends and again step = 0..100
    # We store the first report per title/series combination, and if wraparound occurs
    # we synthetically continue to increase the step/iteration based on the previous epoch counter
    # example: _title_series_wraparound_counter[('title', 'series')] =
    #           {'first_step':None, 'last_step':None, 'adjust_counter':0,}
    _title_series_wraparound_counter = {}

    @property
    def variants(self):
        return self._variants

    def prepare_report(self):
        return self.variants.copy()

    def tag_splitter(self, tag, num_split_parts, split_char='/', join_char='_', default_title='variant',
                     logdir_header='series', auto_reduce_num_split=False, force_add_prefix=None):
        """
        Split a tf.summary tag line to variant and metric.
        Variant is the first part of the split tag, metric is the second.
        :param str tag:
        :param int num_split_parts:
        :param str split_char: a character to split the tag on
        :param str join_char: a character to join the the splits
        :param str default_title: variant to use in case no variant can be inferred automatically
        :param str logdir_header: if 'series_last' then series=header: series, if 'series then series=series :header,
            if 'title_last' then title=header title, if 'title' then title=title header
        :param bool auto_reduce_num_split: if True and the tag is split for less parts then requested,
            then requested number of split parts is adjusted.
        :param str force_add_prefix: always add the prefix to the series name
        :return: (str, str) variant and metric
        """
        splitted_tag = tag.split(split_char)
        if auto_reduce_num_split and num_split_parts > len(splitted_tag) - 1:
            num_split_parts = max(1, len(splitted_tag) - 1)
        series = join_char.join(splitted_tag[-num_split_parts:])
        title = join_char.join(splitted_tag[:-num_split_parts]) or default_title

        if force_add_prefix:
            series = str(force_add_prefix)+series

        # check if we already decided that we need to change the title/series
        graph_id = hash((title, series))
        if graph_id in self._graph_name_lookup:
            return self._graph_name_lookup[graph_id]

        # check if someone other than us used this combination
        with self._add_lock:
            event_writer_id = self._title_series_writers_lookup.get(graph_id, None)
            if not event_writer_id:
                # put us there
                self._title_series_writers_lookup[graph_id] = self._id
            elif event_writer_id != self._id:
                # if there is someone else, change our series name and store us
                org_series = series
                org_title = title
                other_logdir = self._event_writers_id_to_logdir[event_writer_id]
                split_logddir = self._logdir.split('/')
                unique_logdir = set(split_logddir) - set(other_logdir.split('/'))
                header = '/'.join(s for s in split_logddir if s in unique_logdir)
                if logdir_header == 'series_last':
                    series = header + ': ' + series
                elif logdir_header == 'series':
                    series = series + ' :' + header
                elif logdir_header == 'title':
                    title = title + ' ' + header
                else:  # logdir_header == 'title_last':
                    title = header + ' ' + title
                graph_id = hash((title, series))
                # check if for some reason the new series is already occupied
                new_event_writer_id = self._title_series_writers_lookup.get(graph_id)
                if new_event_writer_id is not None and new_event_writer_id != self._id:
                    # well that's about it, nothing else we could do
                    if logdir_header == 'series_last':
                        series = str(self._logdir) + ': ' + org_series
                    elif logdir_header == 'series':
                        series = org_series + ' :' + str(self._logdir)
                    elif logdir_header == 'title':
                        title = org_title + ' ' + str(self._logdir)
                    else:  # logdir_header == 'title_last':
                        title = str(self._logdir) + ' ' + org_title
                    graph_id = hash((title, series))

                self._title_series_writers_lookup[graph_id] = self._id

        # store for next time
        self._graph_name_lookup[graph_id] = (title, series)
        return title, series

    def __init__(self, logger, logdir=None, report_freq=100, image_report_freq=None,
                 histogram_update_freq_multiplier=10, histogram_granularity=50, max_keep_images=None):
        """
        Create a compatible ClearML backend to the TensorFlow SummaryToEventTransformer
        Everything will be serialized directly to the ClearML backend, instead of to the standard TF FileWriter

        :param logger: The task.logger to use for sending the metrics (def: task.get_logger())
        :param report_freq: How often to update the statistics values
        :param image_report_freq: How often to upload images (step % image_update_freq == 0)
        :param histogram_update_freq_multiplier: How often to upload histogram
               (step//update_freq) % histogram_update_freq_multiplier == 0
        :param histogram_granularity: How many histograms (lines) to display in the 3d histogram plot
        :param max_keep_images: Maximum number of images to save before starting to reuse files (per title/metric pair)
        """
        # We are the events_writer, so that's what we'll pass
        IsTensorboardInit.set_tensorboard_used()
        self._logdir = logdir or ('unknown %d' % len(self._event_writers_id_to_logdir))
        # conform directory structure to unix
        if os.path.sep == '\\':
            self._logdir = self._logdir.replace('\\', '/')
        self._id = hash(self._logdir)
        self._event_writers_id_to_logdir[self._id] = self._logdir
        self.max_keep_images = max_keep_images
        self.report_freq = report_freq
        self.image_report_freq = image_report_freq if image_report_freq else report_freq
        self.histogram_granularity = histogram_granularity
        self.histogram_update_freq_multiplier = histogram_update_freq_multiplier
        self._histogram_update_call_counter = 0
        self._logger = logger
        self._visualization_mode = 'RGB'  # 'BGR'
        self._variants = defaultdict(lambda: ())
        self._scalar_report_cache = {}
        self._hist_report_cache = {}
        self._hist_x_granularity = 50
        self._max_step = 0
        self._graph_name_lookup = {}
        self._generic_tensor_type_name_lookup = {}
        self._grad_helper = WeightsGradientHistHelper(
            logger=logger,
            report_freq=report_freq,
            histogram_update_freq_multiplier=histogram_update_freq_multiplier,
            histogram_granularity=histogram_granularity
        )

    def _decode_image(self, img_str, width=None, height=None, color_channels=None):
        # noinspection PyBroadException
        try:
            if isinstance(img_str, bytes):
                imdata = img_str
            else:
                imdata = base64.b64decode(img_str)
            output = BytesIO(imdata)
            im = Image.open(output)
            # if this is a GIF store as is
            if getattr(im, 'is_animated'):
                output.close()
                fd, temp_file = mkstemp(
                    suffix=guess_extension(im.get_format_mimetype()) if hasattr(im, 'get_format_mimetype')
                    else ".{}".format(str(im.format).lower())
                )
                os.write(fd, imdata)
                os.close(fd)
                return temp_file

            image = np.asarray(im)
            output.close()
            if height is not None and height > 0 and width is not None and width > 0:
                # noinspection PyArgumentList
                val = image.reshape(height, width, -1).astype(np.uint8)
            else:
                val = image.astype(np.uint8)
            if val.ndim == 3 and val.shape[2] == 3:
                if self._visualization_mode == 'BGR':
                    val = val[:, :, [2, 1, 0]]
                else:
                    val = val
            elif (val.ndim == 2) or (val.ndim == 3 and val.shape[2] == 1):
                val = np.tile(np.atleast_3d(val), (1, 1, 3))
            elif val.ndim == 3 and val.shape[2] == 4:
                if self._visualization_mode == 'BGR':
                    val = val[:, :, [2, 1, 0]]
                else:
                    val = val[:, :, [0, 1, 2]]
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger = LoggerRoot.get_base_logger(TensorflowBinding)
            logger.warning('Failed decoding debug image [%s, %s, %s]' % (width, height, color_channels))
            logger.warning('Error: %s' % e)
            val = None
        return val

    def _add_image_numpy(self, tag, step, img_data_np, max_keep_images=None):
        # type: (str, int, Union[None, np.ndarray, str], int) -> ()
        # only report images every specific interval
        if step % self.image_report_freq != 0:
            return None

        if img_data_np is None:
            return

        # noinspection PyProtectedMember
        title, series = self.tag_splitter(tag, num_split_parts=3, default_title='Images', logdir_header='title',
                                          auto_reduce_num_split=True,
                                          force_add_prefix=self._logger._get_tensorboard_series_prefix())
        step = self._fix_step_counter(title, series, step)

        # check if this is a local temp file
        if isinstance(img_data_np, str):
            self._logger.report_image(
                title=title,
                series=series,
                iteration=step,
                local_path=img_data_np,
                delete_after_upload=True,
                max_image_history=self.max_keep_images if max_keep_images is None else max_keep_images,
            )
            return

        if img_data_np.dtype != np.uint8:
            # assume scale 0-1
            img_data_np = (img_data_np * 255).astype(np.uint8)

        # if 3d, pack into one big image
        if img_data_np.ndim == 4:
            dims = img_data_np.shape
            stack_dim = int(np.sqrt(dims[0]))
            # noinspection PyArgumentList
            res = img_data_np.reshape(stack_dim, stack_dim, *dims[1:]).transpose((0, 2, 1, 3, 4))
            tile_size_h = res.shape[0] * res.shape[1]
            tile_size_w = res.shape[2] * res.shape[3]
            img_data_np = res.reshape(tile_size_h, tile_size_w, -1)

        self._logger.report_image(
            title=title,
            series=series,
            iteration=step,
            image=img_data_np,
            max_image_history=self.max_keep_images if max_keep_images is None else max_keep_images,
        )

    def _add_image(self, tag, step, img_data):
        # only report images every specific interval
        if step % self.image_report_freq != 0:
            return None

        width = img_data.get('width')
        height = img_data.get('height')
        colorspace = img_data.get('colorspace')
        img_str = img_data['encodedImageString']
        matrix = self._decode_image(img_str, width=width, height=height, color_channels=colorspace)
        if matrix is None:
            return

        return self._add_image_numpy(tag=tag, step=step, img_data_np=matrix)

    def _add_scalar(self, tag, step, scalar_data):
        default_title = tag if not self._logger._get_tensorboard_auto_group_scalars() else 'Scalars'
        series_per_graph = self._logger._get_tensorboard_single_series_per_graph()

        # noinspection PyProtectedMember
        title, series = self.tag_splitter(
            tag, num_split_parts=1, default_title=default_title,
            logdir_header='title' if series_per_graph else 'series_last',
            force_add_prefix=self._logger._get_tensorboard_series_prefix()
        )

        step = self._fix_step_counter(title, series, step)
        tag = self._get_add_scalars_event_tag(default_title)

        possible_title = tag if series_per_graph else None
        possible_tag = None if series_per_graph else tag

        title = title + possible_title if possible_title else title
        series = possible_tag or series
        # update scalar cache
        num, value = self._scalar_report_cache.get((title, series), (0, 0))
        # nan outputs is a string, it's probably a NaN
        if isinstance(scalar_data, six.string_types):
            # noinspection PyBroadException
            try:
                scalar_data = float(scalar_data)
            except Exception:
                scalar_data = float('nan')
        # nan outputs nan
        self._scalar_report_cache[(title, series)] = \
            (num + 1,
             (value + scalar_data) if scalar_data == scalar_data else scalar_data)

        # only report images every specific interval
        if step % self.report_freq != 0:
            return None

        # calculate mean and zero cache
        num, value = self._scalar_report_cache.get((title, series), (0, 0))
        scalar_data = value / num
        self._scalar_report_cache[(title, series)] = (0, 0)

        self._logger.report_scalar(
            title=title,
            series=series,
            iteration=step,
            value=scalar_data,
        )

    def _add_histogram(self, tag, step, hist_data):
        # noinspection PyProtectedMember
        title, series = self.tag_splitter(tag, num_split_parts=1, default_title='Histograms',
                                          logdir_header='series',
                                          force_add_prefix=self._logger._get_tensorboard_series_prefix())

        self._grad_helper.add_histogram(
            title=title,
            series=series,
            step=step,
            hist_data=hist_data
        )

    def _add_plot(self, tag, step, values, vdict):
        # noinspection PyBroadException
        try:
            if values.get('floatVal'):
                plot_values = np.array(values.get('floatVal'), dtype=np.float32)
            else:
                plot_values = np.frombuffer(base64.b64decode(values['tensorContent'].encode('utf-8')),
                                            dtype=np.float32)
            plot_values = plot_values.reshape((int(values['tensorShape']['dim'][0]['size']),
                                               int(values['tensorShape']['dim'][1]['size'])))
            if 'metadata' in vdict:
                if tag not in self._series_name_lookup:
                    self._series_name_lookup[tag] = [(tag, vdict['metadata'].get('displayName', ''),
                                                      vdict['metadata']['pluginData']['pluginName'])]
                else:
                    # this should not happen, maybe it's another run, let increase the value
                    self._series_name_lookup[tag] += [(tag + '_%d' % (len(self._series_name_lookup[tag]) + 1),
                                                       vdict['metadata'].get('displayName', ''),
                                                       vdict['metadata']['pluginData']['pluginName'])]

            tag, series, plugin_name = self._series_name_lookup.get(tag, [(tag, tag, '')])[-1]

            if 'pr_curve' in plugin_name:
                # our thresholds are evenly distributed, in that
                #   width = 1.0 / (num_thresholds - 1)
                #   thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]
                num_thresholds = plot_values.shape[1]
                width = 1.0 / num_thresholds
                thresholds = np.arange(0.0, 1.0, width, dtype=plot_values.dtype)
                data_points = ['Threshold ', 'TP     ', 'FP     ', 'TN     ', 'FN     ', 'Precision ', ' Recall']
                series = [{'name': series, 'data': np.vstack((plot_values[-1], plot_values[-2])).T,
                           'labels': [''.join(data_points) + '<br> {:.3f}  '.format(thresholds[j]) +
                                      '  '.join(['%-3.2f' % v for v in plot_values[:, j]]) for j in
                                      range(num_thresholds)]}]
                reverse_xaxis = False
            else:
                reverse_xaxis = False
                series = [{'name': series, 'data': plot_values}]
            self._logger.report_line_plot(title=tag, series=series, xaxis='', yaxis='',
                                          iteration=step, reverse_xaxis=reverse_xaxis)
        except Exception:
            pass

    def _add_audio(self, tag, step, values, audio_data=None):
        # only report images every specific interval
        if step % self.image_report_freq != 0:
            return None

        if values:
            audio_str = values['encodedAudioString']
            audio_data = base64.b64decode(audio_str)
        if audio_data is None:
            return

        # noinspection PyProtectedMember
        title, series = self.tag_splitter(tag, num_split_parts=3, default_title='Audio', logdir_header='title',
                                          auto_reduce_num_split=True,
                                          force_add_prefix=self._logger._get_tensorboard_series_prefix())
        step = self._fix_step_counter(title, series, step)

        stream = BytesIO(audio_data)
        if values:
            file_extension = guess_extension(values['contentType']) or \
                '.{}'.format(values['contentType'].split('/')[-1])
        else:
            # assume wav as default
            file_extension = '.wav'
        self._logger.report_media(
            title=title,
            series=series,
            iteration=step,
            stream=stream,
            file_extension=file_extension,
            max_history=self.max_keep_images,
        )

    def _add_text(self, tag, step, tensor_bytes):
        # noinspection PyProtectedMember
        title, series = self.tag_splitter(tag, num_split_parts=3, default_title='Text', logdir_header='title',
                                          auto_reduce_num_split=True,
                                          force_add_prefix=self._logger._get_tensorboard_series_prefix())
        step = self._fix_step_counter(title, series, step)

        text = tensor_bytes.decode('utf-8', errors='replace')
        self._logger.report_media(
            title=title,
            series=series,
            iteration=step,
            stream=six.StringIO(text),
            file_extension='.txt',
            max_history=self.max_keep_images,
        )

    @staticmethod
    def _fix_step_counter(title, series, step):
        key = (title, series)
        if key not in EventTrainsWriter._title_series_wraparound_counter:
            EventTrainsWriter._title_series_wraparound_counter[key] = {'first_step': step, 'last_step': step,
                                                                       'adjust_counter': 0}
            return step
        wraparound_counter = EventTrainsWriter._title_series_wraparound_counter[key]
        # we decide on wrap around if the current step is less than 10% of the previous step
        # notice since counter is int and we want to avoid rounding error, we have double check in the if
        if step < wraparound_counter['last_step'] and step < 0.9 * wraparound_counter['last_step']:
            # adjust step base line
            wraparound_counter['adjust_counter'] += wraparound_counter['last_step'] + (1 if step <= 0 else step)

        # return adjusted step
        wraparound_counter['last_step'] = step
        return step + wraparound_counter['adjust_counter']

    def add_event(self, event, step=None, walltime=None, **_):
        supported_metrics = {
            'simpleValue', 'image', 'histo', 'tensor', 'audio'
        }

        def get_data(value_dict, metric_search_order):
            data = None
            metric_type = 'Unsupported'
            for variant in metric_search_order:
                data = value_dict.get(variant)
                if data is not None:
                    metric_type = variant
                    break
            return metric_type, data

        # Support multiple threads accessing this instance (i.e. let TF/Keras do what they need)
        with self._add_lock:
            # TODO: add report frequency threshold (i.e. if we are sending too much data, increase the report_freq)
            # we should measure reports per second and throttle back the reporting details accordingly
            msg_dict = MessageToDict(event)
            summary = msg_dict.get('summary')
            if summary is None:
                msg_dict.pop('step', None)
                msg_dict.pop('wallTime', None)
                keys_list = [key for key in msg_dict.keys() if len(key) > 0]
                keys_list = ', '.join(keys_list)
                LoggerRoot.get_base_logger(TensorflowBinding).debug(
                    'event summary not found, message type unsupported: %s' % keys_list)
                return
            value_dicts = summary.get('value')
            # noinspection PyUnusedLocal
            walltime = walltime or msg_dict.get('step')
            step = step or msg_dict.get('step')
            if step is None:
                # when we start a new epoch there is no step in the msg_dict,
                # we have to extract it manually
                if hasattr(event, 'step'):
                    step = int(event.step)
                else:
                    step = 0
                    LoggerRoot.get_base_logger(TensorflowBinding).debug(
                        'Received event without step, assuming step = {}'.format(step))
            else:
                step = int(step)
            self._max_step = max(self._max_step, step)
            if value_dicts is None:
                LoggerRoot.get_base_logger(TensorflowBinding).debug("Summary arrived without 'value'")
                return

            for vdict in value_dicts:
                tag = vdict.pop('tag', None)
                if tag is None:
                    # we should not get here
                    LoggerRoot.get_base_logger(TensorflowBinding).debug(
                        'No tag for \'value\' existing keys %s' % ', '.join(vdict.keys()))
                    continue
                metric, values = get_data(vdict, supported_metrics)
                if metric == 'simpleValue':
                    self._add_scalar(tag=tag, step=step, scalar_data=values)
                elif metric == 'histo':
                    self._add_histogram(tag=tag, step=step, hist_data=values)
                elif metric == 'image':
                    self._add_image(tag=tag, step=step, img_data=values)
                elif metric == 'audio':
                    self._add_audio(tag, step, values)
                elif metric == 'tensor' and values.get('dtype') == 'DT_STRING':
                    # generic tensor
                    tensor_bytes = base64.b64decode('\n'.join(values['stringVal']))
                    plugin_type = self._generic_tensor_type_name_lookup.get(tag) or \
                        vdict.get('metadata', {}).get('pluginData', {}).get('pluginName', '').lower()
                    if plugin_type == 'audio':
                        self._generic_tensor_type_name_lookup[tag] = plugin_type
                        self._add_audio(tag, step, None, tensor_bytes)
                    elif plugin_type == 'text':
                        self._generic_tensor_type_name_lookup[tag] = plugin_type
                        self._add_text(tag, step, tensor_bytes)
                    else:
                        # we do not support it
                        pass
                elif metric == 'tensor' and values.get('dtype') == 'DT_FLOAT':
                    self._add_plot(tag, step, values, vdict)
                else:
                    LoggerRoot.get_base_logger(TensorflowBinding).debug(
                        'Event unsupported. tag = %s, vdict keys [%s]' % (tag, ', '.join(vdict.keys())))
                    continue

    def get_logdir(self):
        """ Returns a temporary directory name for compatibility with FileWriter. This directory is not actually used.
        :return: '.'
        """
        return '.'

    def flush(self):
        """Flushes the event file to disk.

        Call this method to make sure that all pending events have been written to
        disk.
        """
        self._logger.flush()

    def close(self):
        """Flushes the event file to disk and close the file.

        Call this method when you do not need the summary writer anymore.
        """
        self._logger.flush()

    def reopen(self):
        """Reopens the EventFileWriter.

        Can be called after `close` to add more events in the same directory.
        The events will go into a new events file.

        Does nothing if the EventFileWriter was not closed.
        """
        pass

    def _get_add_scalars_event_tag(self, title_prefix):
        """

        :param str title_prefix: the table title prefix that was added to the series.
        :return: str same as tensorboard use
        """
        # HACK - this is tensorboard Summary util function, original path:
        # ~/torch/utils/tensorboard/summary.py
        def _clean_tag(name):
            import re as _re
            # noinspection RegExpRedundantEscape
            _INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')
            if name is not None:
                new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
                new_name = new_name.lstrip('/')  # Remove leading slashes
                if new_name != name:
                    LoggerRoot.get_base_logger(TensorflowBinding).debug(
                        'Summary name %s is illegal; using %s instead.' % (name, new_name))
                    name = new_name
            return name

        main_path = self._logdir
        # noinspection PyBroadException
        try:
            main_path = _clean_tag(main_path)
            origin_tag = main_path.rpartition("/")[2].replace(title_prefix, "", 1)
            if title_prefix and origin_tag[0] == "_":  # add_scalars tag
                origin_tag = origin_tag[1:]  # Remove the first "_" that was added by the main_tag in tensorboard
            else:
                return ""
        except Exception:
            origin_tag = ""
        return origin_tag

    @classmethod
    def update_current_task(cls, task):
        if cls.__main_task != task:
            with cls._add_lock:
                cls._series_name_lookup = {}
                cls._title_series_writers_lookup = {}
                cls._event_writers_id_to_logdir = {}
                cls._title_series_wraparound_counter = {}
        cls.__main_task = task


# noinspection PyCallingNonCallable
class ProxyEventsWriter(object):
    def __init__(self, events):
        IsTensorboardInit.set_tensorboard_used()
        self._events = events

    def _get_sentinel_event(self):
        ret = None
        for ev in self._events:
            if hasattr(ev, '_get_sentinel_event'):
                # noinspection PyProtectedMember
                ret = ev._get_sentinel_event()
        return ret

    def get_logdir(self):
        ret = None
        for ev in self._events:
            if hasattr(ev, 'get_logdir'):
                ret = ev.get_logdir()
        return ret

    def reopen(self):
        ret = None
        for ev in self._events:
            if hasattr(ev, 'reopen'):
                ret = ev.reopen()
        return ret

    def add_event(self, *args, **kwargs):
        ret = None
        for ev in self._events:
            if hasattr(ev, 'add_event'):
                ret = ev.add_event(*args, **kwargs)
        return ret

    def flush(self):
        ret = None
        for ev in self._events:
            if hasattr(ev, 'flush'):
                ret = ev.flush()
        return ret

    def close(self):
        ret = None
        for ev in self._events:
            if hasattr(ev, 'close'):
                ret = ev.close()
        return ret


# noinspection PyPep8Naming
class PatchSummaryToEventTransformer(object):
    __main_task = None
    __original_getattribute = None
    __original_getattributeX = None
    _original_add_event = None
    _original_add_eventT = None
    _original_add_eventX = None
    defaults_dict = dict(
        report_freq=1, image_report_freq=1, histogram_update_freq_multiplier=5,
        histogram_granularity=50)

    @staticmethod
    def trains_object(self):
        if isinstance(self.event_writer, ProxyEventsWriter):
            # noinspection PyProtectedMember
            trains_writer = [e for e in self.event_writer._events if isinstance(e, EventTrainsWriter)]
            return trains_writer[0] if trains_writer else None
        elif isinstance(self.event_writer, EventTrainsWriter):
            return self.event_writer
        if not self.__dict__.get('_trains_defaults'):
            self.__dict__['_trains_defaults'] = {}
        return self.__dict__['_trains_defaults']

    @staticmethod
    def update_current_task(task, **kwargs):
        PatchSummaryToEventTransformer.defaults_dict.update(kwargs)
        PatchSummaryToEventTransformer.__main_task = task
        # make sure we patched the SummaryToEventTransformer
        PatchSummaryToEventTransformer._patch_summary_to_event_transformer()
        PostImportHookPatching.add_on_import('tensorflow',
                                             PatchSummaryToEventTransformer._patch_summary_to_event_transformer)
        PostImportHookPatching.add_on_import('torch',
                                             PatchSummaryToEventTransformer._patch_summary_to_event_transformer)
        PostImportHookPatching.add_on_import('tensorboardX',
                                             PatchSummaryToEventTransformer._patch_summary_to_event_transformer)

    @staticmethod
    def _patch_summary_to_event_transformer():
        if 'tensorflow' in sys.modules:
            try:
                from tensorflow.python.summary.writer.writer import SummaryToEventTransformer  # noqa
                # only patch once
                if PatchSummaryToEventTransformer.__original_getattribute is None:
                    PatchSummaryToEventTransformer.__original_getattribute = SummaryToEventTransformer.__getattribute__
                    SummaryToEventTransformer.__getattribute__ = PatchSummaryToEventTransformer._patched_getattribute
                    setattr(SummaryToEventTransformer, 'clearml',
                            property(PatchSummaryToEventTransformer.trains_object))
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).debug(str(ex))

        if 'torch' in sys.modules:
            try:
                # only patch once
                if PatchSummaryToEventTransformer._original_add_eventT is None:
                    # noinspection PyUnresolvedReferences
                    from torch.utils.tensorboard.writer import FileWriter as FileWriterT  # noqa
                    PatchSummaryToEventTransformer._original_add_eventT = FileWriterT.add_event
                    FileWriterT.add_event = PatchSummaryToEventTransformer._patched_add_eventT
                    setattr(FileWriterT, 'clearml', None)
            except ImportError:
                # this is a new version of TensorflowX
                pass
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).debug(str(ex))

        if 'tensorboardX' in sys.modules:
            try:
                # only patch once
                if PatchSummaryToEventTransformer.__original_getattributeX is None:
                    # noinspection PyUnresolvedReferences
                    from tensorboardX.writer import SummaryToEventTransformer as SummaryToEventTransformerX  # noqa
                    PatchSummaryToEventTransformer.__original_getattributeX = \
                        SummaryToEventTransformerX.__getattribute__
                    SummaryToEventTransformerX.__getattribute__ = PatchSummaryToEventTransformer._patched_getattributeX
                    setattr(SummaryToEventTransformerX, 'clearml',
                            property(PatchSummaryToEventTransformer.trains_object))
            except ImportError:
                # this is a new version of TensorflowX
                pass
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).debug(str(ex))

            if PatchSummaryToEventTransformer.__original_getattributeX is None:
                try:
                    # only patch once
                    if PatchSummaryToEventTransformer._original_add_eventX is None:
                        from tensorboardX.writer import FileWriter as FileWriterX  # noqa
                        PatchSummaryToEventTransformer._original_add_eventX = FileWriterX.add_event
                        FileWriterX.add_event = PatchSummaryToEventTransformer._patched_add_eventX
                        setattr(FileWriterX, 'clearml', None)
                except ImportError:
                    # this is a new version of TensorflowX
                    pass
                except Exception as ex:
                    LoggerRoot.get_base_logger(TensorflowBinding).debug(str(ex))

    @staticmethod
    def _patched_add_eventT(self, *args, **kwargs):
        if not hasattr(self, 'clearml') or not PatchSummaryToEventTransformer.__main_task:
            return PatchSummaryToEventTransformer._original_add_eventT(self, *args, **kwargs)
        if not self.clearml:  # noqa
            # noinspection PyBroadException
            try:
                logdir = self.get_logdir()
            except Exception:
                logdir = None
            self.clearml = EventTrainsWriter(PatchSummaryToEventTransformer.__main_task.get_logger(),
                                             logdir=logdir, **PatchSummaryToEventTransformer.defaults_dict)
        # noinspection PyBroadException
        try:
            self.clearml.add_event(*args, **kwargs)
        except Exception:
            pass
        return PatchSummaryToEventTransformer._original_add_eventT(self, *args, **kwargs)

    @staticmethod
    def _patched_add_eventX(self, *args, **kwargs):
        if not hasattr(self, 'clearml') or not PatchSummaryToEventTransformer.__main_task:
            return PatchSummaryToEventTransformer._original_add_eventX(self, *args, **kwargs)
        if not self.clearml:
            # noinspection PyBroadException
            try:
                logdir = self.get_logdir()
            except Exception:
                logdir = None
            self.clearml = EventTrainsWriter(PatchSummaryToEventTransformer.__main_task.get_logger(),
                                             logdir=logdir, **PatchSummaryToEventTransformer.defaults_dict)
        # noinspection PyBroadException
        try:
            self.clearml.add_event(*args, **kwargs)
        except Exception:
            pass
        return PatchSummaryToEventTransformer._original_add_eventX(self, *args, **kwargs)

    @staticmethod
    def _patched_getattribute(self, attr):
        get_base = PatchSummaryToEventTransformer.__original_getattribute
        return PatchSummaryToEventTransformer._patched_getattribute_(self, attr, get_base)

    @staticmethod
    def _patched_getattributeX(self, attr):
        get_base = PatchSummaryToEventTransformer.__original_getattributeX
        return PatchSummaryToEventTransformer._patched_getattribute_(self, attr, get_base)

    @staticmethod
    def _patched_getattribute_(self, attr, get_base):
        # no main task, zero chance we have an ClearML event logger
        if PatchSummaryToEventTransformer.__main_task is None:
            return get_base(self, attr)

        # check if we already have an ClearML event logger
        __dict__ = get_base(self, '__dict__')
        if 'event_writer' not in __dict__ or \
                isinstance(__dict__['event_writer'], (ProxyEventsWriter, EventTrainsWriter)):
            return get_base(self, attr)

        # patch the events writer field, and add a double Event Logger (ClearML and original)
        base_eventwriter = __dict__['event_writer']
        # noinspection PyBroadException
        try:
            logdir = base_eventwriter.get_logdir()
        except Exception:
            logdir = None
        defaults_dict = __dict__.get('_trains_defaults') or PatchSummaryToEventTransformer.defaults_dict
        trains_event = EventTrainsWriter(PatchSummaryToEventTransformer.__main_task.get_logger(),
                                         logdir=logdir, **defaults_dict)

        # order is important, the return value of ProxyEventsWriter is the last object in the list
        __dict__['event_writer'] = ProxyEventsWriter([trains_event, base_eventwriter])
        return get_base(self, attr)


class _ModelAdapter(object):
    """ Model adapter which extends the save and save_weights methods of a Keras Model instance """
    _model = None  # type: Any
    _output_model = None  # type: OutputModel

    def __init__(self, model, output_model):
        super(_ModelAdapter, self).__init__()
        super(_ModelAdapter, self).__setattr__('_model', model)
        super(_ModelAdapter, self).__setattr__('_output_model', output_model)
        super(_ModelAdapter, self).__setattr__('_logger', LoggerRoot.get_base_logger(TensorflowBinding))

    def __getattr__(self, attr):
        return getattr(self._model, attr)

    def __setattr__(self, key, value):
        return setattr(self._model, key, value)

    def save(self, filepath, overwrite=True, include_optimizer=True):
        self._model.save(filepath=filepath, overwrite=overwrite, include_optimizer=include_optimizer)
        # TODO: auto generate new objects of filename changes
        try:
            self._output_model.update_weights(weights_filename=filepath, auto_delete_file=True)
        except Exception as ex:
            self._logger.error(str(ex))

    def save_weights(self, filepath, overwrite=True):
        self._model.save_weights(filepath=filepath, overwrite=overwrite)
        # TODO: auto generate new objects of filename changes
        try:
            self._output_model.update_weights(weights_filename=filepath, auto_delete_file=True)
        except Exception as ex:
            self._logger.error(str(ex))


class PatchModelCheckPointCallback(object):
    __main_task = None
    __original_getattribute = None
    defaults_dict = dict(
        config_text=None,
        config_dict=None,
        label_enumeration=None,
        name=None,
        comment=None)

    @staticmethod
    def trains_object(self):
        if isinstance(self.model, _ModelAdapter):
            # noinspection PyProtectedMember
            return self.model._output_model
        if not self.__dict__.get('_trains_defaults'):
            self.__dict__['_trains_defaults'] = {}
        return self.__dict__['_trains_defaults']

    @staticmethod
    def update_current_task(task, **kwargs):
        PatchModelCheckPointCallback.defaults_dict.update(kwargs)
        PatchModelCheckPointCallback.__main_task = task
        # make sure we patched the SummaryToEventTransformer
        PatchModelCheckPointCallback._patch_model_checkpoint()
        PostImportHookPatching.add_on_import('keras', PatchModelCheckPointCallback._patch_model_checkpoint)
        PostImportHookPatching.add_on_import('tensorflow', PatchModelCheckPointCallback._patch_model_checkpoint)

    @staticmethod
    def _patch_model_checkpoint():
        is_keras = 'keras' in sys.modules
        is_tf_keras = 'tensorflow' in sys.modules
        callbacks = None
        if is_keras:
            try:
                import keras.callbacks as callbacks  # noqa
            except ImportError:
                is_keras = False
        if not is_keras and is_tf_keras:
            try:
                # hack: make sure tensorflow.__init__ is called
                import tensorflow  # noqa
                import tensorflow.python.keras.callbacks as callbacks  # noqa
            except ImportError:
                is_tf_keras = False
                callbacks = None
        # we have nothing, quit
        if not is_keras and not is_tf_keras:
            return

        try:
            # only patch once
            if PatchModelCheckPointCallback.__original_getattribute is None and callbacks is not None:
                PatchModelCheckPointCallback.__original_getattribute = callbacks.ModelCheckpoint.__getattribute__
                callbacks.ModelCheckpoint.__getattribute__ = PatchModelCheckPointCallback._patched_getattribute
                setattr(callbacks.ModelCheckpoint, 'clearml',
                        property(PatchModelCheckPointCallback.trains_object))

        except Exception as ex:
            LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))

    @staticmethod
    def _patched_getattribute(self, attr):
        get_base = PatchModelCheckPointCallback.__original_getattribute

        # no main task, zero chance we have an ClearML event logger
        if PatchModelCheckPointCallback.__main_task is None:
            return get_base(self, attr)

        # check if we already have an ClearML event logger
        __dict__ = get_base(self, '__dict__')
        if 'model' not in __dict__ or \
                isinstance(__dict__['model'], _ModelAdapter):
            return get_base(self, attr)

        # patch the events writer field, and add a double Event Logger (ClearML and original)
        base_model = __dict__['model']
        defaults_dict = __dict__.get('_trains_defaults') or PatchModelCheckPointCallback.defaults_dict
        output_model = OutputModel(
            PatchModelCheckPointCallback.__main_task,
            config_text=defaults_dict.get('config_text'),
            config_dict=defaults_dict.get('config_dict'),
            name=defaults_dict.get('name'),
            comment=defaults_dict.get('comment'),
            label_enumeration=defaults_dict.get('label_enumeration') or
            PatchModelCheckPointCallback.__main_task.get_labels_enumeration(),
            framework=Framework.keras,
        )
        output_model.set_upload_destination(
            PatchModelCheckPointCallback.__main_task.get_output_destination(raise_on_error=False))
        trains_model = _ModelAdapter(base_model, output_model)

        # order is important, the return value of ProxyEventsWriter is the last object in the list
        __dict__['model'] = trains_model
        return get_base(self, attr)


# noinspection PyProtectedMember,PyUnresolvedReferences
class PatchTensorFlowEager(object):
    __main_task = None
    __original_fn_scalar = None
    __original_fn_hist = None
    __original_fn_image = None
    __trains_event_writer = {}
    defaults_dict = dict(
        report_freq=1, image_report_freq=1, histogram_update_freq_multiplier=5,
        histogram_granularity=50)

    @staticmethod
    def update_current_task(task, **kwargs):
        if task != PatchTensorFlowEager.__main_task:
            PatchTensorFlowEager.__trains_event_writer = {}

        PatchTensorFlowEager.defaults_dict.update(kwargs)
        PatchTensorFlowEager.__main_task = task
        # make sure we patched the SummaryToEventTransformer
        PatchTensorFlowEager._patch_summary_ops()
        PostImportHookPatching.add_on_import('tensorflow', PatchTensorFlowEager._patch_summary_ops)

    @staticmethod
    def _patch_summary_ops():
        if PatchTensorFlowEager.__original_fn_scalar is not None:
            return
        if 'tensorflow' in sys.modules:
            try:
                # hack: make sure tensorflow.__init__ is called
                import tensorflow  # noqa
                from tensorflow.python.ops import gen_summary_ops  # noqa
                PatchTensorFlowEager.__original_fn_scalar = gen_summary_ops.write_scalar_summary
                gen_summary_ops.write_scalar_summary = PatchTensorFlowEager._write_scalar_summary
                PatchTensorFlowEager.__original_fn_image = gen_summary_ops.write_image_summary
                gen_summary_ops.write_image_summary = PatchTensorFlowEager._write_image_summary
                PatchTensorFlowEager.__original_fn_hist = gen_summary_ops.write_histogram_summary
                gen_summary_ops.write_histogram_summary = PatchTensorFlowEager._write_hist_summary
                PatchTensorFlowEager.__write_summary = gen_summary_ops.write_summary
                gen_summary_ops.write_summary = PatchTensorFlowEager._write_summary
                gen_summary_ops.create_summary_file_writer = partial(IsTensorboardInit._patched_tb__init__,
                                                                     gen_summary_ops.create_summary_file_writer)
                gen_summary_ops.create_summary_db_writer = partial(IsTensorboardInit._patched_tb__init__,
                                                                   gen_summary_ops.create_summary_db_writer)
            except ImportError:
                pass
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).debug(str(ex))

    @staticmethod
    def _get_event_writer(writer):
        if not PatchTensorFlowEager.__main_task:
            return None
        if not PatchTensorFlowEager.__trains_event_writer.get(id(writer)):
            # noinspection PyBroadException
            try:
                logdir = writer.get_logdir()
            except Exception:
                # check if we are in eager mode, let's get the global context lopdir
                # noinspection PyBroadException
                try:
                    from tensorflow.python.eager import context  # noqa
                    logdir = context.context().summary_writer._init_op_fn.keywords.get('logdir')
                except Exception:
                    # noinspection PyBroadException
                    try:
                        from tensorflow.python.ops.summary_ops_v2 import _summary_state  # noqa
                        logdir = _summary_state.writer._init_op_fn.keywords.get('logdir')
                    except Exception:
                        logdir = None
                # noinspection PyBroadException
                try:
                    if logdir is not None:
                        logdir = logdir.numpy().decode()
                except Exception:
                    logdir = None

            PatchTensorFlowEager.__trains_event_writer[id(writer)] = EventTrainsWriter(
                logger=PatchTensorFlowEager.__main_task.get_logger(), logdir=logdir,
                **PatchTensorFlowEager.defaults_dict)
        return PatchTensorFlowEager.__trains_event_writer[id(writer)]

    @staticmethod
    def trains_object(self):
        if not PatchTensorFlowEager.__trains_event_writer:
            return None
        return PatchTensorFlowEager.__trains_event_writer.get(
            id(self), list(PatchTensorFlowEager.__trains_event_writer.values())[0])

    @staticmethod
    def _write_summary(writer, step, tensor, tag, summary_metadata, name=None, **kwargs):
        event_writer = PatchTensorFlowEager._get_event_writer(writer)
        # make sure we can get the tensors values
        if event_writer and isinstance(step, int) or hasattr(step, 'numpy'):
            # noinspection PyBroadException
            try:
                plugin_type = summary_metadata.decode()
                if plugin_type.endswith('scalars'):
                    event_writer._add_scalar(tag=str(tag),
                                             step=int(step.numpy()) if not isinstance(step, int) else step,
                                             scalar_data=tensor.numpy())
                elif plugin_type.endswith('images'):
                    img_data_np = tensor.numpy()
                    PatchTensorFlowEager._add_image_event_helper(event_writer, img_data_np=img_data_np,
                                                                 tag=tag, step=step, **kwargs)
                elif plugin_type.endswith('histograms'):
                    event_writer._add_histogram(
                        tag=str(tag), step=int(step.numpy()) if not isinstance(step, int) else step,
                        hist_data=tensor.numpy()
                    )
                elif plugin_type.endswith('text'):
                    event_writer._add_text(
                        tag=str(tag), step=int(step.numpy()) if not isinstance(step, int) else step,
                        tensor_bytes=tensor.numpy()
                    )
                elif 'audio' in plugin_type:
                    audio_bytes_list = [a for a in tensor.numpy().flatten() if a]
                    for i, audio_bytes in enumerate(audio_bytes_list):
                        event_writer._add_audio(tag=str(tag) + ('/{}'.format(i) if len(audio_bytes_list) > 1 else ''),
                                                step=int(step.numpy()) if not isinstance(step, int) else step,
                                                values=None, audio_data=audio_bytes)
                else:
                    pass  # print('unsupported plugin_type', plugin_type)
            except Exception:
                pass
        return PatchTensorFlowEager.__write_summary(writer, step, tensor, tag, summary_metadata, name, **kwargs)

    @staticmethod
    def _write_scalar_summary(writer, step, tag, value, name=None, **kwargs):
        event_writer = PatchTensorFlowEager._get_event_writer(writer)
        if event_writer and isinstance(step, int) or hasattr(step, 'numpy'):
            try:
                event_writer._add_scalar(tag=str(tag),
                                         step=int(step.numpy()) if not isinstance(step, int) else step,
                                         scalar_data=value.numpy())
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))
        elif event_writer:
            def _report_summary_op(a_writer, a_step, a_tag, a_value, a_name=None, **_):
                if isinstance(a_step, int) or hasattr(a_step, 'numpy'):
                    try:
                        str_tag = a_tag.numpy()
                        str_tag = str_tag.decode() if isinstance(str_tag, bytes) else str(str_tag)
                        event_writer._add_scalar(
                            tag=str_tag,
                            step=int(a_step.numpy()) if not isinstance(a_step, int) else a_step,
                            scalar_data=a_value.numpy())
                    except Exception as a_ex:
                        LoggerRoot.get_base_logger(TensorflowBinding).warning(
                            '_report_summary_op: {}'.format(str(a_ex)))

            # this is a mix of eager and graph execution
            try:
                from tensorflow.python.eager import context as _context
                if not _context.executing_eagerly():
                    from tensorflow import py_function
                    # just creating the operator is enough (for some reason)
                    # to make sure it is added into the execution tree.
                    # the operator itself, will do the reporting to the backend
                    py_function(
                        _report_summary_op,
                        inp=[writer, step, tag, value, name], Tout=[])
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))

        return PatchTensorFlowEager.__original_fn_scalar(writer, step, tag, value, name, **kwargs)

    @staticmethod
    def _write_hist_summary(writer, step, tag, values, name, **kwargs):
        event_writer = PatchTensorFlowEager._get_event_writer(writer)
        if event_writer and isinstance(step, int) or hasattr(step, 'numpy'):
            try:
                event_writer._add_histogram(
                    tag=str(tag), step=int(step.numpy()) if not isinstance(step, int) else step,
                    hist_data=values.numpy()
                )
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))
        elif event_writer:
            def _report_summary_op(a_writer, a_step, a_tag, a_value, a_name=None, **_):
                if isinstance(a_step, int) or hasattr(a_step, 'numpy'):
                    try:
                        str_tag = a_tag.numpy()
                        str_tag = str_tag.decode() if isinstance(str_tag, bytes) else str(str_tag)
                        event_writer._add_histogram(
                            tag=str_tag,
                            step=int(a_step.numpy()) if not isinstance(a_step, int) else a_step,
                            hist_data=a_value.numpy()
                        )
                    except Exception as a_ex:
                        LoggerRoot.get_base_logger(TensorflowBinding).warning(
                            '_report_summary_op: {}'.format(str(a_ex)))

            # this is a mix of eager and graph execution
            try:
                from tensorflow.python.eager import context as _context
                if not _context.executing_eagerly():
                    from tensorflow import py_function
                    # just creating the operator is enough (for some reason)
                    # to make sure it is added into the execution tree.
                    # the operator itself, will do the reporting to the backend
                    py_function(
                        _report_summary_op,
                        inp=[writer, step, tag, values, name], Tout=[])
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))

        return PatchTensorFlowEager.__original_fn_hist(writer, step, tag, values, name, **kwargs)

    @staticmethod
    def _write_image_summary(writer, step, tag, tensor, bad_color, max_images, name, **kwargs):
        event_writer = PatchTensorFlowEager._get_event_writer(writer)
        if event_writer and isinstance(step, int) or hasattr(step, 'numpy'):
            try:
                PatchTensorFlowEager._add_image_event_helper(event_writer, img_data_np=tensor.numpy(),
                                                             tag=tag, step=step, **kwargs)
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))
        elif event_writer:
            def _report_summary_op(a_writer, a_step, a_tag, a_tensor, a_bad_color, a_max_images, a_name=None, **_):
                if isinstance(a_step, int) or hasattr(a_step, 'numpy'):
                    try:
                        str_tag = a_tag.numpy()
                        str_tag = str_tag.decode() if isinstance(str_tag, bytes) else str(str_tag)
                        PatchTensorFlowEager._add_image_event_helper(
                            event_writer, img_data_np=a_tensor.numpy(),
                            tag=str_tag, step=a_step, **kwargs)

                    except Exception as a_ex:
                        LoggerRoot.get_base_logger(TensorflowBinding).warning(
                            '_report_summary_op: {}'.format(str(a_ex)))

            # this is a mix of eager and graph execution
            try:
                from tensorflow.python.eager import context as _context
                if not _context.executing_eagerly():
                    from tensorflow import py_function
                    # just creating the operator is enough (for some reason)
                    # to make sure it is added into the execution tree.
                    # the operator itself, will do the reporting to the backend
                    py_function(
                        _report_summary_op,
                        inp=[writer, step, tag, tensor, bad_color, max_images, name], Tout=[])
            except Exception as ex:
                LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))

        return PatchTensorFlowEager.__original_fn_image(
            writer, step, tag, tensor, bad_color, max_images, name, **kwargs)

    @staticmethod
    def _add_image_event_helper(event_writer, img_data_np, tag, step, **kwargs):
        if img_data_np.ndim == 1 and img_data_np.size >= 3 and \
                (len(img_data_np[0]) < 10 and len(img_data_np[1]) < 10):
            # this is just for making sure these are actually valid numbers
            width = int(img_data_np[0].decode())  # noqa: F841
            height = int(img_data_np[1].decode())  # noqa: F841
            for i in range(2, img_data_np.size):
                img_data = {'width': None, 'height': None,
                            'colorspace': 'RGB', 'encodedImageString': img_data_np[i]}
                image_tag = str(tag) + '/sample_{}'.format(i - 2) if img_data_np.size > 3 else str(tag)
                event_writer._add_image(tag=image_tag,
                                        step=int(step.numpy()) if not isinstance(step, int) else step,
                                        img_data=img_data)
        else:
            event_writer._add_image_numpy(tag=str(tag),
                                          step=int(step.numpy()) if not isinstance(step, int) else step,
                                          img_data_np=img_data_np,
                                          max_keep_images=kwargs.get('max_images'))

    @staticmethod
    def _nothing_op(*_, **__):
        """Convenient else branch for when summaries do not record."""
        from tensorflow.python.framework import constant_op
        return constant_op.constant(False)


# noinspection PyPep8Naming,SpellCheckingInspection
class PatchKerasModelIO(object):
    __main_task = None
    __patched_keras = None
    __patched_tensorflow = None

    @staticmethod
    def update_current_task(task, **_):
        PatchKerasModelIO.__main_task = task
        PatchKerasModelIO._patch_model_checkpoint()
        PostImportHookPatching.add_on_import('tensorflow', PatchKerasModelIO._patch_model_checkpoint)
        PostImportHookPatching.add_on_import('keras', PatchKerasModelIO._patch_model_checkpoint)

    @staticmethod
    def _patch_model_checkpoint():
        if 'keras' in sys.modules and not PatchKerasModelIO.__patched_keras:
            try:
                from keras.engine.network import Network  # noqa
            except ImportError:
                Network = None
            try:
                from keras.engine.functional import Functional  # noqa
            except ImportError:
                Functional = None
            try:
                from keras.engine.sequential import Sequential  # noqa
            except ImportError:
                Sequential = None
            try:
                from keras import models as keras_saving  # noqa
            except ImportError:
                keras_saving = None
            # check that we are not patching anything twice
            if PatchKerasModelIO.__patched_tensorflow:
                PatchKerasModelIO.__patched_keras = [
                    Network if PatchKerasModelIO.__patched_tensorflow[0] != Network else None,
                    Sequential if PatchKerasModelIO.__patched_tensorflow[1] != Sequential else None,
                    keras_saving if PatchKerasModelIO.__patched_tensorflow[2] != keras_saving else None,
                    Functional if PatchKerasModelIO.__patched_tensorflow[3] != Functional else None,
                    None,
                    None,
                ]
            else:
                PatchKerasModelIO.__patched_keras = [Network, Sequential, keras_saving, Functional, None, None]
            PatchKerasModelIO._patch_io_calls(*PatchKerasModelIO.__patched_keras)

        if 'tensorflow' in sys.modules and not PatchKerasModelIO.__patched_tensorflow:
            try:
                # hack: make sure tensorflow.__init__ is called
                import tensorflow  # noqa
                from tensorflow.python.keras.engine.network import Network  # noqa
            except ImportError:
                Network = None
            try:
                # hack: make sure tensorflow.__init__ is called
                import tensorflow  # noqa
                from tensorflow.python.keras.engine.functional import Functional  # noqa
            except ImportError:
                Functional = None
            try:
                # hack: make sure tensorflow.__init__ is called
                import tensorflow  # noqa
                from tensorflow.python.keras.engine.sequential import Sequential  # noqa
            except ImportError:
                Sequential = None
            try:
                # hack: make sure tensorflow.__init__ is called
                import tensorflow  # noqa
                from tensorflow.python.keras import models as keras_saving_legacy  # noqa
            except ImportError:
                keras_saving_legacy = None

            try:
                # hack: make sure tensorflow.__init__ is called
                import tensorflow  # noqa
                from tensorflow.keras import models as keras_saving  # noqa
            except ImportError:
                keras_saving = None

            try:
                # hack: make sure tensorflow.__init__ is called
                import tensorflow  # noqa
                from tensorflow.python.keras.saving import hdf5_format as keras_hdf5  # noqa
            except ImportError:
                keras_hdf5 = None

            if PatchKerasModelIO.__patched_keras:
                PatchKerasModelIO.__patched_tensorflow = [
                    Network if PatchKerasModelIO.__patched_keras[0] != Network else None,
                    Sequential if PatchKerasModelIO.__patched_keras[1] != Sequential else None,
                    keras_saving if PatchKerasModelIO.__patched_keras[2] != keras_saving else None,
                    Functional if PatchKerasModelIO.__patched_keras[3] != Functional else None,
                    keras_saving_legacy if PatchKerasModelIO.__patched_keras[4] != keras_saving_legacy else None,
                    keras_hdf5 if PatchKerasModelIO.__patched_keras[5] != keras_hdf5 else None,
                ]
            else:
                PatchKerasModelIO.__patched_tensorflow = [
                    Network, Sequential, keras_saving, Functional, keras_saving_legacy, keras_hdf5]
            PatchKerasModelIO._patch_io_calls(*PatchKerasModelIO.__patched_tensorflow)

    @staticmethod
    def _patch_io_calls(Network, Sequential, keras_saving, Functional, keras_saving_legacy=None, keras_hdf5=None):
        try:
            if Sequential is not None:
                Sequential._updated_config = _patched_call(Sequential._updated_config,
                                                           PatchKerasModelIO._updated_config)
                if hasattr(Sequential.from_config, '__func__'):
                    # noinspection PyUnresolvedReferences
                    Sequential.from_config = classmethod(_patched_call(Sequential.from_config.__func__,
                                                                       PatchKerasModelIO._from_config))
                else:
                    Sequential.from_config = _patched_call(Sequential.from_config, PatchKerasModelIO._from_config)

            if Network is not None:
                Network._updated_config = _patched_call(Network._updated_config, PatchKerasModelIO._updated_config)
                if hasattr(Sequential.from_config, '__func__'):
                    # noinspection PyUnresolvedReferences
                    Network.from_config = classmethod(_patched_call(Network.from_config.__func__,
                                                                    PatchKerasModelIO._from_config))
                else:
                    Network.from_config = _patched_call(Network.from_config, PatchKerasModelIO._from_config)
                Network.save = _patched_call(Network.save, PatchKerasModelIO._save)
                Network.save_weights = _patched_call(Network.save_weights, PatchKerasModelIO._save_weights)
                Network.load_weights = _patched_call(Network.load_weights, PatchKerasModelIO._load_weights)
            elif Functional is not None:
                Functional._updated_config = _patched_call(
                    Functional._updated_config, PatchKerasModelIO._updated_config)
                if hasattr(Sequential.from_config, '__func__'):
                    # noinspection PyUnresolvedReferences
                    Functional.from_config = classmethod(_patched_call(Functional.from_config.__func__,
                                                                       PatchKerasModelIO._from_config))
                else:
                    Functional.from_config = _patched_call(Functional.from_config, PatchKerasModelIO._from_config)
                Functional.save = _patched_call(Functional.save, PatchKerasModelIO._save)
                Functional.save_weights = _patched_call(Functional.save_weights, PatchKerasModelIO._save_weights)
                Functional.load_weights = _patched_call(Functional.load_weights, PatchKerasModelIO._load_weights)

            if keras_saving is not None:
                keras_saving.save_model = _patched_call(keras_saving.save_model, PatchKerasModelIO._save_model)
                keras_saving.load_model = _patched_call(keras_saving.load_model, PatchKerasModelIO._load_model)

            if keras_saving_legacy is not None:
                keras_saving_legacy.save_model = _patched_call(
                    keras_saving_legacy.save_model, PatchKerasModelIO._save_model)
                keras_saving_legacy.load_model = _patched_call(
                    keras_saving_legacy.load_model, PatchKerasModelIO._load_model)

            if keras_hdf5 is not None:
                keras_hdf5.save_weights_to_hdf5_group = _patched_call(
                    keras_hdf5.save_weights_to_hdf5_group, PatchKerasModelIO._save_weights)
                keras_hdf5.load_weights_from_hdf5_group = _patched_call(
                    keras_hdf5.load_weights_from_hdf5_group, PatchKerasModelIO._load_weights)
                keras_hdf5.load_weights_from_hdf5_group_by_name = _patched_call(
                    keras_hdf5.load_weights_from_hdf5_group_by_name, PatchKerasModelIO._load_weights)
                if hasattr(keras_hdf5, 'load_model_from_hdf5'):
                    keras_hdf5.load_model_from_hdf5 = _patched_call(
                        keras_hdf5.load_model_from_hdf5, PatchKerasModelIO._load_model)
                if hasattr(keras_hdf5, 'save_model_to_hdf5'):
                    keras_hdf5.save_model_to_hdf5 = _patched_call(
                        keras_hdf5.save_model_to_hdf5, PatchKerasModelIO._save_model)

        except Exception as ex:
            LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))

    @staticmethod
    def _updated_config(original_fn, self):
        config = original_fn(self)
        # check if we have main task
        if PatchKerasModelIO.__main_task is None:
            return config

        try:
            # there is no actual file, so we create the OutputModel without one

            # check if object already has InputModel
            if not hasattr(self, 'trains_out_model'):
                self.trains_out_model = []

            # check if object already has InputModel
            model_name_id = config.get('name', getattr(self, 'name', 'unknown'))
            if self.trains_out_model:
                self.trains_out_model[-1].config_dict = config
            else:
                # todo: support multiple models for the same task
                self.trains_out_model.append(OutputModel(
                    task=PatchKerasModelIO.__main_task,
                    config_dict=config,
                    name=PatchKerasModelIO.__main_task.name + ' ' + model_name_id,
                    label_enumeration=PatchKerasModelIO.__main_task.get_labels_enumeration(),
                    framework=Framework.keras,
                ))
        except Exception as ex:
            LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))

        return config

    @staticmethod
    def _from_config(original_fn, *args, **kwargs):
        try:
            self = original_fn(*args, **kwargs)
        except Exception as ex:
            if not running_remotely():
                raise ex
            self = _Empty()

        # check if we have main task
        if PatchKerasModelIO.__main_task is None:
            return self

        try:
            # check if object already has InputModel
            if not hasattr(self, 'trains_in_model'):
                self.trains_in_model = None

            # get config
            config_dict = kwargs['config'] if 'config' in kwargs else args[0]
            # check if object already has InputModel
            self.trains_in_model = InputModel.empty(
                config_dict=config_dict,
                label_enumeration=PatchKerasModelIO.__main_task.get_labels_enumeration(),
            )
            # todo: support multiple models for the same task
            PatchKerasModelIO.__main_task.connect(self.trains_in_model)
            # if we are running remotely we should deserialize the object
            # because someone might have changed the configuration
            # Hack: disabled
            if False and running_remotely():
                # reload the model
                model_config = self.trains_in_model.config_dict
                # verify that this is the same model so we are not deserializing a diff model
                if (config_dict and config_dict.get('config') and model_config and model_config.get('config') and
                    config_dict.get('config').get('name') == model_config.get('config').get('name')) or \
                        (not config_dict and not model_config):
                    if 'config' in kwargs:
                        kwargs['config'] = model_config
                    else:
                        args = (model_config,) + args[1:]
                    model = original_fn(*args, **kwargs)
                    model.trains_in_model = self.trains_in_model
                    return model

        except Exception as ex:
            LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))

        return self

    @staticmethod
    def _load_weights(original_fn, self, *args, **kwargs):
        # check if we have main task
        if PatchKerasModelIO.__main_task is None:
            return original_fn(self, *args, **kwargs)

        # get filepath
        if self and getattr(self, 'filename', None):
            filepath = getattr(self, 'filename', None)
        else:
            filepath = kwargs['filepath'] if 'filepath' in kwargs else args[0]

        # Hack: disabled
        if False and running_remotely():
            # register/load model weights
            filepath = WeightsFileHandler.restore_weights_file(self, filepath, Framework.keras,
                                                               PatchKerasModelIO.__main_task)
            if 'filepath' in kwargs:
                kwargs['filepath'] = filepath
            else:
                args = (filepath,) + args[1:]
            # load model
            return original_fn(self, *args, **kwargs)

        # try to load the files, if something happened exception will be raised before we register the file
        model = original_fn(self, *args, **kwargs)
        # register/load model weights
        WeightsFileHandler.restore_weights_file(self, filepath, Framework.keras, PatchKerasModelIO.__main_task)
        return model

    @staticmethod
    def _save(original_fn, self, *args, **kwargs):
        if hasattr(self, 'trains_out_model') and self.trains_out_model:
            # noinspection PyProtectedMember
            self.trains_out_model[-1]._processed = False
        original_fn(self, *args, **kwargs)
        # no need to specially call, because the original save uses "save_model" which we overload
        # noinspection PyProtectedMember
        if not hasattr(self, 'trains_out_model') or not self.trains_out_model or \
                not hasattr(self.trains_out_model[-1], '_processed') or not self.trains_out_model[-1]._processed:
            PatchKerasModelIO._update_outputmodel(self, *args, **kwargs)

    @staticmethod
    def _save_weights(original_fn, self, *args, **kwargs):
        original_fn(self, *args, **kwargs)
        PatchKerasModelIO._update_outputmodel(self, *args, **kwargs)

    @staticmethod
    def _update_outputmodel(self, *args, **kwargs):
        # check if we have main task
        if PatchKerasModelIO.__main_task is None:
            return

        try:
            # get filepath
            if self and getattr(self, 'filename', None):
                filepath = getattr(self, 'filename', None)
            else:
                filepath = kwargs['filepath'] if 'filepath' in kwargs else args[0]

            # this will already generate an output model
            # noinspection PyBroadException
            try:
                config = self._updated_config()
            except Exception:
                # we failed to convert the network to json, for some reason (most likely internal keras error)
                config = {}

            if filepath:
                WeightsFileHandler.create_output_model(
                    self, filepath, Framework.keras, PatchKerasModelIO.__main_task,
                    config_obj=config or None, singlefile=True)

        except Exception as ex:
            LoggerRoot.get_base_logger(TensorflowBinding).warning(str(ex))

    @staticmethod
    def _save_model(original_fn, model, filepath, *args, **kwargs):
        original_fn(model, filepath, *args, **kwargs)
        if PatchKerasModelIO.__main_task:
            PatchKerasModelIO._update_outputmodel(model, filepath)

    @staticmethod
    def _load_model(original_fn, filepath, *args, **kwargs):
        if not PatchKerasModelIO.__main_task:
            return original_fn(filepath, *args, **kwargs)

        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            # register/load model weights
            filepath = WeightsFileHandler.restore_weights_file(empty, filepath, Framework.keras,
                                                               PatchKerasModelIO.__main_task)
            model = original_fn(filepath, *args, **kwargs)
        else:
            model = original_fn(filepath, *args, **kwargs)
            # register/load model weights
            WeightsFileHandler.restore_weights_file(empty, filepath, Framework.keras, PatchKerasModelIO.__main_task)
        # update the input model object
        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass

        return model


class PatchTensorflowModelIO(object):
    __main_task = None
    __patched = None

    @staticmethod
    def update_current_task(task, **_):
        PatchTensorflowModelIO.__main_task = task
        PatchTensorflowModelIO._patch_model_checkpoint()
        PostImportHookPatching.add_on_import('tensorflow', PatchTensorflowModelIO._patch_model_checkpoint)

    @staticmethod
    def _patch_model_checkpoint():
        if PatchTensorflowModelIO.__patched:
            return

        if 'tensorflow' not in sys.modules:
            return

        PatchTensorflowModelIO.__patched = True
        # noinspection PyBroadException
        try:
            # hack: make sure tensorflow.__init__ is called
            import tensorflow  # noqa
            # noinspection PyUnresolvedReferences
            from tensorflow.python.training.saver import Saver  # noqa
            # noinspection PyBroadException
            try:
                Saver.save = _patched_call(Saver.save, PatchTensorflowModelIO._save)
            except Exception:
                pass
            # noinspection PyBroadException
            try:
                Saver.restore = _patched_call(Saver.restore, PatchTensorflowModelIO._restore)
            except Exception:
                pass
        except ImportError:
            pass
        except Exception:
            LoggerRoot.get_base_logger(TensorflowBinding).debug('Failed patching tensorflow')

        # noinspection PyBroadException
        try:
            # make sure we import the correct version of save
            import tensorflow  # noqa
            from tensorflow.saved_model import save  # noqa
            # actual import
            from tensorflow.python.saved_model import save as saved_model  # noqa
        except ImportError:
            # noinspection PyBroadException
            try:
                # make sure we import the correct version of save
                import tensorflow  # noqa
                from tensorflow.saved_model.experimental import save  # noqa
                # actual import
                import tensorflow.saved_model.experimental as saved_model  # noqa
            except ImportError:
                saved_model = None
            except Exception:
                saved_model = None

        except Exception:
            saved_model = None

        if saved_model is not None:
            saved_model.save = _patched_call(saved_model.save, PatchTensorflowModelIO._save_model)

        # noinspection PyBroadException
        try:
            # make sure we import the correct version of save
            import tensorflow  # noqa
            # actual import
            from tensorflow.saved_model import load  # noqa
            # noinspection PyUnresolvedReferences
            import tensorflow.saved_model as saved_model_load  # noqa
            saved_model_load.load = _patched_call(saved_model_load.load, PatchTensorflowModelIO._load)
        except ImportError:
            pass
        except Exception:
            LoggerRoot.get_base_logger(TensorflowBinding).debug('Failed patching tensorflow')

        # noinspection PyBroadException
        try:
            # make sure we import the correct version of save
            import tensorflow  # noqa
            # actual import
            # noinspection PyUnresolvedReferences
            from tensorflow.saved_model import loader as loader1  # noqa
            loader1.load = _patched_call(loader1.load, PatchTensorflowModelIO._load)
        except ImportError:
            pass
        except Exception:
            LoggerRoot.get_base_logger(TensorflowBinding).debug('Failed patching tensorflow')

        # noinspection PyBroadException
        try:
            # make sure we import the correct version of save
            import tensorflow  # noqa
            # actual import
            # noinspection PyUnresolvedReferences
            from tensorflow.compat.v1.saved_model import loader as loader2  # noqa
            loader2.load = _patched_call(loader2.load, PatchTensorflowModelIO._load)
        except ImportError:
            pass
        except Exception:
            LoggerRoot.get_base_logger(TensorflowBinding).debug('Failed patching tensorflow')

        # noinspection PyBroadException
        try:
            import tensorflow  # noqa
            from tensorflow.train import Checkpoint  # noqa
            # noinspection PyBroadException
            try:
                Checkpoint.save = _patched_call(Checkpoint.save, PatchTensorflowModelIO._ckpt_save)
            except Exception:
                pass
            # noinspection PyBroadException
            try:
                Checkpoint.restore = _patched_call(Checkpoint.restore, PatchTensorflowModelIO._ckpt_restore)
            except Exception:
                pass
            # noinspection PyBroadException
            try:
                Checkpoint.write = _patched_call(Checkpoint.write, PatchTensorflowModelIO._ckpt_write)
            except Exception:
                pass
        except ImportError:
            pass
        except Exception:
            LoggerRoot.get_base_logger(TensorflowBinding).debug('Failed patching tensorflow')

    @staticmethod
    def _save(original_fn, self, sess, save_path, *args, **kwargs):
        saved_path = original_fn(self, sess, save_path, *args, **kwargs)
        if not saved_path:
            return saved_path
        # store output Model
        return WeightsFileHandler.create_output_model(self, saved_path, Framework.tensorflow,
                                                      PatchTensorflowModelIO.__main_task)

    @staticmethod
    def _save_model(original_fn, obj, export_dir, *args, **kwargs):
        original_fn(obj, export_dir, *args, **kwargs)
        # store output Model
        WeightsFileHandler.create_output_model(obj, export_dir, Framework.tensorflow,
                                               PatchTensorflowModelIO.__main_task)

    @staticmethod
    def _restore(original_fn, self, sess, save_path, *args, **kwargs):
        if PatchTensorflowModelIO.__main_task is None:
            return original_fn(self, sess, save_path, *args, **kwargs)

        # Hack: disabled
        if False and running_remotely():
            # register/load model weights
            save_path = WeightsFileHandler.restore_weights_file(self, save_path, Framework.tensorflow,
                                                                PatchTensorflowModelIO.__main_task)
            # load model
            return original_fn(self, sess, save_path, *args, **kwargs)

        # load model, if something is wrong, exception will be raised before we register the input model
        model = original_fn(self, sess, save_path, *args, **kwargs)
        # register/load model weights
        WeightsFileHandler.restore_weights_file(self, save_path, Framework.tensorflow,
                                                PatchTensorflowModelIO.__main_task)
        return model

    @staticmethod
    def _load(original_fn, sess, tags, export_dir, *args, **saver_kwargs):
        if PatchTensorflowModelIO.__main_task is None:
            return original_fn(sess, tags, export_dir, *args, **saver_kwargs)

        # register input model
        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            export_dir = WeightsFileHandler.restore_weights_file(empty, export_dir, Framework.tensorflow,
                                                                 PatchTensorflowModelIO.__main_task)
            model = original_fn(sess, tags, export_dir, *args, **saver_kwargs)
        else:
            # try to load model before registering, it might fail
            model = original_fn(sess, tags, export_dir, *args, **saver_kwargs)
            WeightsFileHandler.restore_weights_file(empty, export_dir, Framework.tensorflow,
                                                    PatchTensorflowModelIO.__main_task)

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass
        return model

    @staticmethod
    def _ckpt_save(original_fn, self, file_prefix, *args, **kwargs):
        checkpoint_path = original_fn(self, file_prefix, *args, **kwargs)
        if PatchTensorflowModelIO.__main_task is None:
            return checkpoint_path
        WeightsFileHandler.create_output_model(self, checkpoint_path, Framework.tensorflow,
                                               PatchTensorflowModelIO.__main_task)
        return checkpoint_path

    @staticmethod
    def _ckpt_write(original_fn, self, file_prefix, *args, **kwargs):
        checkpoint_path = original_fn(self, file_prefix, *args, **kwargs)
        if PatchTensorflowModelIO.__main_task is None:
            return checkpoint_path
        WeightsFileHandler.create_output_model(self, checkpoint_path, Framework.tensorflow,
                                               PatchTensorflowModelIO.__main_task)
        return checkpoint_path

    @staticmethod
    def _ckpt_restore(original_fn, self, save_path, *args, **kwargs):
        if PatchTensorflowModelIO.__main_task is None:
            return original_fn(self, save_path, *args, **kwargs)

        # register input model
        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            save_path = WeightsFileHandler.restore_weights_file(empty, save_path, Framework.tensorflow,
                                                                PatchTensorflowModelIO.__main_task)
            model = original_fn(self, save_path, *args, **kwargs)
        else:
            # try to load model before registering it, in case it fails.
            model = original_fn(self, save_path, *args, **kwargs)
            WeightsFileHandler.restore_weights_file(empty, save_path, Framework.tensorflow,
                                                    PatchTensorflowModelIO.__main_task)

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass
        return model


class PatchTensorflow2ModelIO(object):
    __main_task = None
    __patched = None

    @staticmethod
    def update_current_task(task, **_):
        PatchTensorflow2ModelIO.__main_task = task
        PatchTensorflow2ModelIO._patch_model_checkpoint()
        PostImportHookPatching.add_on_import('tensorflow', PatchTensorflow2ModelIO._patch_model_checkpoint)

    @staticmethod
    def _patch_model_checkpoint():
        if PatchTensorflow2ModelIO.__patched:
            return

        if 'tensorflow' not in sys.modules:
            return

        PatchTensorflow2ModelIO.__patched = True
        # noinspection PyBroadException
        try:
            # hack: make sure tensorflow.__init__ is called
            import tensorflow  # noqa
            from tensorflow.python.training.tracking import util  # noqa
            # noinspection PyBroadException
            try:
                util.TrackableSaver.save = _patched_call(util.TrackableSaver.save,
                                                         PatchTensorflow2ModelIO._save)
            except Exception:
                pass
            # noinspection PyBroadException
            try:
                util.TrackableSaver.restore = _patched_call(util.TrackableSaver.restore,
                                                            PatchTensorflow2ModelIO._restore)
            except Exception:
                pass
        except ImportError:
            pass
        except Exception:
            LoggerRoot.get_base_logger(TensorflowBinding).debug('Failed patching tensorflow v2')

    @staticmethod
    def _save(original_fn, self, file_prefix, *args, **kwargs):
        model = original_fn(self, file_prefix, *args, **kwargs)
        # store output Model
        # noinspection PyBroadException
        try:
            WeightsFileHandler.create_output_model(self, file_prefix, Framework.tensorflow,
                                                   PatchTensorflow2ModelIO.__main_task)
        except Exception:
            pass
        return model

    @staticmethod
    def _restore(original_fn, self, save_path, *args, **kwargs):
        if PatchTensorflow2ModelIO.__main_task is None:
            return original_fn(self, save_path, *args, **kwargs)

        # Hack: disabled
        if False and running_remotely():
            # register/load model weights
            # noinspection PyBroadException
            try:
                save_path = WeightsFileHandler.restore_weights_file(self, save_path, Framework.tensorflow,
                                                                    PatchTensorflow2ModelIO.__main_task)
            except Exception:
                pass
            # load model
            return original_fn(self, save_path, *args, **kwargs)

        # load model, if something is wrong, exception will be raised before we register the input model
        model = original_fn(self, save_path, *args, **kwargs)
        # register/load model weights
        # noinspection PyBroadException
        try:
            WeightsFileHandler.restore_weights_file(self, save_path, Framework.tensorflow,
                                                    PatchTensorflow2ModelIO.__main_task)
        except Exception:
            pass
        return model
