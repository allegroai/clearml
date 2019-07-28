import abc
import time
from threading import Lock

import attr
import cv2
import numpy as np
import pathlib2
import six
from ...backend_api.services import events
from six.moves.urllib.parse import urlparse, urlunparse

from ...config import config


@six.add_metaclass(abc.ABCMeta)
class MetricsEventAdapter(object):
    """
    Adapter providing all the base attributes required by a metrics event and defining an interface used by the
    metrics manager when batching and writing events.
    """

    _default_nan_value = 0.
    """ Default value used when a np.nan value is encountered """

    @attr.attrs(cmp=False, slots=True)
    class FileEntry(object):
        """ File entry used to report on file data that needs to be uploaded prior to sending the event """

        event = attr.attrib()

        name = attr.attrib()
        """ File name """

        stream = attr.attrib()
        """ File-like object containing the file's data """

        url_prop = attr.attrib()
        """ Property name that should be updated with the uploaded url """

        key_prop = attr.attrib()

        upload_uri = attr.attrib()

        url = attr.attrib(default=None)

        exception = attr.attrib(default=None)

        delete_local_file = attr.attrib(default=True)
        """ Local file path, if exists, delete the file after upload completed """

        def set_exception(self, exp):
            self.exception = exp
            self.event.upload_exception = exp

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value

    @property
    def variant(self):
        return self._variant

    def __init__(self, metric, variant, iter=None, timestamp=None, task=None, gen_timestamp_if_none=True):
        if not timestamp and gen_timestamp_if_none:
            timestamp = int(time.time() * 1000)
        self._metric = metric
        self._variant = variant
        self._iter = iter
        self._timestamp = timestamp
        self._task = task

        # Try creating an event just to trigger validation
        _ = self.get_api_event()
        self.upload_exception = None

    @abc.abstractmethod
    def get_api_event(self):
        """ Get an API event instance """
        pass

    def get_file_entry(self):
        """ Get information for a file that should be uploaded before this event is sent """
        pass

    def update(self, task=None, **kwargs):
        """ Update event properties """
        if task:
            self._task = task

    def _get_base_dict(self):
        """ Get a dict with the base attributes """
        res = dict(
            task=self._task,
            timestamp=self._timestamp,
            metric=self._metric,
            variant=self._variant
        )
        if self._iter is not None:
            res.update(iter=self._iter)
        return res

    @classmethod
    def _convert_np_nan(cls, val):
        if np.isnan(val) or np.isinf(val):
            return cls._default_nan_value
        return val


class ScalarEvent(MetricsEventAdapter):
    """ Scalar event adapter """

    def __init__(self, metric, variant, value, iter, **kwargs):
        self._value = self._convert_np_nan(value)
        super(ScalarEvent, self).__init__(metric=metric, variant=variant, iter=iter, **kwargs)

    def get_api_event(self):
        return events.MetricsScalarEvent(
            value=self._value,
            **self._get_base_dict())


class VectorEvent(MetricsEventAdapter):
    """ Vector event adapter """

    def __init__(self, metric, variant, values, iter, **kwargs):
        self._values = [self._convert_np_nan(v) for v in values]
        super(VectorEvent, self).__init__(metric=metric, variant=variant, iter=iter, **kwargs)

    def get_api_event(self):
        return events.MetricsVectorEvent(
            values=self._values,
            **self._get_base_dict())


class PlotEvent(MetricsEventAdapter):
    """ Plot event adapter """

    def __init__(self, metric, variant, plot_str, iter=None, **kwargs):
        self._plot_str = plot_str
        super(PlotEvent, self).__init__(metric=metric, variant=variant, iter=iter, **kwargs)

    def get_api_event(self):
        return events.MetricsPlotEvent(
            plot_str=self._plot_str,
            **self._get_base_dict())


class ImageEventNoUpload(MetricsEventAdapter):

    def __init__(self, metric, variant, src, iter=0, **kwargs):
        parts = urlparse(src)
        self._url = urlunparse((parts.scheme, parts.netloc, '', '', '', ''))
        self._key = urlunparse(('', '', parts.path, parts.params, parts.query, parts.fragment))
        super(ImageEventNoUpload, self).__init__(metric, variant, iter=iter, **kwargs)

    def get_api_event(self):
        return events.MetricsImageEvent(
            url=self._url,
            key=self._key,
            **self._get_base_dict())


class UploadEvent(MetricsEventAdapter):
    """ Image event adapter """
    _format = '.' + str(config.get('metrics.images.format', 'JPEG')).upper().lstrip('.')
    _quality = int(config.get('metrics.images.quality', 87))
    _subsampling = int(config.get('metrics.images.subsampling', 0))

    _metric_counters = {}
    _metric_counters_lock = Lock()
    _image_file_history_size = int(config.get('metrics.file_history_size', 5))

    def __init__(self, metric, variant, image_data, local_image_path=None, iter=0, upload_uri=None,
                 image_file_history_size=None, delete_after_upload=False, **kwargs):
        if image_data is not None and not hasattr(image_data, 'shape'):
            raise ValueError('Image must have a shape attribute')
        self._image_data = image_data
        self._local_image_path = local_image_path
        self._url = None
        self._key = None
        self._count = self._get_metric_count(metric, variant)
        if not image_file_history_size:
            image_file_history_size = self._image_file_history_size
        if image_file_history_size < 1:
            self._filename = '%s_%s_%08d' % (metric, variant, self._count)
        else:
            self._filename = '%s_%s_%08d' % (metric, variant, self._count % image_file_history_size)
        self._upload_uri = upload_uri
        self._delete_after_upload = delete_after_upload

        # get upload uri upfront, either predefined image format or local file extension
        # e.g.: image.png -> .png or image.raw.gz -> .raw.gz
        image_format = self._format.lower() if self._image_data is not None else \
            '.' + '.'.join(pathlib2.Path(self._local_image_path).parts[-1].split('.')[1:])
        self._upload_filename = str(pathlib2.Path(self._filename).with_suffix(image_format))
        super(UploadEvent, self).__init__(metric, variant, iter=iter, **kwargs)

    @classmethod
    def _get_metric_count(cls, metric, variant, next=True):
        """ Returns the next count number for the given metric/variant (rotates every few calls) """
        counters = cls._metric_counters
        key = '%s_%s' % (metric, variant)
        try:
            cls._metric_counters_lock.acquire()
            value = counters.get(key, -1)
            if next:
                value = counters[key] = value + 1
            return value
        finally:
            cls._metric_counters_lock.release()

    # return No event (just the upload)
    def get_api_event(self):
        return None

    def update(self, url=None, key=None, **kwargs):
        super(UploadEvent, self).update(**kwargs)
        if url is not None:
            self._url = url
        if key is not None:
            self._key = key

    def get_file_entry(self):
        local_file = None
        # don't provide file in case this event is out of the history window
        last_count = self._get_metric_count(self.metric, self.variant, next=False)
        if abs(self._count - last_count) > self._image_file_history_size:
            output = None
        elif self._image_data is not None:
            image_data = self._image_data
            if not isinstance(image_data, np.ndarray):
                # try conversion, if it fails we'll leave it to the user.
                image_data = np.ndarray(image_data, dtype=np.uint8)
            image_data = np.atleast_3d(image_data)
            if image_data.dtype != np.uint8:
                if np.issubdtype(image_data.dtype, np.floating) and image_data.max() <= 1.0:
                    image_data = (image_data*255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
            shape = image_data.shape
            height, width, channel = shape[:3]
            if channel == 1:
                image_data = np.reshape(image_data, (height, width))

            # serialize image
            _, img_bytes = cv2.imencode(
                self._format, image_data,
                params=(cv2.IMWRITE_JPEG_QUALITY, self._quality),
            )

            output = six.BytesIO(img_bytes.tostring())
            output.seek(0)
        else:
            local_file = self._local_image_path
            try:
                output = open(local_file, 'rb')
            except Exception as e:
                # something happened to the file, we should skip it
                from ...debugging.log import LoggerRoot
                LoggerRoot.get_base_logger().warning(str(e))
                return None

        return self.FileEntry(
            event=self,
            name=self._upload_filename,
            stream=output,
            url_prop='url',
            key_prop='key',
            upload_uri=self._upload_uri,
            delete_local_file=local_file if self._delete_after_upload else None,
        )

    def get_target_full_upload_uri(self, storage_uri, storage_key_prefix):
        e_storage_uri = self._upload_uri or storage_uri
        # if we have an entry (with or without a stream), we'll generate the URL and store it in the event
        filename = self._upload_filename
        key = '/'.join(x for x in (storage_key_prefix, self.metric, self.variant, filename.strip('/')) if x)
        url = '/'.join(x.strip('/') for x in (e_storage_uri, key))
        return key, url


class ImageEvent(UploadEvent):
    def __init__(self, metric, variant, image_data, local_image_path=None, iter=0, upload_uri=None,
                 image_file_history_size=None, delete_after_upload=False, **kwargs):
        super(ImageEvent, self).__init__(metric, variant, image_data=image_data, local_image_path=local_image_path,
                                         iter=iter, upload_uri=upload_uri,
                                         image_file_history_size=image_file_history_size,
                                         delete_after_upload=delete_after_upload, **kwargs)

    def get_api_event(self):
        return events.MetricsImageEvent(
            url=self._url,
            key=self._key,
            **self._get_base_dict())
