import json
import os
from functools import partial
from logging import warning
from multiprocessing.pool import ThreadPool
from time import time

from pathlib2 import Path

from ...backend_api.services import events as api_events
from ..base import InterfaceBase
from ...config import config
from ...debugging import get_logger
from ...storage.helper import StorageHelper

from .events import MetricsEventAdapter
from ...utilities.process.mp import SingletonLock


log = get_logger('metrics')


class Metrics(InterfaceBase):
    """ Metrics manager and batch writer """
    _storage_lock = SingletonLock()
    _file_upload_starvation_warning_sec = config.get('network.metrics.file_upload_starvation_warning_sec', None)
    _file_upload_retries = 3
    _upload_pool = None
    _file_upload_pool = None
    __offline_filename = 'metrics.jsonl'

    @property
    def storage_key_prefix(self):
        return self._storage_key_prefix

    def _get_storage(self, storage_uri=None):
        """ Storage helper used to upload files """
        try:
            # use a lock since this storage object will be requested by thread pool threads, so we need to make sure
            # any singleton initialization will occur only once
            self._storage_lock.acquire()
            storage_uri = storage_uri or self._storage_uri
            return StorageHelper.get(storage_uri)
        except Exception as e:
            log.error('Failed getting storage helper for %s: %s' % (storage_uri, str(e)))
        finally:
            self._storage_lock.release()

    def __init__(self, session, task, storage_uri, storage_uri_suffix='metrics', iteration_offset=0, log=None):
        super(Metrics, self).__init__(session, log=log)
        self._task_id = task.id
        self._task_iteration_offset = iteration_offset
        self._storage_uri = storage_uri.rstrip('/') if storage_uri else None
        self._storage_key_prefix = storage_uri_suffix.strip('/') if storage_uri_suffix else None
        self._file_related_event_time = None
        self._file_upload_time = None
        self._offline_log_filename = None
        if self._offline_mode:
            offline_folder = Path(task.get_offline_mode_folder())
            offline_folder.mkdir(parents=True, exist_ok=True)
            self._offline_log_filename = offline_folder / self.__offline_filename

    def write_events(self, events, async_enable=True, callback=None, **kwargs):
        """
        Write events to the backend, uploading any required files.
        :param events: A list of event objects
        :param async_enable: If True, upload is performed asynchronously and an AsyncResult object is returned,
            otherwise a blocking call is made and the upload result is returned.
        :param callback: A optional callback called when upload was completed in case async is True
        :return: .backend_api.session.CallResult if async is False otherwise AsyncResult. Note that if no events were
            sent, None will be returned.
        """
        if not events:
            return

        storage_uri = kwargs.pop('storage_uri', self._storage_uri)

        if not async_enable:
            return self._do_write_events(events, storage_uri)

        def safe_call(*args, **kwargs):
            try:
                return self._do_write_events(*args, **kwargs)
            except Exception as e:
                return e

        self._initialize_upload_pools()
        return self._upload_pool.apply_async(
            safe_call,
            args=(events, storage_uri),
            callback=partial(self._callback_wrapper, callback))

    def set_iteration_offset(self, offset):
        self._task_iteration_offset = offset

    def get_iteration_offset(self):
        return self._task_iteration_offset

    def _callback_wrapper(self, callback, res):
        """ A wrapper for the async callback for handling common errors """
        if not res:
            # no result yet
            return
        elif isinstance(res, Exception):
            # error
            self.log.error('Error trying to send metrics: %s' % str(res))
        elif not res.ok():
            # bad result, log error
            self.log.error('Failed reporting metrics: %s' % str(res.meta))
        # call callback, even if we received an error
        if callback:
            callback(res)

    def _do_write_events(self, events, storage_uri=None):
        """ Sends an iterable of events as a series of batch operations. note: metric send does not raise on error"""
        assert isinstance(events, (list, tuple))
        assert all(isinstance(x, MetricsEventAdapter) for x in events)

        # def event_key(ev):
        #     return (ev.metric, ev.variant)
        #
        # events = sorted(events, key=event_key)
        # multiple_events_for = [k for k, v in groupby(events, key=event_key) if len(list(v)) > 1]
        # if multiple_events_for:
        #     log.warning(
        #         'More than one metrics event sent for these metric/variant combinations in a report: %s' %
        #         ', '.join('%s/%s' % k for k in multiple_events_for))

        storage_uri = storage_uri or self._storage_uri

        def update_and_get_file_entry(ev):
            entry = ev.get_file_entry()
            kwargs = {}
            if entry:
                key, url = ev.get_target_full_upload_uri(storage_uri, self.storage_key_prefix, quote_uri=False)
                kwargs[entry.key_prop] = key
                kwargs[entry.url_prop] = url
                if not entry.stream:
                    # if entry has no stream, we won't upload it
                    entry = None
                else:
                    if not isinstance(entry.stream, Path) and not hasattr(entry.stream, 'read'):
                        raise ValueError('Invalid file object %s' % entry.stream)
                    entry.url = url
            ev.update(task=self._task_id, iter_offset=self._task_iteration_offset, **kwargs)
            return entry

        # prepare event needing file upload
        entries = []
        for ev in events[:]:
            try:
                e = update_and_get_file_entry(ev)
                if e:
                    entries.append(e)
            except Exception as ex:
                log.warning(str(ex))
                events.remove(ev)

        # upload the needed files
        if entries:
            # upload files
            def upload(e):
                upload_uri = e.upload_uri or storage_uri

                try:
                    storage = self._get_storage(upload_uri)
                    retries = getattr(e, 'retries', None) or self._file_upload_retries
                    if isinstance(e.stream, Path):
                        url = storage.upload(e.stream.as_posix(), e.url, retries=retries)
                    else:
                        url = storage.upload_from_stream(e.stream, e.url, retries=retries)
                    e.event.update(url=url)
                except Exception as exp:
                    log.warning("Failed uploading to {} ({})".format(
                        upload_uri if upload_uri else "(Could not calculate upload uri)",
                        exp,
                    ))

                    e.set_exception(exp)
                if not isinstance(e.stream, Path):
                    e.stream.close()
                if e.delete_local_file:
                    # noinspection PyBroadException
                    try:
                        Path(e.delete_local_file).unlink()
                    except Exception:
                        pass

            self._initialize_upload_pools()
            res = self._file_upload_pool.map_async(upload, entries)
            res.wait()

            # remember the last time we uploaded a file
            self._file_upload_time = time()

        t_f, t_u, t_ref = \
            (self._file_related_event_time, self._file_upload_time, self._file_upload_starvation_warning_sec)
        if t_f and t_u and t_ref and (t_f - t_u) > t_ref:
            log.warning('Possible metrics file upload starvation: '
                        'files were not uploaded for {} seconds'.format(t_ref))

        # send the events in a batched request
        good_events = [ev for ev in events if ev.upload_exception is None]
        error_events = [ev for ev in events if ev.upload_exception is not None]

        if error_events:
            log.error("Not uploading {}/{} events because the data upload failed".format(
                len(error_events),
                len(events),
            ))

        if good_events:
            _events = [ev.get_api_event() for ev in good_events]
            batched_requests = [api_events.AddRequest(event=ev) for ev in _events if ev]
            if batched_requests:
                if self._offline_mode:
                    with open(self._offline_log_filename.as_posix(), 'at') as f:
                        f.write(json.dumps([b.to_dict() for b in batched_requests])+'\n')
                    return

                req = api_events.AddBatchRequest(requests=batched_requests)
                return self.send(req, raise_on_errors=False)

        return None

    @staticmethod
    def _initialize_upload_pools():
        if not Metrics._upload_pool:
            Metrics._upload_pool = ThreadPool(processes=1)
        if not Metrics._file_upload_pool:
            Metrics._file_upload_pool = ThreadPool(
                processes=config.get('network.metrics.file_upload_threads', 4))

    @staticmethod
    def close_async_threads():
        file_pool = Metrics._file_upload_pool
        Metrics._file_upload_pool = None
        pool = Metrics._upload_pool
        Metrics._upload_pool = None

        if file_pool:
            # noinspection PyBroadException
            try:
                file_pool.terminate()
                file_pool.join()
            except Exception:
                pass

        if pool:
            # noinspection PyBroadException
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass

    @classmethod
    def report_offline_session(cls, task, folder):
        from ... import StorageManager
        filename = Path(folder) / cls.__offline_filename
        if not filename.is_file():
            return False
        # noinspection PyProtectedMember
        remote_url = task._get_default_report_storage_uri()
        if remote_url and remote_url.endswith('/'):
            remote_url = remote_url[:-1]
        uploaded_files = set()
        task_id = task.id
        with open(filename.as_posix(), 'rt') as f:
            i = 0
            while True:
                try:
                    line = f.readline()
                    if not line:
                        break
                    list_requests = json.loads(line)
                    for r in list_requests:
                        org_task_id = r['task']
                        r['task'] = task_id
                        if r.get('key') and r.get('url'):
                            debug_sample = (Path(folder) / 'data').joinpath(*(r['key'].split('/')))
                            r['key'] = r['key'].replace(
                                '.{}{}'.format(org_task_id, os.sep), '.{}{}'.format(task_id, os.sep), 1)
                            r['url'] = '{}/{}'.format(remote_url, r['key'])
                            if debug_sample not in uploaded_files and debug_sample.is_file():
                                uploaded_files.add(debug_sample)
                                StorageManager.upload_file(local_file=debug_sample.as_posix(), remote_url=r['url'])
                        elif r.get('plot_str'):
                            # hack plotly embedded images links
                            # noinspection PyBroadException
                            try:
                                task_id_sep = '.{}{}'.format(org_task_id, os.sep)
                                plot = json.loads(r['plot_str'])
                                if plot.get('layout', {}).get('images'):
                                    for image in plot['layout']['images']:
                                        if task_id_sep not in image['source']:
                                            continue
                                        pre, post = image['source'].split(task_id_sep, 1)
                                        pre = os.sep.join(pre.split(os.sep)[-2:])
                                        debug_sample = (Path(folder) / 'data').joinpath(
                                            pre+'.{}'.format(org_task_id), post)
                                        image['source'] = '/'.join(
                                            [remote_url, pre + '.{}'.format(task_id), post])
                                        if debug_sample not in uploaded_files and debug_sample.is_file():
                                            uploaded_files.add(debug_sample)
                                            StorageManager.upload_file(
                                                local_file=debug_sample.as_posix(), remote_url=image['source'])
                                r['plot_str'] = json.dumps(plot)
                            except Exception:
                                pass
                    i += 1
                except StopIteration:
                    break
                except Exception as ex:
                    warning('Failed reporting metric, line {} [{}]'.format(i, ex))
                batch_requests = api_events.AddBatchRequest(requests=list_requests)
                if batch_requests.requests:
                    res = task.session.send(batch_requests)
                    if res and not res.ok():
                        warning("failed logging metric task to backend ({:d} lines, {})".format(
                            len(batch_requests.requests), str(res.meta)))
        return True
