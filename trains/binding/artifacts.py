import weakref

import numpy as np
import hashlib
from tempfile import mkdtemp
from threading import Thread, Event
from multiprocessing.pool import ThreadPool

from pathlib2 import Path
from ..debugging.log import LoggerRoot

try:
    import pandas as pd
except ImportError:
    pd = None


class Artifacts(object):
    _flush_frequency_sec = 300.
    # notice these two should match
    _save_format = '.csv.gz'
    _compression = 'gzip'
    # hashing constants
    _hash_block_size = 65536

    class _ProxyDictWrite(dict):
        """ Dictionary wrapper that updates an arguments instance on any item set in the dictionary """
        def __init__(self, artifacts_manager, *args, **kwargs):
            super(Artifacts._ProxyDictWrite, self).__init__(*args, **kwargs)
            self._artifacts_manager = artifacts_manager
            # list of artifacts we should not upload (by name & weak-reference)
            self.local_artifacts = {}

        def __setitem__(self, key, value):
            # check that value is of type pandas
            if isinstance(value, np.ndarray) or (pd and isinstance(value, pd.DataFrame)):
                super(Artifacts._ProxyDictWrite, self).__setitem__(key, value)

                if self._artifacts_manager:
                    self._artifacts_manager.flush()
            else:
                raise ValueError('Artifacts currently supports pandas.DataFrame objects only')

        def disable_upload(self, name):
            if name in self.keys():
                self.local_artifacts[name] = weakref.ref(self.get(name))

        def do_upload(self, name):
            # return True is this artifact should be uploaded
            return name not in self.local_artifacts or self.local_artifacts[name] != self.get(name)

    @property
    def artifacts(self):
        return self._artifacts_dict

    @property
    def summary(self):
        return self._summary

    def __init__(self, task):
        self._task = task
        # notice the double link, this important since the Artifact
        # dictionary needs to signal the Artifacts base on changes
        self._artifacts_dict = self._ProxyDictWrite(self)
        self._last_artifacts_upload = {}
        self._thread = None
        self._flush_event = Event()
        self._exit_flag = False
        self._thread_pool = ThreadPool()
        self._summary = ''
        self._temp_folder = []

    def add_artifact(self, name, artifact, upload=True):
        # currently we support pandas.DataFrame (which we will upload as csv.gz)
        # or numpy array, which we will upload as npz
        self._artifacts_dict[name] = artifact
        if not upload:
            self._artifacts_dict.disable_upload(name)

    def flush(self):
        # start the thread if it hasn't already:
        self._start()
        # flush the current state of all artifacts
        self._flush_event.set()

    def stop(self, wait=True):
        # stop the daemon thread and quit
        # wait until thread exists
        self._exit_flag = True
        self._flush_event.set()
        if wait:
            if self._thread:
                self._thread.join()
            # remove all temp folders
            for f in self._temp_folder:
                try:
                    Path(f).rmdir()
                except Exception:
                    pass

    def _start(self):
        if not self._thread:
            # start the daemon thread
            self._flush_event.clear()
            self._thread = Thread(target=self._daemon)
            self._thread.daemon = True
            self._thread.start()

    def _daemon(self):
        while not self._exit_flag:
            self._flush_event.wait(self._flush_frequency_sec)
            self._flush_event.clear()
            try:
                self._upload_artifacts()
            except Exception as e:
                LoggerRoot.get_base_logger().warning(str(e))

        # create summary
        self._summary = self._get_statistics()

    def _upload_artifacts(self):
        logger = self._task.get_logger()
        for name, artifact in self._artifacts_dict.items():
            if not self._artifacts_dict.do_upload(name):
                # only register artifacts, and leave, TBD
                continue
            local_csv = (Path(self._get_temp_folder()) / (name + self._save_format)).absolute()
            if local_csv.exists():
                # we are still uploading... get another temp folder
                local_csv = (Path(self._get_temp_folder(force_new=True)) / (name + self._save_format)).absolute()
            artifact.to_csv(local_csv.as_posix(), index=False, compression=self._compression)
            current_sha2 = self.sha256sum(local_csv.as_posix(), skip_header=32)
            if name in self._last_artifacts_upload:
                previous_sha2 = self._last_artifacts_upload[name]
                if previous_sha2 == current_sha2:
                    # nothing to do, we can skip the upload
                    local_csv.unlink()
                    continue
            self._last_artifacts_upload[name] = current_sha2
            # now upload and delete at the end.
            logger.report_image_and_upload(title='artifacts', series=name, path=local_csv.as_posix(),
                                           delete_after_upload=True, iteration=self._task.get_last_iteration(),
                                           max_image_history=2)

    def _get_statistics(self):
        summary = ''
        thread_pool = ThreadPool()

        try:
            # build hash row sets
            artifacts_summary = []
            for a_name, a_df in self._artifacts_dict.items():
                if not pd or not isinstance(a_df, pd.DataFrame):
                    continue

                a_unique_hash = set()

                def hash_row(r):
                    a_unique_hash.add(hash(bytes(r)))

                a_shape = a_df.shape
                # parallelize
                thread_pool.map(hash_row, a_df.values)
                # add result
                artifacts_summary.append((a_name, a_shape, a_unique_hash,))

            # build intersection summary
            for i, (name, shape, unique_hash) in enumerate(artifacts_summary):
                summary += '[{name}]: shape={shape}, {unique} unique rows, {percentage:.1f}% uniqueness\n'.format(
                    name=name, shape=shape, unique=len(unique_hash), percentage=100*len(unique_hash)/float(shape[0]))
                for name2, shape2, unique_hash2 in artifacts_summary[i+1:]:
                    intersection = len(unique_hash & unique_hash2)
                    summary += '\tIntersection with [{name2}] {intersection} rows: {percentage:.1f}%\n'.format(
                        name2=name2, intersection=intersection, percentage=100*intersection/float(len(unique_hash2)))
        except Exception as e:
            LoggerRoot.get_base_logger().warning(str(e))
        finally:
            thread_pool.close()
            thread_pool.terminate()
        return summary

    def _get_temp_folder(self, force_new=False):
        if force_new or not self._temp_folder:
            new_temp = mkdtemp(prefix='artifacts_')
            self._temp_folder.append(new_temp)
            return new_temp
        return self._temp_folder[0]

    @staticmethod
    def sha256sum(filename, skip_header=0):
        # create sha2 of the file, notice we skip the header of the file (32 bytes)
        # because sometimes that is the only change
        h = hashlib.sha256()
        b = bytearray(Artifacts._hash_block_size)
        mv = memoryview(b)
        with open(filename, 'rb', buffering=0) as f:
            # skip header
            f.read(skip_header)
            for n in iter(lambda: f.readinto(mv), 0):
                h.update(mv[:n])
        return h.hexdigest()
