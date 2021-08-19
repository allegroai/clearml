import logging
import os
from time import time
from typing import Optional, AnyStr, IO
from ..config import config


class ProgressReport(object):
    def __init__(self, verbose, total_size, log, report_chunk_size_mb):
        self.current_status_mb = 0.
        self.last_reported = 0.
        self._tic = time()
        self._verbose = verbose
        self._report_chunk_size = report_chunk_size_mb
        self._log = log
        self._log_flag = False
        self._total_size = total_size

    def __call__(self, chunk_size, *_, **__):
        chunk_size /= 1024. * 1024.
        self.current_status_mb += chunk_size
        last_part = self.current_status_mb - self.last_reported

        if self._verbose or (last_part >= self._report_chunk_size):
            time_diff = time() - self._tic
            self.speed = (last_part / time_diff) if time_diff != 0 else 0
            self._tic = time()
            self.last_reported = self.current_status_mb
            self._report(self._total_size, self.current_status_mb, self.speed)

    def _report(self, total_mb, current_mb, speed_mbps):
        # type: (float, float, float) -> None
        pass


class UploadProgressReport(ProgressReport):
    def __init__(self, filename, verbose, total_size, log, report_chunk_size_mb=0):
        if not report_chunk_size_mb:
            report_chunk_size_mb = int(config.get('storage.log.report_upload_chunk_size_mb', 0) or 5)
        super(UploadProgressReport, self).__init__(verbose, total_size, log, report_chunk_size_mb)
        self._filename = filename

    def _report(self, total_mb, current_mb, speed_mbps):
        # type: (float, float, float) -> None
        self._log.info(
            'Uploading: %.2fMB / %.2fMB @ %.2fMBs from %s' %
            (current_mb, total_mb, speed_mbps, self._filename)
        )

    @classmethod
    def from_stream(cls, stream, filename, verbose, log):
        # type: (IO[AnyStr], str, bool, logging.Logger) -> Optional[UploadProgressReport]
        if hasattr(stream, 'seek'):
            total_size = cls._get_stream_length(stream)
            return UploadProgressReport(filename, verbose, total_size, log)

    @classmethod
    def from_file(cls, filename, verbose, log):
        # type: (str, bool, logging.Logger) -> UploadProgressReport
        total_size_mb = float(os.path.getsize(filename)) / (1024. * 1024.)
        return UploadProgressReport(filename, verbose, total_size_mb, log)

    @staticmethod
    def _get_stream_length(stream):
        # type: (IO[AnyStr]) -> int
        current_position = stream.tell()
        # seek to end of file
        stream.seek(0, 2)
        total_length = stream.tell()
        # seek back to current position to support
        # partially read file-like objects
        stream.seek(current_position or 0)
        return total_length


class DownloadProgressReport(ProgressReport):
    def __init__(self, total_size, verbose, remote_path, log, report_chunk_size_mb=0):
        if not report_chunk_size_mb:
            report_chunk_size_mb = int(config.get('storage.log.report_download_chunk_size_mb', 0) or 5)

        super(DownloadProgressReport, self).__init__(verbose, total_size, log, report_chunk_size_mb)
        self._remote_path = remote_path

    def _report(self, total_mb, current_mb, speed_mbps):
        # type: (float, float, float) -> None
        self._log.info('Downloading: %.2fMB / %.2fMB @ %.2fMBs from %s' %
                       (current_mb, total_mb, speed_mbps, self._remote_path))
