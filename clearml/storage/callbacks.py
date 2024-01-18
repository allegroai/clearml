import logging
import os
from time import time
from typing import Optional, AnyStr, IO
from ..config import config

try:
    from tqdm import tqdm  # noqa
except ImportError:
    tqdm = None


class ProgressReport(object):
    report_upload_chunk_size_mb = None
    report_download_chunk_size_mb = None

    def __init__(self, verbose, total_size, log, report_chunk_size_mb,
                 description_prefix=None, description_suffix=None,
                 max_time_between_reports_sec=10.0, report_start=None):
        self.current_status_mb = 0.
        self.last_reported = 0.
        self._tic = time()
        self._verbose = verbose
        self._report_chunk_size = report_chunk_size_mb
        self._log = log
        self._log_flag = False
        self._total_size = total_size
        self._description_prefix = description_prefix
        self._description_suffix = description_suffix
        self._max_time_between_reports_sec = max_time_between_reports_sec
        self._report_start = report_start if report_start is not None else bool(tqdm is not None)
        self._tqdm = None
        self._tqdm_init = False

    def close(self, report_completed=False, report_summary=False, report_prefix=None, report_suffix=None):
        # call this one when we are done
        if self._tqdm is not None:
            # if we created a self._tqdm object we need to close it
            if report_completed:
                self._tqdm.update(
                    self._tqdm.total - min(self._tqdm.total, self.last_reported)
                )
            self._tqdm.close()
            self._tqdm = None

        if report_summary:
            self._log.info(
                '{} {:.2f} MB successfully {}'.format(
                    report_prefix or self._description_prefix, self._total_size,
                    report_suffix or self._description_suffix).strip()
            )

    def _get_tqdm(self):
        if self._tqdm_init:
            return self._tqdm

        self._tqdm_init = True

        # create the tqdm progress bar
        if tqdm:
            # noinspection PyBroadException
            try:
                self._tqdm = tqdm(
                    total=round(float(self._total_size), 2),
                    # desc="{} {}".format(description_prefix, description_suffix).strip(),
                    unit="MB",
                    unit_scale=False,
                    ncols=80,
                    bar_format="{bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} MB "
                               "[{elapsed}<{remaining}, {rate_fmt}{postfix}]: {desc}",
                )
            except Exception:
                # failed initializing TQDM (maybe interface changed?)
                self._tqdm = None

        return self._tqdm

    def __call__(self, chunk_size, *_, **__):
        chunk_size /= 1024. * 1024.
        self.current_status_mb += chunk_size
        last_part = self.current_status_mb - self.last_reported

        if (self._verbose or (last_part >= self._report_chunk_size) or
                (self.last_reported and self.current_status_mb >= self._total_size-0.01) or
                (time()-self._tic > self._max_time_between_reports_sec)):
            time_diff = time() - self._tic
            self.speed = (last_part / time_diff) if time_diff != 0 else 0
            self._report(self._total_size, self.current_status_mb, self.speed)
            self._tic = time()
            self.last_reported = self.current_status_mb

    def _report(self, total_mb, current_mb, speed_mbps):
        # type: (float, float, float) -> None
        if self._report_start and self.last_reported <= 0:
            # first time - print before initializing the tqdm bar
            self._log.info(
                "{}: {:.2f}MB {}".format(
                    self._description_prefix, total_mb, self._description_suffix).strip(" :")
            )

        # initialize or reuse the bar
        _tqdm = self._get_tqdm()
        if _tqdm:
            # make sure we do not spill over due to rounding
            if round(float(current_mb), 2) >= _tqdm.total:
                _tqdm.update(_tqdm.total - self.last_reported)
            else:
                _tqdm.update(current_mb - self.last_reported)
        else:
            self._log.info(
                "{}: {:.2f}MB / {:.2f}MB @ {:.2f}MBs {}".format(
                    self._description_prefix,
                    current_mb,
                    total_mb,
                    speed_mbps,
                    self._description_suffix
                ).strip(" :")
            )


class UploadProgressReport(ProgressReport):
    def __init__(self, filename, verbose, total_size, log, report_chunk_size_mb=None, report_start=None):
        report_chunk_size_mb = report_chunk_size_mb if report_chunk_size_mb is not None \
            else ProgressReport.report_upload_chunk_size_mb or \
            int(config.get("storage.log.report_upload_chunk_size_mb", 5))
        super(UploadProgressReport, self).__init__(
            verbose, total_size, log, report_chunk_size_mb,
            description_prefix="Uploading", description_suffix="to {}".format(filename),
            report_start=report_start,
        )
        self._filename = filename

    @classmethod
    def from_stream(cls, stream, filename, verbose, log):
        # type: (IO[AnyStr], str, bool, logging.Logger) -> Optional[UploadProgressReport]
        if hasattr(stream, 'seek'):
            total_size_mb = cls._get_stream_length(stream) // (1024 * 1024)
            return UploadProgressReport(filename, verbose, total_size_mb, log)

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
    def __init__(self, total_size, verbose, remote_path, log, report_chunk_size_mb=None, report_start=None):
        report_chunk_size_mb = report_chunk_size_mb if report_chunk_size_mb is not None \
            else ProgressReport.report_download_chunk_size_mb or \
            int(config.get("storage.log.report_download_chunk_size_mb", 5))
        super(DownloadProgressReport, self).__init__(
            verbose, total_size, log, report_chunk_size_mb,
            description_prefix="Downloading", description_suffix="from {}".format(remote_path),
            report_start=report_start,
        )
        self._remote_path = remote_path
