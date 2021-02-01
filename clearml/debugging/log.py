""" Logging convenience functions and wrappers """
import inspect
import logging
import logging.handlers
import os
import sys
from platform import system

from pathlib2 import Path
from six import BytesIO

default_level = logging.INFO


class _LevelRangeFilter(logging.Filter):

    def __init__(self, min_level, max_level, name=''):
        super(_LevelRangeFilter, self).__init__(name)
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record):
        return self.min_level <= record.levelno <= self.max_level


class LoggerRoot(object):
    __base_logger = None

    @classmethod
    def _make_stream_handler(cls, level=None, stream=sys.stdout, colored=False):
        ch = logging.StreamHandler(stream=stream)
        ch.setLevel(level)
        formatter = None

        # if colored, try to import colorama & coloredlogs (by default, not in the requirements)
        if colored:
            try:
                import colorama
                from coloredlogs import ColoredFormatter
                colorama.init()
                formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            except ImportError:
                colored = False

        # if we don't need or failed getting colored formatter
        if not colored or not formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch.setFormatter(formatter)
        return ch

    @classmethod
    def get_base_logger(cls, level=None, stream=sys.stdout, colored=False):
        if LoggerRoot.__base_logger:
            return LoggerRoot.__base_logger
        # avoid nested imports
        from ..config import get_log_redirect_level

        LoggerRoot.__base_logger = logging.getLogger('clearml')
        level = level if level is not None else default_level
        LoggerRoot.__base_logger.setLevel(level)

        redirect_level = get_log_redirect_level()

        # Do not redirect to stderr if the target stream is already stderr
        if redirect_level is not None and stream not in (None, sys.stderr):
            # Adjust redirect level in case requested level is higher (e.g. logger is requested for CRITICAL
            # and redirect is set for ERROR, in which case we redirect from CRITICAL)
            redirect_level = max(level, redirect_level)
            LoggerRoot.__base_logger.addHandler(
                cls._make_stream_handler(redirect_level, sys.stderr, colored)
            )

            if level < redirect_level:
                # Not all levels were redirected, remaining should be sent to requested stream
                handler = cls._make_stream_handler(level, stream, colored)
                handler.addFilter(_LevelRangeFilter(min_level=level, max_level=redirect_level - 1))
                LoggerRoot.__base_logger.addHandler(handler)
        else:
            LoggerRoot.__base_logger.addHandler(
                cls._make_stream_handler(level, stream, colored)
            )

        LoggerRoot.__base_logger.propagate = False
        return LoggerRoot.__base_logger

    @classmethod
    def flush(cls):
        if LoggerRoot.__base_logger:
            for h in LoggerRoot.__base_logger.handlers:
                h.flush()


def add_options(parser):
    """ Add logging options to an argparse.ArgumentParser object """
    level = logging.getLevelName(default_level)
    parser.add_argument(
        '--log-level', '-l', default=level, help='Log level (default is %s)' % level)


def apply_logging_args(args):
    """ Apply logging args from an argparse.ArgumentParser parsed args """
    global default_level
    default_level = logging.getLevelName(args.log_level.upper())


def get_logger(path=None, level=None, stream=None, colored=False):
    """ Get a python logging object named using the provided filename and preconfigured with a color-formatted
        stream handler
    """
    # noinspection PyBroadException
    try:
        path = path or os.path.abspath((inspect.stack()[1])[1])
    except BaseException:
        # if for some reason we could not find the calling file, use our own
        path = os.path.abspath(__file__)
    root_log = LoggerRoot.get_base_logger(level=default_level, stream=sys.stdout, colored=colored)
    log = root_log.getChild(Path(path).stem)
    if level is not None:
        log.setLevel(level)
    if stream:
        ch = logging.StreamHandler(stream=stream)
        if level is not None:
            ch.setLevel(level)
    log.propagate = True
    return log


def _add_file_handler(logger, log_dir, fh, formatter=None):
    """ Adds a file handler to a logger """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if not formatter:
        log_format = '%(asctime)s %(name)s x_x[%(levelname)s] %(message)s'
        formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def add_rotating_file_handler(logger, log_dir, log_file_prefix, max_bytes=10 * 1024 * 1024, backup_count=20,
                              formatter=None):
    """ Create and add a rotating file handler to a logger """
    fh = logging.handlers.RotatingFileHandler(
        str(Path(log_dir) / ('%s.log' % log_file_prefix)), maxBytes=max_bytes, backupCount=backup_count)
    _add_file_handler(logger, log_dir, fh, formatter)


def add_time_rotating_file_handler(logger, log_dir, log_file_prefix, when='midnight', formatter=None):
    """
        Create and add a time rotating file handler to a logger.
        Possible values for when are 'midnight', weekdays ('w0'-'W6', when 0 is Monday), and 's', 'm', 'h' amd 'd' for
            seconds, minutes, hours and days respectively (case-insensitive)
    """
    fh = logging.handlers.TimedRotatingFileHandler(
        str(Path(log_dir) / ('%s.log' % log_file_prefix)), when=when)
    _add_file_handler(logger, log_dir, fh, formatter)


def get_null_logger(name=None):
    """ Get a logger with a null handler """
    log = logging.getLogger(name if name else 'null')
    if not log.handlers:
        # avoid nested imports
        from ..config import config

        log.addHandler(logging.NullHandler())
        log.propagate = config.get("log.null_log_propagate", False)
    return log


class TqdmLog(object):
    """ Tqdm (progressbar) wrapped logging class """

    class _TqdmIO(BytesIO):
        """ IO wrapper class for Tqdm """

        def __init__(self, level=20, logger=None, *args, **kwargs):
            self._log = logger or get_null_logger()
            self._level = level
            BytesIO.__init__(self, *args, **kwargs)

        def write(self, buf):
            self._buf = buf.strip('\r\n\t ')

        def flush(self):
            self._log.log(self._level, self._buf)

    def __init__(self, total, desc='', log_level=20, ascii=False, logger=None, smoothing=0, mininterval=5, initial=0):
        from tqdm import tqdm

        self._io = self._TqdmIO(level=log_level, logger=logger)
        self._tqdm = tqdm(total=total, desc=desc, file=self._io, ascii=ascii if not system() == 'Windows' else True,
                          smoothing=smoothing,
                          mininterval=mininterval, initial=initial)

    def update(self, n=None):
        if n is not None:
            self._tqdm.update(n=n)
        else:
            self._tqdm.update()

    def close(self):
        self._tqdm.close()
