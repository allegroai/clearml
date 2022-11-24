""" Logging convenience functions and wrappers """
import inspect
import logging
import logging.handlers
import os
import sys
from os import getenv
from platform import system
from typing import Optional, Union

from pathlib2 import Path
from six import BytesIO

default_level = logging.INFO

_levelToName = {
    logging.CRITICAL: 'CRITICAL',
    logging.ERROR: 'ERROR',
    logging.WARNING: 'WARNING',
    logging.INFO: 'INFO',
    logging.DEBUG: 'DEBUG',
    logging.NOTSET: 'NOTSET',
}

_nameToLevel = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}


def resolve_logging_level(level):
    # type: (Union[str, int]) -> Optional[int]
    # noinspection PyBroadException
    try:
        level = int(level)
    except Exception:
        pass
    if isinstance(level, str):
        return _nameToLevel.get(level.upper(), None)
    if level in _levelToName:
        return level


class PickledLogger(logging.getLoggerClass()):

    def __init__(self, *args, **kwargs):
        super(PickledLogger, self).__init__(*args, **kwargs)
        self._init_kwargs = None

    @staticmethod
    def wrapper(a_instance, func, **kwargs):
        # if python 3.7 and above Loggers are pickle-able
        if sys.version_info.major >= 3 and sys.version_info.minor >= 7:
            return a_instance

        safe_logger = PickledLogger(name=kwargs.get('name'))
        safe_logger.__dict__ = a_instance.__dict__
        if 'stream' in kwargs and kwargs['stream']:
            kwargs['stream'] = 'stdout' if kwargs['stream'] == sys.stdout else (
                'stderr' if kwargs['stream'] == sys.stderr else kwargs['stream'])
        else:
            kwargs['stream'] = None
        kwargs['_func'] = func
        safe_logger._init_kwargs = kwargs
        return safe_logger

    def __getstate__(self):
        return self._init_kwargs or {}

    def __setstate__(self, state):
        state['stream'] = sys.stdout if state['stream'] == 'stdout' else (
            sys.stderr if state['stream'] == 'stderr' else state['stream'])
        _func = state.pop('_func') or self.__class__
        self.__dict__ = _func(**state).__dict__


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

        # Note we can't use LOG_LEVEL_ENV_VAR defined in clearml.config.defs due to a circular dependency
        if level is None and getenv("CLEARML_LOG_LEVEL"):
            level = resolve_logging_level(getenv("CLEARML_LOG_LEVEL").strip())
            if level is None:
                print('Invalid value in environment variable CLEARML_LOG_LEVEL: %s' % getenv("CLEARML_LOG_LEVEL"))

        clearml_logger = logging.getLogger('clearml')

        if level is None:
            level = clearml_logger.level

        # avoid nested imports
        from ..config import get_log_redirect_level
        LoggerRoot.__base_logger = PickledLogger.wrapper(
            clearml_logger,
            func=cls.get_base_logger,
            level=level,
            stream=stream,
            colored=colored
        )

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
    root_log = LoggerRoot.get_base_logger(stream=sys.stdout, colored=colored)
    log = root_log.getChild(Path(path).stem)
    if level is not None:
        log.setLevel(level)
    if stream:
        ch = logging.StreamHandler(stream=stream)
        if level is not None:
            ch.setLevel(level)
        log.addHandler(ch)
    log.propagate = True
    return PickledLogger.wrapper(
        log, func=get_logger, path=path, level=level, stream=stream, colored=colored)


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
    return PickledLogger.wrapper(log, func=get_null_logger, name=name)


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
