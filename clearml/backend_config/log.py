import logging.config
from copy import deepcopy

from pathlib2 import Path


def logger(path=None):
    name = "clearml"
    if path:
        p = Path(path)
        module = (p.parent if p.stem.startswith('_') else p).stem
        name = "clearml.%s" % module
    return logging.getLogger(name)


def initialize(logging_config=None, extra=None):
    if extra is not None:
        from logging import Logger

        class _Logger(Logger):
            __extra = extra.copy()

            def _log(self, level, msg, args, exc_info=None, extra=None, **kwargs):
                extra = extra or {}
                extra.update(self.__extra)
                super(_Logger, self)._log(level, msg, args, exc_info=exc_info, extra=extra, **kwargs)

        Logger.manager.loggerClass = _Logger

    if logging_config is not None:
        # Use deepcopy since Python's logging infrastructure might modify the dict
        logging.config.dictConfig(deepcopy(dict(logging_config)))
