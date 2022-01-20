import json
from os.path import expanduser

import six
from pathlib2 import Path

from . import running_remotely
from .defs import SESSION_CACHE_FILE


class SessionCache(object):
    """
    Handle SDK session cache.
    TODO: Improve error handling to something like "except (FileNotFoundError, PermissionError, JSONDecodeError)"
    TODO: that's both six-compatible and tested
    """

    SESSION_CACHE_FOLDER = "~/.clearml"

    @classmethod
    def _load_cache(cls):
        # noinspection PyBroadException
        try:
            flag = 'rb' if six.PY2 else 'rt'
            with (Path(expanduser(cls.SESSION_CACHE_FOLDER)) / SESSION_CACHE_FILE).open(flag) as fp:
                return json.load(fp)
        except Exception:
            return {}

    @classmethod
    def _store_cache(cls, cache):
        # noinspection PyBroadException
        try:
            Path(expanduser(cls.SESSION_CACHE_FOLDER)).mkdir(parents=True, exist_ok=True)
            flag = 'wb' if six.PY2 else 'wt'
            with (Path(expanduser(cls.SESSION_CACHE_FOLDER)) / SESSION_CACHE_FILE).open(flag) as fp:
                json.dump(cache, fp)
        except Exception:
            pass

    @classmethod
    def store_dict(cls, unique_cache_name, dict_object):
        # type: (str, dict) -> None
        # disable session cache when running in remote execution mode
        if running_remotely():
            return
        cache = cls._load_cache()
        cache[unique_cache_name] = dict_object
        cls._store_cache(cache)

    @classmethod
    def load_dict(cls, unique_cache_name):
        # type: (str) -> dict
        # disable session cache when running in remote execution mode
        if running_remotely():
            return {}
        cache = cls._load_cache()
        return cache.get(unique_cache_name, {}) if cache else {}
