from os import getenv, environ

from .converters import text_to_bool
from .entry import Entry, NotSet


class EnvEntry(Entry):
    @classmethod
    def default_conversions(cls):
        conversions = super(EnvEntry, cls).default_conversions().copy()
        conversions[bool] = text_to_bool
        return conversions

    def __init__(self, key, *more_keys, **kwargs):
        super(EnvEntry, self).__init__(key, *more_keys, **kwargs)
        self._ignore_errors = kwargs.pop('ignore_errors', False)

    def _get(self, key):
        value = getenv(key, "").strip()
        return value or NotSet

    def _set(self, key, value):
        environ[key] = value

    def __str__(self):
        return "env:{}".format(super(EnvEntry, self).__str__())

    def error(self, message):
        if not self._ignore_errors:
            print("Environment configuration: {}".format(message))

    def exists(self):
        return any(key for key in self.keys if getenv(key) is not None)
