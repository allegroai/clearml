from os import getenv, environ

from .converters import text_to_bool
from .entry import Entry, NotSet


class EnvEntry(Entry):
    @classmethod
    def default_conversions(cls):
        conversions = super(EnvEntry, cls).default_conversions().copy()
        conversions[bool] = text_to_bool
        return conversions

    def _get(self, key):
        value = getenv(key, "").strip()
        return value or NotSet

    def _set(self, key, value):
        environ[key] = value

    def __str__(self):
        return "env:{}".format(super(EnvEntry, self).__str__())

    def error(self, message):
        print("Environment configuration: {}".format(message))
