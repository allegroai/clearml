import abc
from typing import Optional, Any, Tuple, Callable, Dict

import six

from .converters import any_to_bool

try:
    from typing import Text
except ImportError:
    # windows conda-less hack
    Text = Any


NotSet = object()

Converter = Callable[[Any], Any]


@six.add_metaclass(abc.ABCMeta)
class Entry(object):
    """
    Configuration entry definition
    """

    @classmethod
    def default_conversions(cls):
        # type: () -> Dict[Any, Converter]
        return {
            bool: any_to_bool,
            six.text_type: lambda s: six.text_type(s).strip(),
        }

    def __init__(self, key, *more_keys, **kwargs):
        # type: (Text, Text, Any) -> None
        """
        :param key: Entry's key (at least one).
        :param more_keys: More alternate keys for this entry.
        :param type: Value type. If provided, will be used choosing a default conversion or
        (if none exists) for casting the environment value.
        :param converter: Value converter. If provided, will be used to convert the environment value.
        :param default: Default value. If provided, will be used as the default value on calls to get() and get_pair()
        in case no value is found for any key and no specific default value was provided in the call.
        Default value is None.
        :param help: Help text describing this entry
        """
        self.keys = (key,) + more_keys
        self.type = kwargs.pop("type", six.text_type)
        self.converter = kwargs.pop("converter", None)
        self.default = kwargs.pop("default", None)
        self.help = kwargs.pop("help", None)

    def __str__(self):
        return str(self.key)

    @property
    def key(self):
        return self.keys[0]

    def convert(self, value, converter=None):
        # type: (Any, Converter) -> Optional[Any]
        converter = converter or self.converter
        if not converter:
            converter = self.default_conversions().get(self.type, self.type)
        return converter(value)

    def get_pair(self, default=NotSet, converter=None):
        # type: (Any, Converter) -> Optional[Tuple[Text, Any]]
        for key in self.keys:
            value = self._get(key)
            if value is NotSet:
                continue
            try:
                value = self.convert(value, converter)
            except Exception as ex:
                self.error("invalid value {key}={value}: {ex}".format(**locals()))
                break
            return key, value
        result = self.default if default is NotSet else default
        return self.key, result

    def get(self, default=NotSet, converter=None):
        # type: (Any, Converter) -> Optional[Any]
        return self.get_pair(default=default, converter=converter)[1]

    def set(self, value):
        # type: (Any, Any) -> (Text, Any)
        key, _ = self.get_pair(default=None, converter=None)
        self._set(key, str(value))

    def _set(self, key, value):
        # type: (Text, Text) -> None
        pass

    @abc.abstractmethod
    def _get(self, key):
        # type: (Text) -> Any
        pass

    @abc.abstractmethod
    def error(self, message):
        # type: (Text) -> None
        pass

    def exists(self):
        # type: () -> bool
        return any(key for key in self.keys if self._get(key) is not NotSet)
