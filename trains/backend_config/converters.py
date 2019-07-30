import base64
from distutils.util import strtobool
from typing import Union, Optional, Any, TypeVar, Callable, Tuple

import six

try:
    from typing import Text
except ImportError:
    # windows conda-less hack
    Text = Any


ConverterType = TypeVar("ConverterType", bound=Callable[[Any], Any])


def base64_to_text(value):
    # type: (Any) -> Text
    return base64.b64decode(value).decode("utf-8")


def text_to_bool(value):
    # type: (Text) -> bool
    return bool(strtobool(value))


def any_to_bool(value):
    # type: (Optional[Union[int, float, Text]]) -> bool
    if isinstance(value, six.text_type):
        return text_to_bool(value)
    return bool(value)


def or_(*converters, **kwargs):
    # type: (ConverterType, Tuple[Exception, ...]) -> ConverterType
    """
    Wrapper that implements an "optional converter" pattern. Allows specifying a converter
    for which a set of exceptions is ignored (and the original value is returned)
    :param converters: A converter callable
    :param exceptions: A tuple of exception types to ignore
    """
    # noinspection PyUnresolvedReferences
    exceptions = kwargs.get("exceptions", (ValueError, TypeError))

    def wrapper(value):
        for converter in converters:
            try:
                return converter(value)
            except exceptions:
                pass
        return value

    return wrapper
