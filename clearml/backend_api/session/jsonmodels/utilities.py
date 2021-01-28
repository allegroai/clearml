from __future__ import absolute_import

import six
import re
from collections import namedtuple

SCALAR_TYPES = tuple(list(six.string_types) + [int, float, bool])

ECMA_TO_PYTHON_FLAGS = {
    'i': re.I,
    'm': re.M,
}

PYTHON_TO_ECMA_FLAGS = dict(
    (value, key) for key, value in ECMA_TO_PYTHON_FLAGS.items()
)

PythonRegex = namedtuple('PythonRegex', ['regex', 'flags'])


def _normalize_string_type(value):
    if isinstance(value, six.string_types):
        return six.text_type(value)
    else:
        return value


def _compare_dicts(one, two):
    if len(one) != len(two):
        return False

    for key, value in one.items():
        if key not in one or key not in two:
            return False

        if not compare_schemas(one[key], two[key]):
            return False
    return True


def _compare_lists(one, two):
    if len(one) != len(two):
        return False

    they_match = False
    for first_item in one:
        for second_item in two:
            if they_match:
                continue
            they_match = compare_schemas(first_item, second_item)
    return they_match


def _assert_same_types(one, two):
    if not isinstance(one, type(two)) or not isinstance(two, type(one)):
        raise RuntimeError('Types mismatch! "{type1}" and "{type2}".'.format(
            type1=type(one).__name__, type2=type(two).__name__))


def compare_schemas(one, two):
    """Compare two structures that represents JSON schemas.

    For comparison you can't use normal comparison, because in JSON schema
    lists DO NOT keep order (and Python lists do), so this must be taken into
    account during comparison.

    Note this wont check all configurations, only first one that seems to
    match, which can lead to wrong results.

    :param one: First schema to compare.
    :param two: Second schema to compare.
    :rtype: `bool`

    """
    one = _normalize_string_type(one)
    two = _normalize_string_type(two)

    _assert_same_types(one, two)

    if isinstance(one, list):
        return _compare_lists(one, two)
    elif isinstance(one, dict):
        return _compare_dicts(one, two)
    elif isinstance(one, SCALAR_TYPES):
        return one == two
    elif one is None:
        return one is two
    else:
        raise RuntimeError('Not allowed type "{type}"'.format(
            type=type(one).__name__))


def is_ecma_regex(regex):
    """Check if given regex is of type ECMA 262 or not.

    :rtype: bool

    """
    parts = regex.split('/')

    if len(parts) == 1:
        return False

    if len(parts) < 3:
        raise ValueError('Given regex isn\'t ECMA regex nor Python regex.')
    parts.pop()
    parts.append('')

    raw_regex = '/'.join(parts)
    if raw_regex.startswith('/') and raw_regex.endswith('/'):
        return True
    return False


def convert_ecma_regex_to_python(value):
    """Convert ECMA 262 regex to Python tuple with regex and flags.

    If given value is already Python regex it will be returned unchanged.

    :param string value: ECMA regex.
    :return: 2-tuple with `regex` and `flags`
    :rtype: namedtuple

    """
    if not is_ecma_regex(value):
        return PythonRegex(value, [])

    parts = value.split('/')
    flags = parts.pop()

    try:
        result_flags = [ECMA_TO_PYTHON_FLAGS[f] for f in flags]
    except KeyError:
        raise ValueError('Wrong flags "{}".'.format(flags))

    return PythonRegex('/'.join(parts[1:]), result_flags)


def convert_python_regex_to_ecma(value, flags=[]):
    """Convert Python regex to ECMA 262 regex.

    If given value is already ECMA regex it will be returned unchanged.

    :param string value: Python regex.
    :param list flags: List of flags (allowed flags: `re.I`, `re.M`)
    :return: ECMA 262 regex
    :rtype: str

    """
    if is_ecma_regex(value):
        return value

    result_flags = [PYTHON_TO_ECMA_FLAGS[f] for f in flags]
    result_flags = ''.join(result_flags)

    return '/{value}/{flags}'.format(value=value, flags=result_flags)
