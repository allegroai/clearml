from __future__ import division

import six
import humanfriendly


def parse_human_size(value):
    if isinstance(value, six.string_types):
        return humanfriendly.parse_size(value)
    return value


def get_percentage(config, key, required=True, default=None):
    if required:
        value = config.get(key)
    else:
        value = config.get(key, default)
        if value is None:
            return
    try:
        if isinstance(value, six.string_types):
            value = value.strip()
            if value.endswith('%'):
                # "50%" => 0.5
                return float(value.strip('%')) / 100.
            # "50" => 50

        value = float(value)
        if value < 1:
            # 0.5 => 50% => 0.5
            return value

        # 50 => 0.5, 10.5 => 0.105
        return value / 100.

    except ValueError as e:
        raise ValueError('Config: failed parsing %s: %s' % (key, e))


def get_human_size_default(config, key, default=None):
    raw_value = config.get(key, default)

    if raw_value is None:
        return default

    return parse_human_size(raw_value)
