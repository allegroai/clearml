import hashlib
import re
import sys
from typing import Optional, Union

from six.moves.urllib.parse import quote, urlparse, urlunparse
import six
import fnmatch

from ..debugging.log import LoggerRoot


def get_config_object_matcher(**patterns):
    unsupported = {k: v for k, v in patterns.items() if not isinstance(v, six.string_types)}
    if unsupported:
        raise ValueError('Unsupported object matcher (expecting string): %s'
                         % ', '.join('%s=%s' % (k, v) for k, v in unsupported.items()))

    # optimize simple patters
    starts_with = {k: v.rstrip('*') for k, v in patterns.items() if '*' not in v.rstrip('*') and '?' not in v}
    patterns = {k: v for k, v in patterns.items() if v not in starts_with}

    def _matcher(**kwargs):
        for key, value in kwargs.items():
            if not value:
                continue
            start = starts_with.get(key)
            if start:
                if value.startswith(start):
                    return True
            else:
                pat = patterns.get(key)
                if pat and fnmatch.fnmatch(value, pat):
                    return True

    return _matcher


def quote_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return url
    parsed = parsed._replace(path=quote(parsed.path))
    return urlunparse(parsed)


def encode_string_to_filename(text):
    return quote(text, safe=" ")


def sha256sum(filename, skip_header=0, block_size=65536):
    # type: (str, int, int) -> (Optional[str], Optional[str])
    # create sha2 of the file, notice we skip the header of the file (32 bytes)
    # because sometimes that is the only change
    h = hashlib.sha256()
    file_hash = hashlib.sha256()
    b = bytearray(block_size)
    mv = memoryview(b)
    try:
        with open(filename, 'rb', buffering=0) as f:
            # skip header
            if skip_header:
                file_hash.update(f.read(skip_header))
            # noinspection PyUnresolvedReferences
            for n in iter(lambda: f.readinto(mv), 0):
                h.update(mv[:n])
                if skip_header:
                    file_hash.update(mv[:n])
    except Exception as e:
        LoggerRoot.get_base_logger().warning(str(e))
        return None, None

    return h.hexdigest(), file_hash.hexdigest() if skip_header else None


def md5text(text, seed=1337):
    # type: (str, Union[int, str]) -> str
    """
    Return md5 hash of a string
    Do not use this hash for security, if needed use something stronger like SHA2

    :param text: string to hash
    :param seed: use prefix seed for hashing
    :return: md5 string
    """
    h = hashlib.md5()
    h.update((str(seed) + str(text)).encode('utf-8'))
    return h.hexdigest()


def is_windows():
    """
    :return: True if currently running on windows OS
    """
    return sys.platform == 'win32'


def format_size(size_in_bytes, binary=False):
    # type: (Union[int, float], bool) -> str
    """
    Return the size in human readable format (string)
    Matching humanfriendly.format_size outputs

    :param size_in_bytes: number of bytes
    :param binary: If `True` 1 Kb equals 1024 bytes, if False (default) 1 KB = 1000 bytes
    :return: string representation of the number of bytes (b,Kb,Mb,Gb, Tb,)
        >>> format_size(0)
        '0 bytes'
        >>> format_size(1)
        '1 byte'
        >>> format_size(5)
        '5 bytes'
        > format_size(1000)
        '1 KB'
        > format_size(1024, binary=True)
        '1 KiB'
        >>> format_size(1000 ** 3 * 4)
        '4 GB'
    """
    size = float(size_in_bytes)
    # single byte is the exception here
    if size == 1:
        return '{} byte'.format(int(size))
    k = 1024 if binary else 1000
    scale = ('bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB') if binary else ('bytes', 'KB', 'MB', 'GB', 'TB', 'PB')
    for i, m in enumerate(scale):
        if size < k**(i+1) or i == len(scale)-1:
            return ('{:.2f}'.format(size/(k**i)).rstrip('0').rstrip('.')
                    if i > 0 else '{}'.format(int(size))) + ' ' + m
    # we should never get here
    return '{} {}'.format(int(size), scale[0])


def parse_size(size, binary=False):
    """
    Parse a human readable data size and return the number of bytes.
    Match humanfriendly.parse_size

    :param size: The human readable file size to parse (a string).
    :param binary: :data:`True` to use binary multiples of bytes (base-2) for
                   ambiguous unit symbols and names, :data:`False` to use
                   decimal multiples of bytes (base-10).
    :returns: The corresponding size in bytes (an integer).
    :raises: :exc:`InvalidSize` when the input can't be parsed.

    This function knows how to parse sizes in bytes, kilobytes, megabytes,
    gigabytes, terabytes and petabytes. Some examples:
        >>> parse_size('42')
        42
        >>> parse_size('13b')
        13
        >>> parse_size('5 bytes')
        5
        >>> parse_size('1 KB')
        1000
        >>> parse_size('1 kilobyte')
        1000
        >>> parse_size('1 KiB')
        1024
        >>> parse_size('1 KB', binary=True)
        1024
        >>> parse_size('1.5 GB')
        1500000000
        >>> parse_size('1.5 GB', binary=True)
        1610612736
    """
    def tokenize(text):
        tokenized_input = []
        for token in re.split(r'(\d+(?:\.\d+)?)', text):
            token = token.strip()
            if re.match(r'\d+\.\d+', token):
                tokenized_input.append(float(token))
            elif token.isdigit():
                tokenized_input.append(int(token))
            elif token:
                tokenized_input.append(token)
        return tokenized_input
    tokens = tokenize(str(size))
    if tokens and isinstance(tokens[0], (int, float)):
        disk_size_units_b = \
            (('B', 'bytes'), ('KiB', 'kibibyte'), ('MiB', 'mebibyte'), ('GiB', 'gibibyte'),
             ('TiB', 'tebibyte'), ('PiB', 'pebibyte'))
        disk_size_units_d = \
            (('B', 'bytes'), ('KB', 'kilobyte'), ('MB', 'megabyte'), ('GB', 'gigabyte'),
             ('TB', 'terabyte'), ('PB', 'petabyte'))
        disk_size_units_b = [(1024 ** i, s[0], s[1]) for i, s in enumerate(disk_size_units_b)]
        k = 1024 if binary else 1000
        disk_size_units_d = [(k ** i, s[0], s[1]) for i, s in enumerate(disk_size_units_d)]
        disk_size_units = (disk_size_units_b + disk_size_units_d) \
            if binary else (disk_size_units_d + disk_size_units_b)

        # Get the normalized unit (if any) from the tokenized input.
        normalized_unit = tokens[1].lower() if len(tokens) == 2 and isinstance(tokens[1], str) else ''
        # If the input contains only a number, it's assumed to be the number of
        # bytes. The second token can also explicitly reference the unit bytes.
        if len(tokens) == 1 or normalized_unit.startswith('b'):
            return int(tokens[0])
        # Otherwise we expect two tokens: A number and a unit.
        if normalized_unit:
            # Convert plural units to singular units, for details:
            # https://github.com/xolox/python-humanfriendly/issues/26
            normalized_unit = normalized_unit.rstrip('s')
            for k, low, high in disk_size_units:
                # First we check for unambiguous symbols (KiB, MiB, GiB, etc)
                # and names (kibibyte, mebibyte, gibibyte, etc) because their
                # handling is always the same.
                if normalized_unit in (low.lower(), high.lower()):
                    return int(tokens[0] * k)
                # Now we will deal with ambiguous prefixes (K, M, G, etc),
                # symbols (KB, MB, GB, etc) and names (kilobyte, megabyte,
                # gigabyte, etc) according to the caller's preference.
                if (normalized_unit in (low.lower(), high.lower()) or
                        normalized_unit.startswith(low.lower())):
                    return int(tokens[0] * k)

    raise ValueError("Failed to parse size! (input {} was tokenized as {})".format(size, tokens))
