import hashlib
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
