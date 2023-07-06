import fnmatch
import hashlib
import json
import os.path
import re
import sys
from typing import Optional, Union, Sequence, Dict
from zlib import crc32

import six
from pathlib2 import Path
from six.moves.urllib.parse import quote, urlparse, urlunparse

from ..debugging.log import LoggerRoot


def get_config_object_matcher(**patterns):
    unsupported = {k: v for k, v in patterns.items() if not isinstance(v, six.string_types)}
    if unsupported:
        raise ValueError(
            "Unsupported object matcher (expecting string): %s"
            % ", ".join("%s=%s" % (k, v) for k, v in unsupported.items())
        )

    # optimize simple patters
    starts_with = {k: v.rstrip("*") for k, v in patterns.items() if "*" not in v.rstrip("*") and "?" not in v}
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
    if parsed.scheme not in ("http", "https"):
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
        with open(filename, "rb", buffering=0) as f:
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
    return hash_text(text=text, seed=seed, hash_func="md5")


def crc32text(text, seed=1337):
    # type: (str, Union[int, str]) -> str
    """
    Return crc32 hash of a string
    Do not use this hash for security, if needed use something stronger like SHA2

    :param text: string to hash
    :param seed: use prefix seed for hashing
    :return: crc32 hex in string (32bits = 8 characters in hex)
    """
    return "{:08x}".format(crc32((str(seed) + str(text)).encode("utf-8")))


def hash_text(text, seed=1337, hash_func="md5"):
    # type: (str, Union[int, str], str) -> str
    """
    Return hash_func (md5/sha1/sha256/sha384/sha512) hash of a string

    :param text: string to hash
    :param seed: use prefix seed for hashing
    :param hash_func: hashing function. currently supported md5 sha256
    :return: hashed string
    """
    assert hash_func in ("md5", "sha256", "sha256", "sha384", "sha512")
    h = getattr(hashlib, hash_func)()
    h.update((str(seed) + str(text)).encode("utf-8"))
    return h.hexdigest()


def hash_dict(a_dict, seed=1337, hash_func="md5"):
    # type: (Dict, Union[int, str], str) -> str
    """
    Return hash_func (crc32/md5/sha1/sha256/sha384/sha512) hash of the dict values
    (dict must be JSON serializable)

    :param a_dict: a dictionary to hash
    :param seed: use prefix seed for hashing
    :param hash_func: hashing function. currently supported md5 sha256
    :return: hashed string
    """
    assert hash_func in ("crc32", "md5", "sha256", "sha256", "sha384", "sha512")
    repr_string = json.dumps(a_dict, sort_keys=True)
    if hash_func == "crc32":
        return crc32text(repr_string, seed=seed)
    else:
        return hash_text(repr_string, seed=seed, hash_func=hash_func)


def is_windows():
    """
    :return: True if currently running on windows OS
    """
    return sys.platform == "win32"


def format_size(size_in_bytes, binary=False, use_nonbinary_notation=False, use_b_instead_of_bytes=False):
    # type: (Union[int, float], bool, bool, bool) -> str
    """
    Return the size in human readable format (string)
    Matching humanfriendly.format_size outputs

    :param size_in_bytes: number of bytes
    :param binary: If `True` 1 Kb equals 1024 bytes, if False (default) 1 KB = 1000 bytes
    :param use_nonbinary_notation: Only applies if binary is `True`. If this is `True`,
        the binary scale (KiB, MiB etc.) will be replaced with the regular scale (KB, MB etc.)
    :param use_b_instead_of_bytes: If `True`, return the formatted size with `B` as the
        scale instead of `byte(s)` (when applicable)
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
    if size == 1 and not use_b_instead_of_bytes:
        return "{} byte".format(int(size))
    k = 1024 if binary else 1000
    scale = (
        ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB"]
        if (binary and not use_nonbinary_notation)
        else ["bytes", "KB", "MB", "GB", "TB", "PB"]
    )
    if use_b_instead_of_bytes:
        scale[0] = "B"
    for i, m in enumerate(scale):
        if size < k ** (i + 1) or i == len(scale) - 1:
            return (
                ("{:.2f}".format(size / (k**i)).rstrip("0").rstrip(".") if i > 0 else "{}".format(int(size)))
                + " "
                + m
            )
    # we should never get here
    return "{} {}".format(int(size), scale[0])


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
        for token in re.split(r"(\d+(?:\.\d+)?)", text):
            token = token.strip()
            if re.match(r"\d+\.\d+", token):
                tokenized_input.append(float(token))
            elif token.isdigit():
                tokenized_input.append(int(token))
            elif token:
                tokenized_input.append(token)
        return tokenized_input

    tokens = tokenize(str(size))
    if tokens and isinstance(tokens[0], (int, float)):
        disk_size_units_b = (
            ("B", "bytes"),
            ("KiB", "kibibyte"),
            ("MiB", "mebibyte"),
            ("GiB", "gibibyte"),
            ("TiB", "tebibyte"),
            ("PiB", "pebibyte"),
        )
        disk_size_units_d = (
            ("B", "bytes"),
            ("KB", "kilobyte"),
            ("MB", "megabyte"),
            ("GB", "gigabyte"),
            ("TB", "terabyte"),
            ("PB", "petabyte"),
        )
        disk_size_units_b = [(1024**i, s[0], s[1]) for i, s in enumerate(disk_size_units_b)]
        k = 1024 if binary else 1000
        disk_size_units_d = [(k**i, s[0], s[1]) for i, s in enumerate(disk_size_units_d)]
        disk_size_units = (disk_size_units_b + disk_size_units_d) if binary else (disk_size_units_d + disk_size_units_b)

        # Get the normalized unit (if any) from the tokenized input.
        normalized_unit = tokens[1].lower() if len(tokens) == 2 and isinstance(tokens[1], str) else ""
        # If the input contains only a number, it's assumed to be the number of
        # bytes. The second token can also explicitly reference the unit bytes.
        if len(tokens) == 1 or normalized_unit.startswith("b"):
            return int(tokens[0])
        # Otherwise we expect two tokens: A number and a unit.
        if normalized_unit:
            # Convert plural units to singular units, for details:
            # https://github.com/xolox/python-humanfriendly/issues/26
            normalized_unit = normalized_unit.rstrip("s")
            for k, low, high in disk_size_units:
                # First we check for unambiguous symbols (KiB, MiB, GiB, etc)
                # and names (kibibyte, mebibyte, gibibyte, etc) because their
                # handling is always the same.
                if normalized_unit in (low.lower(), high.lower()):
                    return int(tokens[0] * k)
                # Now we will deal with ambiguous prefixes (K, M, G, etc),
                # symbols (KB, MB, GB, etc) and names (kilobyte, megabyte,
                # gigabyte, etc) according to the caller's preference.
                if normalized_unit in (low.lower(), high.lower()) or normalized_unit.startswith(low.lower()):
                    return int(tokens[0] * k)

    raise ValueError("Failed to parse size! (input {} was tokenized as {})".format(size, tokens))


def get_common_path(list_of_files):
    # type: (Sequence[Union[str, Path]]) -> Optional[str]
    """
    Return the common path of a list of files

    :param list_of_files: list of files (str or Path objects)
    :return: Common path string (always absolute) or None if common path could not be found
    """
    if not list_of_files:
        return None

    # a single file has its parent as common path
    if len(list_of_files) == 1:
        return Path(list_of_files[0]).absolute().parent.as_posix()

    # find common path to support folder structure inside zip
    common_path_parts = Path(list_of_files[0]).absolute().parts
    for f in list_of_files:
        f_parts = Path(f).absolute().parts
        num_p = min(len(f_parts), len(common_path_parts))
        if f_parts[:num_p] == common_path_parts[:num_p]:
            common_path_parts = common_path_parts[:num_p]
            continue
        num_p = min([i for i, (a, b) in enumerate(zip(common_path_parts[:num_p], f_parts[:num_p])) if a != b] or [-1])
        # no common path, break
        if num_p < 0:
            common_path_parts = []
            break
        # update common path
        common_path_parts = common_path_parts[:num_p]

    if common_path_parts:
        common_path = Path()
        for f in common_path_parts:
            common_path /= f
        return common_path.as_posix()

    return None


def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory


def safe_extract(tar, path=".", members=None, numeric_owner=False):
    """Tarfile member sanitization (addresses CVE-2007-4559)"""
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path, members, numeric_owner=numeric_owner)
