from pathlib2 import Path
from fnmatch import fnmatch
from typing import Union


def matches_any_wildcard(path, wildcards, recursive=True):
    # type: (str, Union[str, list], bool) -> bool
    """
    Checks if given pattern matches any supplied wildcard

    :param path: path to check
    :param wildcards: wildcards to check against
    :param recursive: whether or not the check is recursive. Default: True
        E.g. for path='directory/file.ext' and wildcards='*.ext',
        recursive=False will return False, but recursive=True will
        return True

    :return: True if the path matches any wildcard and False otherwise
    """
    path = Path(path).as_posix()
    if wildcards is None:
        wildcards = ["*"]
    if not isinstance(wildcards, list):
        wildcards = [wildcards]
    wildcards = [str(w) for w in wildcards]
    if not recursive:
        path = path.split("/")
    for wildcard in wildcards:
        if not recursive:
            wildcard = wildcard.split("/")
            matched = True
            if len(path) != len(wildcard):
                continue
            for path_segment, wildcard_segment in zip(path, wildcard):
                if not fnmatch(path_segment, wildcard_segment):
                    matched = False
                    break
            if matched:
                return True
        else:
            wildcard_file = wildcard.split("/")[-1]
            wildcard_dir = wildcard[: -len(wildcard_file)] + "*"
            if fnmatch(path, wildcard_dir) and fnmatch("/" + path, "*/" + wildcard_file):
                return True
