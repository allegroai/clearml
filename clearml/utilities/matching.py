import fnmatch
from typing import Union


def matches_any_wildcard(pattern, wildcards):
    # type: (str, Union[str, list]) -> bool
    """
    Checks if given pattern matches any supplied wildcard

    :param pattern: pattern to check
    :param wildcards: wildcards to check against

    :return: True if pattern matches any wildcard and False otherwise
    """
    if isinstance(wildcards, str):
        wildcards = [wildcards]
    for wildcard in wildcards:
        if fnmatch.fnmatch(pattern, wildcard):
            return True
    return False
