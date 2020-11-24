import os
from subprocess import check_output

from furl import furl


def get_command_output(command, path=None, strip=True):
    """
    Run a command and return its output
    :raises CalledProcessError: when command execution fails
    :raises UnicodeDecodeError: when output decoding fails
    """
    with open(os.devnull, "wb") as devnull:
        result = check_output(command, cwd=path, stderr=devnull).decode()
        return result.strip() if strip else result


def remove_user_pass_from_url(url):
    # remove user / password, if we have it embedded in the git repo
    url = str(url)
    # noinspection PyBroadException
    try:
        url = furl(url).remove(username=True, password=True).tostr()
    except ValueError:
        # check if for any reason we have two @
        # (e.g. ssh://user@abc.com@domain.com/path or ssh://user@abc.com:pass@domain.com/path)
        if len(url.split('@')) >= 3:
            # noinspection PyBroadException
            try:
                url = furl(url.replace('@', '_', 1)).remove(username=True, password=True).tostr()
            except Exception:
                pass
    except Exception:
        pass
    return url
