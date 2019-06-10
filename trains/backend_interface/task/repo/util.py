import os
from subprocess import check_output


def get_command_output(command, path=None):
    """
    Run a command and return its output
    :raises CalledProcessError: when command execution fails
    :raises UnicodeDecodeError: when output decoding fails
    """
    with open(os.devnull, "wb") as devnull:
        return check_output(command, cwd=path, stderr=devnull).decode().strip()
