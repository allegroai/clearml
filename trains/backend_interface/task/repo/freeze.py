import sys

from .util import get_command_output


def pip_freeze():
    try:
        return get_command_output([sys.executable, "-m", "pip", "freeze"]).splitlines()
    except Exception as ex:
        print('Failed calling "pip freeze": {}'.format(str(ex)))
    return []
