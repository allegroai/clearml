import sys

from .util import get_command_output


def pip_freeze():
    req_lines = []
    local_packages = []
    try:
        req_lines = get_command_output([sys.executable, "-m", "pip", "freeze"]).splitlines()
        # fix "package @ file://" from pip freeze to "package"
        for i, r in enumerate(req_lines):
            parts = r.split('@', 1)
            if parts and len(parts) == 2 and parts[1].strip().lower().startswith('file://'):
                req_lines[i] = parts[0]
                local_packages.append((i, parts[0].strip()))
        # if we found local packages, at least get their versions (using pip list)
        if local_packages:
            # noinspection PyBroadException
            try:
                list_lines = get_command_output(
                    [sys.executable, "-m", "pip", "list", "--format", "freeze"]).splitlines()
                for index, name in local_packages:
                    line = [r for r in list_lines if r.strip().startswith(name+'==')]
                    if not line:
                        continue
                    line = line[0]
                    req_lines[index] = line.strip()
            except Exception:
                pass
    except Exception as ex:
        print('Failed calling "pip freeze": {}'.format(str(ex)))
    return req_lines
