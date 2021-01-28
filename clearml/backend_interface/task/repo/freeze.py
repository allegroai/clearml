import sys
import os
import json

from .util import get_command_output


def pip_freeze(combine_conda_with_pip=False):
    req_lines = []
    local_packages = []
    conda_lines = []
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix and not conda_prefix.endswith(os.path.sep):
        conda_prefix += os.path.sep

    if conda_prefix and sys.executable.startswith(conda_prefix):
        pip_lines = get_command_output([sys.executable, "-m", "pip", "freeze"]).splitlines()
        conda_packages_json = get_command_output(['conda', 'list', '--json'])
        conda_packages_json = json.loads(conda_packages_json)
        for r in conda_packages_json:
            # check if this is a pypi package, if it is, leave it outside
            if not r.get('channel') or r.get('channel') == 'pypi':
                name = (r['name'].replace('-', '_'), r['name'])
                pip_req_line = [pip_l for pip_l in pip_lines
                                if pip_l.split('==', 1)[0].strip() in name or pip_l.split('@', 1)[0].strip() in name]
                if pip_req_line and \
                        ('@' not in pip_req_line[0] or
                         not pip_req_line[0].split('@', 1)[1].strip().startswith('file://')):
                    req_lines.append(pip_req_line[0])
                    continue

                req_lines.append('{}=={}'.format(name[0], r['version']) if r.get('version') else '{}'.format(name[0]))
                continue

            # check if we have it in our required packages
            name = r['name']
            # hack support pytorch/torch different naming convention
            if name == 'pytorch':
                name = 'torch'
            # skip over packages with _
            if name.startswith('_'):
                continue
            conda_lines.append('{}=={}'.format(name, r['version']) if r.get('version') else '{}'.format(name))
        # make sure we see the conda packages, put them into the pip as well
        if combine_conda_with_pip and conda_lines:
            req_lines += ['', '# Conda Packages', ''] + conda_lines
    else:
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

    return "\n".join(req_lines), "\n".join(conda_lines)
