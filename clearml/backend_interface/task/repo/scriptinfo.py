import os
import sys
from copy import copy
from datetime import datetime
from functools import partial
from tempfile import mkstemp, gettempdir

import attr
import logging
import json
from pathlib2 import Path
from threading import Thread, Event

from .util import get_command_output, remove_user_pass_from_url
from ....backend_api import Session
from ....config import deferred_config, VCS_WORK_DIR
from ....debugging import get_logger
from .detectors import GitEnvDetector, GitDetector, HgEnvDetector, HgDetector, Result as DetectionResult


class ScriptInfoError(Exception):
    pass


class ScriptRequirements(object):
    _detailed_import_report = deferred_config('development.detailed_import_report', False)
    _max_requirements_size = 512 * 1024
    _packages_remove_version = ('setuptools', )
    _ignore_packages = set()

    @classmethod
    def _get_logger(cls):
        return get_logger("Repository Detection")

    def __init__(self, root_folder):
        self._root_folder = root_folder

    def get_requirements(self, entry_point_filename=None, add_missing_installed_packages=False,
                         detailed_req_report=None):
        # noinspection PyBroadException
        try:
            from ....utilities.pigar.reqs import get_installed_pkgs_detail
            from ....utilities.pigar.__main__ import GenerateReqs
            installed_pkgs = self._remove_package_versions(
                get_installed_pkgs_detail(), self._packages_remove_version)
            gr = GenerateReqs(save_path='', project_path=self._root_folder, installed_pkgs=installed_pkgs,
                              ignores=['.git', '.hg', '.idea', '__pycache__', '.ipynb_checkpoints',
                                       'site-packages', 'dist-packages'])
            reqs, try_imports, guess, local_pks = gr.extract_reqs(
                module_callback=ScriptRequirements.add_trains_used_packages, entry_point_filename=entry_point_filename)
            if add_missing_installed_packages and guess:
                for k in guess:
                    if k not in reqs:
                        reqs[k] = guess[k]
            return self.create_requirements_txt(reqs, local_pks, detailed=detailed_req_report)
        except Exception as ex:
            self._get_logger().warning("Failed auto-generating package requirements: {}".format(ex))
            return '', ''

    @staticmethod
    def add_trains_used_packages(modules):
        # hack: forcefully insert storage modules if we have them
        # noinspection PyBroadException
        try:
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            import boto3  # noqa: F401
            modules.add('boto3', 'clearml.storage', 0)
        except Exception:
            pass
        # noinspection PyBroadException
        try:
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            from google.cloud import storage  # noqa: F401
            modules.add('google_cloud_storage', 'clearml.storage', 0)
        except Exception:
            pass
        # noinspection PyBroadException
        try:
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            from azure.storage.blob import ContentSettings  # noqa: F401
            modules.add('azure_storage_blob', 'clearml.storage', 0)
        except Exception:
            pass

        # bugfix, replace sklearn with scikit-learn name
        if 'sklearn' in modules:
            sklearn = modules.pop('sklearn', {})
            for fname, lines in sklearn.items():
                modules.add('scikit_learn', fname, lines)

        # if we have torch and it supports tensorboard, we should add that as well
        # (because it will not be detected automatically)
        if 'torch' in modules and 'tensorboard' not in modules and 'tensorboardX' not in modules:
            # noinspection PyBroadException
            try:
                # see if this version of torch support tensorboard
                # noinspection PyPackageRequirements,PyUnresolvedReferences
                import torch.utils.tensorboard  # noqa: F401
                # noinspection PyPackageRequirements,PyUnresolvedReferences
                import tensorboard  # noqa: F401
                modules.add('tensorboard', 'torch', 0)
            except Exception:
                pass

        # remove setuptools, we should not specify this module version. It is installed by default
        if 'setuptools' in modules:
            modules.pop('setuptools', {})

        # add forced requirements:
        # noinspection PyBroadException
        try:
            from ..task import Task
            # noinspection PyProtectedMember
            for package, version in Task._force_requirements.items():
                modules.add(package, 'clearml', 0)
        except Exception:
            pass

        return modules

    @staticmethod
    def create_requirements_txt(reqs, local_pks=None, detailed=None):
        # write requirements.txt
        if detailed is None:
            detailed = ScriptRequirements._detailed_import_report

        # noinspection PyBroadException
        try:
            conda_requirements = ''
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix and not conda_prefix.endswith(os.path.sep):
                conda_prefix += os.path.sep
            if conda_prefix and sys.executable.startswith(conda_prefix):
                conda_packages_json = get_command_output(['conda', 'list', '--json'])
                conda_packages_json = json.loads(conda_packages_json)
                reqs_lower = {k.lower(): (k, v) for k, v in reqs.items()}
                for r in conda_packages_json:
                    # the exception is cudatoolkit which we want to log anyhow
                    if r.get('name') == 'cudatoolkit' and r.get('version'):
                        conda_requirements += '{0} {1} {2}\n'.format(r.get('name'), '==', r.get('version'))
                        continue
                    # check if this is a pypi package, if it is, leave it outside
                    if not r.get('channel') or r.get('channel') == 'pypi':
                        continue
                    # check if we have it in our required packages
                    name = r['name'].lower()
                    # hack support pytorch/torch different naming convention
                    if name == 'pytorch':
                        name = 'torch'
                    k, v = None, None
                    if name in reqs_lower:
                        k, v = reqs_lower.get(name, (None, None))
                    else:
                        name = name.replace('-', '_')
                        if name in reqs_lower:
                            k, v = reqs_lower.get(name, (None, None))

                    if k and v is not None:
                        if v.version:
                            conda_requirements += '{0} {1} {2}\n'.format(k, '==', v.version)
                        else:
                            conda_requirements += '{0}\n'.format(k)
        except Exception:
            conda_requirements = ''

        # add forced requirements:
        forced_packages = {}
        ignored_packages = ScriptRequirements._ignore_packages
        # noinspection PyBroadException
        try:
            from ..task import Task
            # noinspection PyProtectedMember
            forced_packages = copy(Task._force_requirements)
            # noinspection PyProtectedMember
            ignored_packages = Task._ignore_requirements | ignored_packages
        except Exception:
            pass

        # python version header
        requirements_txt = '# Python ' + sys.version.replace('\n', ' ').replace('\r', ' ') + '\n'

        if local_pks:
            requirements_txt += '\n# Local modules found - skipping:\n'
            for k, v in local_pks.sorted_items():
                if v.version:
                    requirements_txt += '# {0} == {1}\n'.format(k, v.version)
                else:
                    requirements_txt += '# {0}\n'.format(k)

        # requirement summary
        requirements_txt += '\n'
        for k, v in reqs.sorted_items():
            if k in ignored_packages or k.lower() in ignored_packages:
                continue
            version = v.version if v else None
            if k in forced_packages:
                forced_version = forced_packages.pop(k, None)
                if forced_version is not None:
                    version = forced_version
            # requirements_txt += ''.join(['# {0}\n'.format(c) for c in v.comments.sorted_items()])
            requirements_txt += ScriptRequirements._make_req_line(k, version or None)

        # add forced requirements that we could not find installed on the system
        for k in sorted(forced_packages.keys()):
            requirements_txt += ScriptRequirements._make_req_line(k, forced_packages.get(k))

        requirements_txt_packages_only = requirements_txt
        if detailed:
            requirements_txt_packages_only = \
                requirements_txt + '\n# Skipping detailed import analysis, it is too large\n'

            # requirements details (in comments)
            requirements_txt += '\n' + \
                                '# Detailed import analysis\n' \
                                '# **************************\n'

            if local_pks:
                for k, v in local_pks.sorted_items():
                    requirements_txt += '\n'
                    requirements_txt += '# IMPORT LOCAL PACKAGE {0}\n'.format(k)
                    requirements_txt += ''.join(['# {0}\n'.format(c) for c in v.comments.sorted_items()])

            for k, v in reqs.sorted_items():
                if not v:
                    continue
                requirements_txt += '\n'
                if k == '-e':
                    requirements_txt += '# IMPORT PACKAGE {0} {1}\n'.format(k, v.version)
                else:
                    requirements_txt += '# IMPORT PACKAGE {0}\n'.format(k)
                requirements_txt += ''.join(['# {0}\n'.format(c) for c in v.comments.sorted_items()])

        # make sure we do not exceed the size a size limit
        return (requirements_txt if len(requirements_txt) < ScriptRequirements._max_requirements_size
                else requirements_txt_packages_only,
                conda_requirements)

    @staticmethod
    def _make_req_line(k, version):
        requirements_txt = ''
        if k == '-e' and version:
            requirements_txt += '{0}\n'.format(version)
        elif k.startswith('-e '):
            requirements_txt += '{0} {1}\n'.format(k.replace('-e ', '', 1), version or '')
        elif version and str(version or ' ').strip()[0].isdigit():
            requirements_txt += '{0} {1} {2}\n'.format(k, '==', version)
        elif version and str(version).strip():
            requirements_txt += '{0} {1}\n'.format(k, version)
        else:
            requirements_txt += '{0}\n'.format(k)
        return requirements_txt

    @staticmethod
    def _remove_package_versions(installed_pkgs, package_names_to_remove_version):
        installed_pkgs = {k: (v[0], None if str(k) in package_names_to_remove_version else v[1])
                          for k, v in installed_pkgs.items()}

        return installed_pkgs


class _JupyterObserver(object):
    _thread = None
    _exit_event = Event()
    _sync_event = Event()
    _sample_frequency = 30.
    _first_sample_frequency = 3.
    _jupyter_history_logger = None
    _store_notebook_artifact = deferred_config('development.store_jupyter_notebook_artifact', True)

    @classmethod
    def _get_logger(cls):
        return get_logger("Repository Detection")

    @classmethod
    def observer(cls, jupyter_notebook_filename, log_history):
        if cls._thread is not None:
            # order of signaling is important!
            cls._exit_event.set()
            cls._sync_event.set()
            cls._thread.join()

        if log_history and cls._jupyter_history_logger is None:
            cls._jupyter_history_logger = _JupyterHistoryLogger()
            cls._jupyter_history_logger.hook()

        cls._sync_event.clear()
        cls._exit_event.clear()
        cls._thread = Thread(target=cls._daemon, args=(jupyter_notebook_filename, ))
        cls._thread.daemon = True
        cls._thread.start()

    @classmethod
    def signal_sync(cls, *_, **__):
        cls._sync_event.set()

    @classmethod
    def close(cls):
        if not cls._thread:
            return
        cls._exit_event.set()
        cls._sync_event.set()
        cls._thread.join()
        cls._thread = None

    @classmethod
    def _daemon(cls, jupyter_notebook_filename):
        from clearml import Task

        # load jupyter notebook package
        # noinspection PyBroadException
        try:
            # noinspection PyPackageRequirements
            from nbconvert.exporters.script import ScriptExporter
            _script_exporter = ScriptExporter()
        except Exception as ex:
            cls._get_logger().warning('Could not read Jupyter Notebook: {}'.format(ex))
            return
        # load pigar
        # noinspection PyBroadException
        try:
            from ....utilities.pigar.reqs import get_installed_pkgs_detail, file_import_modules
            from ....utilities.pigar.modules import ReqsModules
            from ....utilities.pigar.log import logger
            logger.setLevel(logging.WARNING)
        except Exception:
            file_import_modules = None
        # load IPython
        # noinspection PyBroadException
        try:
            # noinspection PyPackageRequirements
            from IPython import get_ipython
        except Exception:
            # should not happen
            get_ipython = None

        # setup local notebook files
        if jupyter_notebook_filename:
            notebook = Path(jupyter_notebook_filename)
            local_jupyter_filename = jupyter_notebook_filename
        else:
            notebook = None
            fd, local_jupyter_filename = mkstemp(suffix='.ipynb')
            os.close(fd)
        last_update_ts = None
        counter = 0
        prev_script_hash = None

        # noinspection PyBroadException
        try:
            from ....version import __version__
            our_module = cls.__module__.split('.')[0], __version__
        except Exception:
            our_module = None

        # noinspection PyBroadException
        try:
            import re
            replace_ipython_pattern = re.compile(r'\n([ \t]*)get_ipython\(\)')
            replace_ipython_display_pattern = re.compile(r'\n([ \t]*)display\(')
        except Exception:
            replace_ipython_pattern = None
            replace_ipython_display_pattern = None

        # main observer loop, check if we need to exit
        while not cls._exit_event.wait(timeout=0.):
            # wait for timeout or sync event
            cls._sync_event.wait(cls._sample_frequency if counter else cls._first_sample_frequency)

            cls._sync_event.clear()
            counter += 1
            # noinspection PyBroadException
            try:
                # if there is no task connected, do nothing
                task = Task.current_task()
                if not task:
                    continue

                script_code = None
                fmodules = None
                current_cell = None
                # if we have a local file:
                if notebook:
                    if not notebook.exists():
                        continue
                    # check if notebook changed
                    if last_update_ts is not None and notebook.stat().st_mtime - last_update_ts <= 0:
                        continue
                    last_update_ts = notebook.stat().st_mtime
                else:
                    # serialize notebook to a temp file
                    if cls._jupyter_history_logger:
                        script_code, current_cell = cls._jupyter_history_logger.history_to_str()
                    else:
                        # noinspection PyBroadException
                        try:
                            # noinspection PyBroadException
                            try:
                                os.unlink(local_jupyter_filename)
                            except Exception:
                                pass
                            get_ipython().run_line_magic('history', '-t -f {}'.format(local_jupyter_filename))
                            with open(local_jupyter_filename, 'r') as f:
                                script_code = f.read()
                            # load the modules
                            from ....utilities.pigar.modules import ImportedModules
                            fmodules = ImportedModules()
                            for nm in set([str(m).split('.')[0] for m in sys.modules]):
                                fmodules.add(nm, 'notebook', 0)
                        except Exception:
                            continue

                # get notebook python script
                if script_code is None and local_jupyter_filename:
                    script_code, _ = _script_exporter.from_filename(local_jupyter_filename)
                    if cls._store_notebook_artifact:
                        # also upload the jupyter notebook as artifact
                        task.upload_artifact(
                            name='notebook',
                            artifact_object=Path(local_jupyter_filename),
                            preview='See `notebook preview` artifact',
                            metadata={'UPDATE': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')},
                            wait_on_upload=True,
                        )
                        # noinspection PyBroadException
                        try:
                            from nbconvert.exporters import HTMLExporter  # noqa
                            html, _ = HTMLExporter().from_filename(filename=local_jupyter_filename)
                            local_html = Path(gettempdir()) / 'notebook_{}.html'.format(task.id)
                            with open(local_html.as_posix(), 'wt', encoding="utf-8") as f:
                                f.write(html)
                            task.upload_artifact(
                                name='notebook preview', artifact_object=local_html,
                                preview='Click `FILE PATH` link',
                                metadata={'UPDATE': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')},
                                delete_after_upload=True,
                                wait_on_upload=True,
                            )
                        except Exception:
                            pass

                current_script_hash = hash(script_code + (current_cell or ''))
                if prev_script_hash and prev_script_hash == current_script_hash:
                    continue

                # remove ipython direct access from the script code
                # we will not be able to run them anyhow
                if replace_ipython_pattern:
                    script_code = replace_ipython_pattern.sub(r'\n# \g<1>get_ipython()', script_code)
                if replace_ipython_display_pattern:
                    script_code = replace_ipython_display_pattern.sub(r'\n\g<1>print(', script_code)

                requirements_txt = ''
                conda_requirements = ''
                # parse jupyter python script and prepare pip requirements (pigar)
                # if backend supports requirements
                if file_import_modules and Session.check_min_api_version('2.2'):
                    if fmodules is None:
                        fmodules, _ = file_import_modules(
                            notebook.parts[-1] if notebook else 'notebook', script_code)
                        if current_cell:
                            cell_fmodules, _ = file_import_modules(
                                notebook.parts[-1] if notebook else 'notebook', current_cell)
                            # noinspection PyBroadException
                            try:
                                fmodules |= cell_fmodules
                            except Exception:
                                pass
                    # add current cell to the script
                    if current_cell:
                        script_code += '\n' + current_cell
                    fmodules = ScriptRequirements.add_trains_used_packages(fmodules)
                    # noinspection PyUnboundLocalVariable
                    installed_pkgs = get_installed_pkgs_detail()
                    # make sure we are in installed packages
                    if our_module and (our_module[0] not in installed_pkgs):
                        installed_pkgs[our_module[0]] = our_module

                    # noinspection PyUnboundLocalVariable
                    reqs = ReqsModules()
                    for name in fmodules:
                        if name in installed_pkgs:
                            pkg_name, version = installed_pkgs[name]
                            reqs.add(pkg_name, version, fmodules[name])
                    requirements_txt, conda_requirements = ScriptRequirements.create_requirements_txt(reqs)

                # update script
                prev_script_hash = current_script_hash
                data_script = task.data.script
                data_script.diff = script_code
                data_script.requirements = {'pip': requirements_txt, 'conda': conda_requirements}
                # noinspection PyProtectedMember
                task._update_script(script=data_script)
                # update requirements
                # noinspection PyProtectedMember
                task._update_requirements(requirements=requirements_txt)
            except Exception:
                pass


class ScriptInfo(object):
    max_diff_size_bytes = 500000

    plugins = [GitEnvDetector(), HgEnvDetector(), HgDetector(), GitDetector()]
    """ Script info detection plugins, in order of priority """

    @classmethod
    def _get_logger(cls):
        return get_logger("Repository Detection")

    @classmethod
    def _jupyter_install_post_store_hook(cls, jupyter_notebook_filename, log_history=False):
        # noinspection PyBroadException
        try:
            if 'IPython' in sys.modules:
                # noinspection PyPackageRequirements
                from IPython import get_ipython
                if get_ipython():
                    _JupyterObserver.observer(jupyter_notebook_filename, log_history)
                    get_ipython().events.register('pre_run_cell', _JupyterObserver.signal_sync)
                    if log_history:
                        get_ipython().events.register('post_run_cell', _JupyterObserver.signal_sync)
        except Exception:
            pass

    @classmethod
    def _get_jupyter_notebook_filename(cls):
        # check if we are running in vscode, we have the jupyter notebook defined:
        if 'IPython' in sys.modules:
            # noinspection PyBroadException
            try:
                from IPython import get_ipython  # noqa
                ip = get_ipython()
                # vscode-jupyter PR #8531 added this variable
                local_ipynb_file = ip.__dict__.get('user_ns', {}).get('__vsc_ipynb_file__') if ip else None
                if local_ipynb_file:
                    # now replace the .ipynb with .py
                    # we assume we will have that file available for monitoring
                    local_ipynb_file = Path(local_ipynb_file)
                    script_entry_point = local_ipynb_file.with_suffix('.py').as_posix()

                    # install the post store hook,
                    # notice that if we do not have a local file we serialize/write every time the entire notebook
                    cls._jupyter_install_post_store_hook(local_ipynb_file.as_posix(), log_history=False)

                    return script_entry_point
            except Exception:
                pass

        if not (sys.argv[0].endswith(os.path.sep + 'ipykernel_launcher.py') or
                sys.argv[0].endswith(os.path.join(os.path.sep, 'ipykernel', '__main__.py'))) \
                or len(sys.argv) < 3 or not sys.argv[2].endswith('.json'):
            return None

        server_info = None

        # we can safely assume that we can import the notebook package here
        # noinspection PyBroadException
        try:
            # noinspection PyBroadException
            try:
                # noinspection PyPackageRequirements
                from notebook.notebookapp import list_running_servers  # <= Notebook v6
            except Exception:
                # noinspection PyPackageRequirements
                from jupyter_server.serverapp import list_running_servers

            import requests
            current_kernel = sys.argv[2].split(os.path.sep)[-1].replace('kernel-', '').replace('.json', '')

            # noinspection PyBroadException
            try:
                server_info = next(list_running_servers())
            except Exception:
                # on some jupyter notebook versions this function can crash on parsing the json file,
                # we will parse it manually here
                # noinspection PyPackageRequirements
                import ipykernel
                from glob import glob
                import json
                for f in glob(os.path.join(os.path.dirname(ipykernel.get_connection_file()), '??server-*.json')):
                    # noinspection PyBroadException
                    try:
                        with open(f, 'r') as json_data:
                            server_info = json.load(json_data)
                    except Exception:
                        server_info = None
                    if server_info:
                        break

            cookies = None
            password = None
            if server_info and server_info.get('password'):
                # we need to get the password
                from ....config import config
                password = config.get('development.jupyter_server_password', '')
                if not password:
                    cls._get_logger().warning(
                        'Password protected Jupyter Notebook server was found! '
                        'Add `sdk.development.jupyter_server_password=<jupyter_password>` to ~/clearml.conf')
                    return os.path.join(os.getcwd(), 'error_notebook_not_found.py')

                r = requests.get(url=server_info['url'] + 'login')
                cookies = {'_xsrf': r.cookies.get('_xsrf', '')}
                r = requests.post(server_info['url'] + 'login?next', cookies=cookies,
                                  data={'_xsrf': cookies['_xsrf'], 'password': password})
                cookies.update(r.cookies)

            auth_token = server_info.get('token') or os.getenv('JUPYTERHUB_API_TOKEN') or ''
            try:
                r = requests.get(
                    url=server_info['url'] + 'api/sessions', cookies=cookies,
                    headers={'Authorization': 'token {}'.format(auth_token), })
            except requests.exceptions.SSLError:
                # disable SSL check warning
                from urllib3.exceptions import InsecureRequestWarning
                # noinspection PyUnresolvedReferences
                requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
                # fire request
                r = requests.get(
                    url=server_info['url'] + 'api/sessions', cookies=cookies,
                    headers={'Authorization': 'token {}'.format(auth_token), }, verify=False)
                # enable SSL check warning
                import warnings
                warnings.simplefilter('default', InsecureRequestWarning)

            # send request to the jupyter server
            try:
                r.raise_for_status()
            except Exception as ex:
                cls._get_logger().warning('Failed accessing the jupyter server{}: {}'.format(
                    ' [password={}]'.format(password) if server_info.get('password') else '', ex))
                return os.path.join(os.getcwd(), 'error_notebook_not_found.py')

            notebooks = r.json()

            cur_notebook = None
            for n in notebooks:
                if n['kernel']['id'] == current_kernel:
                    cur_notebook = n
                    break

            notebook_path = cur_notebook['notebook'].get('path', '')
            notebook_name = cur_notebook['notebook'].get('name', '')

            is_google_colab = False
            # check if this is google.colab, then there is no local file
            # noinspection PyBroadException
            try:
                # noinspection PyPackageRequirements
                from IPython import get_ipython
                if get_ipython() and 'google.colab' in get_ipython().extension_manager.loaded:
                    is_google_colab = True
            except Exception:
                pass

            if is_google_colab:
                script_entry_point = str(notebook_name or 'notebook').replace(
                    '>', '_').replace('<', '_').replace('.ipynb', '.py')
                if not script_entry_point.lower().endswith('.py'):
                    script_entry_point += '.py'
                local_ipynb_file = None
            else:
                # always slash, because this is from uri (so never backslash not even on windows)
                entry_point_filename = notebook_path.split('/')[-1]

                # now we should try to find the actual file
                entry_point = (Path.cwd() / entry_point_filename).absolute()
                if not entry_point.is_file():
                    entry_point = (Path.cwd() / notebook_path).absolute()

                # fix for VSCode pushing uuid at the end of the notebook name.
                if not entry_point.exists():
                    # noinspection PyBroadException
                    try:
                        alternative_entry_point = '-'.join(entry_point_filename.split('-')[:-5])+'.ipynb'
                        # now we should try to find the actual file
                        entry_point_alternative = (Path.cwd() / alternative_entry_point).absolute()
                        if not entry_point_alternative.is_file():
                            entry_point_alternative = (Path.cwd() / alternative_entry_point).absolute()

                        # If we found it replace it
                        if entry_point_alternative.exists():
                            entry_point = entry_point_alternative
                    except Exception as ex:
                        cls._get_logger().warning('Failed accessing jupyter notebook {}: {}'.format(notebook_path, ex))

                # get local ipynb for observer
                local_ipynb_file = entry_point.as_posix()

                # now replace the .ipynb with .py
                # we assume we will have that file available with the Jupyter notebook plugin
                entry_point = entry_point.with_suffix('.py')

                script_entry_point = entry_point.as_posix()

            # install the post store hook,
            # notice that if we do not have a local file we serialize/write every time the entire notebook
            cls._jupyter_install_post_store_hook(local_ipynb_file, is_google_colab)

            return script_entry_point
        except Exception:
            return None

    @classmethod
    def _get_entry_point(cls, repo_root, script_path):
        repo_root = Path(repo_root).absolute()

        try:
            # Use os.path.relpath as it calculates up dir movements (../)
            entry_point = os.path.relpath(
                str(script_path), str(cls._get_working_dir(repo_root, return_abs=True)))
        except ValueError:
            # Working directory not under repository root
            entry_point = script_path.relative_to(repo_root)

        return Path(entry_point).as_posix()

    @classmethod
    def _cwd(cls):
        # return the current working directory (solve for hydra changing it)
        # check if running with hydra
        if sys.modules.get('hydra'):
            # noinspection PyBroadException
            try:
                # noinspection PyPackageRequirements
                import hydra  # noqa
                return Path(hydra.utils.get_original_cwd()).absolute()
            except Exception:
                pass
        return Path.cwd().absolute()

    @classmethod
    def _get_working_dir(cls, repo_root, return_abs=False):
        # get the repository working directory (might be different from actual cwd)
        repo_root = Path(repo_root).absolute()
        cwd = cls._cwd()

        try:
            # do not change: test if we are under the repo root folder, it will throw an exception if we are not
            relative = cwd.relative_to(repo_root).as_posix()
            return cwd.as_posix() if return_abs else relative
        except ValueError:
            # Working directory not under repository root, default to repo root
            return repo_root.as_posix() if return_abs else '.'

    @classmethod
    def _absolute_path(cls, file_path, cwd):
        # return the absolute path, relative to a specific working directory (cwd)
        file_path = Path(file_path)
        if file_path.is_absolute():
            return file_path.as_posix()
        # Convert to absolute and squash 'path/../folder'
        return os.path.abspath((Path(cwd).absolute() / file_path).as_posix())

    @classmethod
    def _get_script_code(cls, script_path):
        # noinspection PyBroadException
        try:
            with open(script_path, 'r') as f:
                script_code = f.read()
            return script_code
        except Exception:
            pass
        return ''

    @classmethod
    def _get_script_info(
            cls, filepaths, check_uncommitted=True, create_requirements=True, log=None,
            uncommitted_from_remote=False, detect_jupyter_notebook=True,
            add_missing_installed_packages=False, detailed_req_report=None, force_single_script=False):
        jupyter_filepath = cls._get_jupyter_notebook_filename() if detect_jupyter_notebook else None
        if jupyter_filepath:
            scripts_path = [Path(os.path.normpath(jupyter_filepath)).absolute()]
        else:
            cwd = cls._cwd()
            scripts_path = [Path(cls._absolute_path(os.path.normpath(f), cwd)) for f in filepaths if f]
            scripts_path = [f for f in scripts_path if f.exists()]
            if not scripts_path:
                raise ScriptInfoError(
                    "Script file {} could not be found".format(filepaths)
                )

        scripts_dir = [f.parent for f in scripts_path]

        def _log(msg, *args, **kwargs):
            if not log:
                return
            log.warning(
                "Failed auto-detecting task repository: {}".format(
                    msg.format(*args, **kwargs)
                )
            )

        script_dir = scripts_dir[0]
        script_path = scripts_path[0]

        if force_single_script:
            plugin = None
        else:
            plugin = next((p for p in cls.plugins if p.exists(script_dir)), None)

        repo_info = DetectionResult()
        messages = []
        auxiliary_git_diff = None

        if not plugin:
            if log:
                log.info("No repository found, storing script code instead")
        else:
            try:
                repo_info = plugin.get_info(
                    str(script_dir), include_diff=check_uncommitted, diff_from_remote=uncommitted_from_remote)
            except SystemExit:
                raise
            except Exception as ex:
                _log("no info for {} ({})", scripts_dir, ex)
            else:
                if repo_info.is_empty():
                    _log("no info for {}", scripts_dir)

        repo_root = repo_info.root or script_dir
        if not plugin:
            working_dir = '.'
            entry_point = str(script_path.name)
        else:
            # allow to override the VCS working directory (notice relative to the git repo)
            # because we can have a sync folder on remote pycharm sessions
            # not syncing from the Git repo, but from a subfolder, so the pycharm plugin need to pass the override
            working_dir = VCS_WORK_DIR.get() if VCS_WORK_DIR.get() else cls._get_working_dir(repo_root)
            entry_point = cls._get_entry_point(repo_root, script_path)

        if check_uncommitted:
            diff = cls._get_script_code(script_path.as_posix()) \
                if not plugin or not repo_info.commit else repo_info.diff
            # make sure diff is not too big:
            if len(diff) > cls.max_diff_size_bytes:
                messages.append(
                    "======> WARNING! Git diff to large to store "
                    "({}kb), skipping uncommitted changes <======".format(len(diff)//1024))
                auxiliary_git_diff = diff
                diff = '# WARNING! git diff too large to store, clear this section to execute without it.\n' \
                       '# full git diff available in Artifacts/auxiliary_git_diff\n' \
                       '# Clear the section before enqueueing Task!\n'

        else:
            diff = ''
        # if this is not jupyter, get the requirements.txt
        requirements = ''
        conda_requirements = ''
        # create requirements if backend supports requirements
        # if jupyter is present, requirements will be created in the background, when saving a snapshot
        if not jupyter_filepath and Session.check_min_api_version('2.2'):
            script_requirements = ScriptRequirements(
                Path(repo_root).as_posix() if repo_info.url else script_path.as_posix())
            if create_requirements:
                requirements, conda_requirements = script_requirements.get_requirements(
                    entry_point_filename=script_path.as_posix()
                    if not repo_info.url and script_path.is_file() else None,
                    add_missing_installed_packages=add_missing_installed_packages,
                    detailed_req_report=detailed_req_report,
                )
        else:
            script_requirements = None

        script_info = dict(
            repository=remove_user_pass_from_url(repo_info.url),
            branch=repo_info.branch,
            version_num=repo_info.commit,
            entry_point=entry_point,
            working_dir=working_dir,
            diff=diff,
            requirements={'pip': requirements, 'conda': conda_requirements} if requirements else None,
            binary='python{}.{}'.format(sys.version_info.major, sys.version_info.minor),
            repo_root=repo_root,
            jupyter_filepath=jupyter_filepath,
        )

        # if repo_info.modified:
        #     messages.append(
        #         "======> WARNING! UNCOMMITTED CHANGES IN REPOSITORY {} <======".format(
        #             script_info.get("repository", "")
        #         )
        #     )

        if not any(script_info.values()):
            script_info = None

        return (ScriptInfoResult(script=script_info, warning_messages=messages, auxiliary_git_diff=auxiliary_git_diff),
                script_requirements)

    @classmethod
    def get(cls, filepaths=None, check_uncommitted=True, create_requirements=True, log=None,
            uncommitted_from_remote=False, detect_jupyter_notebook=True, add_missing_installed_packages=False,
            detailed_req_report=None, force_single_script=False):
        try:
            if not filepaths:
                filepaths = [sys.argv[0], ]
            return cls._get_script_info(
                filepaths=filepaths,
                check_uncommitted=check_uncommitted,
                create_requirements=create_requirements, log=log,
                uncommitted_from_remote=uncommitted_from_remote,
                detect_jupyter_notebook=detect_jupyter_notebook,
                add_missing_installed_packages=add_missing_installed_packages,
                detailed_req_report=detailed_req_report,
                force_single_script=force_single_script,
            )
        except SystemExit:
            pass
        except BaseException as ex:
            if log:
                log.warning("Failed auto-detecting task repository: {}".format(ex))
        return ScriptInfoResult(), None

    @classmethod
    def is_running_from_module(cls):
        # noinspection PyBroadException
        try:
            return '__main__' in sys.modules and vars(sys.modules['__main__'])['__package__']
        except Exception:
            return False

    @classmethod
    def detect_running_module(cls, script_dict):
        # noinspection PyBroadException
        try:
            # If this is jupyter, do not try to detect the running module, we know what we have.
            if script_dict.get('jupyter_filepath'):
                return script_dict

            if cls.is_running_from_module():
                argvs = ''
                git_root = os.path.abspath(str(script_dict['repo_root'])) if script_dict['repo_root'] else None
                for a in sys.argv[1:]:
                    if git_root and os.path.exists(a):
                        # check if common to project:
                        a_abs = os.path.abspath(a)
                        if os.path.commonpath([a_abs, git_root]) == git_root:
                            # adjust path relative to working dir inside git repo
                            a = ' ' + os.path.relpath(
                                a_abs, os.path.join(git_root, str(script_dict['working_dir'])))
                    argvs += ' {}'.format(a)

                # noinspection PyBroadException
                try:
                    module_name = vars(sys.modules['__main__'])['__spec__'].name
                except Exception:
                    module_name = vars(sys.modules['__main__'])['__package__']

                # update the script entry point to match the real argv and module call
                script_dict['entry_point'] = '-m {}{}'.format(module_name, (' ' + argvs) if argvs else '')
        except Exception:
            pass
        return script_dict

    @classmethod
    def close(cls):
        _JupyterObserver.close()


@attr.s
class ScriptInfoResult(object):
    script = attr.ib(default=None)
    warning_messages = attr.ib(factory=list)
    auxiliary_git_diff = attr.ib(default=None)


class _JupyterHistoryLogger(object):
    _reg_replace_ipython = r'\n([ \t]*)get_ipython\(\)'
    _reg_replace_magic = r'\n([ \t]*)%'
    _reg_replace_bang = r'\n([ \t]*)!'

    def __init__(self):
        self._exception_raised = False
        self._cells_code = {}
        self._counter = 0
        self._ip = None
        self._current_cell = None
        # noinspection PyBroadException
        try:
            import re
            self._replace_ipython_pattern = re.compile(self._reg_replace_ipython)
            self._replace_magic_pattern = re.compile(self._reg_replace_magic)
            self._replace_bang_pattern = re.compile(self._reg_replace_bang)
        except Exception:
            self._replace_ipython_pattern = None
            self._replace_magic_pattern = None
            self._replace_bang_pattern = None

    def hook(self, ip=None):
        if not ip:
            # noinspection PyBroadException
            try:
                # noinspection PyPackageRequirements
                from IPython import get_ipython
            except Exception:
                return
            self._ip = get_ipython()
        else:
            self._ip = ip

        # noinspection PyBroadException
        try:
            # if this is colab, the callbacks do not contain the raw_cell content, so we have to patch it
            if 'google.colab' in self._ip.extension_manager.loaded:
                self._ip._org_run_cell = self._ip.run_cell
                self._ip.run_cell = partial(self._patched_run_cell, self._ip)
        except Exception:
            pass

        # start with the current history
        self._initialize_history()
        self._ip.events.register('post_run_cell', self._post_cell_callback)
        self._ip.events.register('pre_run_cell', self._pre_cell_callback)
        self._ip.set_custom_exc((Exception,), self._exception_callback)

    def _patched_run_cell(self, shell, *args, **kwargs):
        # noinspection PyBroadException
        try:
            raw_cell = kwargs.get('raw_cell') or args[0]
            self._current_cell = raw_cell
        except Exception:
            pass
        # noinspection PyProtectedMember
        return shell._org_run_cell(*args, **kwargs)

    def history(self, filename):
        with open(filename, 'wt') as f:
            for k, v in sorted(self._cells_code.items(), key=lambda p: p[0]):
                f.write(v)

    def history_to_str(self):
        # return a pair: (history as str, current cell if we are in still in cell execution otherwise None)
        return '\n'.join(v for k, v in sorted(self._cells_code.items(), key=lambda p: p[0])), self._current_cell

    # noinspection PyUnusedLocal
    def _exception_callback(self, shell, etype, value, tb, tb_offset=None):
        self._exception_raised = True
        return shell.showtraceback()

    def _pre_cell_callback(self, *args, **_):
        # noinspection PyBroadException
        try:
            if args:
                self._current_cell = args[0].raw_cell
            # we might have this value from somewhere else
            if self._current_cell:
                self._current_cell = self._conform_code(self._current_cell, replace_magic_bang=True)
        except Exception:
            pass

    def _post_cell_callback(self, *_, **__):
        # noinspection PyBroadException
        try:
            self._current_cell = None
            if self._exception_raised:
                # do nothing
                self._exception_raised = False
                return

            self._exception_raised = False
            # add the cell history
            # noinspection PyBroadException
            try:
                cell_code = '\n' + self._ip.history_manager.input_hist_parsed[-1]
            except Exception:
                return

            # fix magic / bang in code
            cell_code = self._conform_code(cell_code)

            self._cells_code[self._counter] = cell_code
            self._counter += 1
        except Exception:
            pass

    def _initialize_history(self):
        # only once
        if -1 in self._cells_code:
            return
        # noinspection PyBroadException
        try:
            cell_code = '\n' + '\n'.join(self._ip.history_manager.input_hist_parsed[:-1])
        except Exception:
            return

        cell_code = self._conform_code(cell_code)
        self._cells_code[-1] = cell_code

    def _conform_code(self, cell_code, replace_magic_bang=False):
        # fix magic / bang in code
        if self._replace_ipython_pattern:
            cell_code = self._replace_ipython_pattern.sub(r'\n# \g<1>get_ipython()', cell_code)
        if replace_magic_bang and self._replace_magic_pattern and self._replace_bang_pattern:
            cell_code = self._replace_magic_pattern.sub(r'\n# \g<1>%', cell_code)
            cell_code = self._replace_bang_pattern.sub(r'\n# \g<1>!', cell_code)
        return cell_code
