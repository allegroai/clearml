import os
import sys
from copy import copy
from tempfile import mkstemp

import attr
import collections
import logging
import json
from furl import furl
from pathlib2 import Path
from threading import Thread, Event

from .util import get_command_output
from ....backend_api import Session
from ....debugging import get_logger
from .detectors import GitEnvDetector, GitDetector, HgEnvDetector, HgDetector, Result as DetectionResult

_logger = get_logger("Repository Detection")


class ScriptInfoError(Exception):
    pass


class ScriptRequirements(object):
    _max_requirements_size = 512 * 1024

    def __init__(self, root_folder):
        self._root_folder = root_folder

    def get_requirements(self, entry_point_filename=None):
        try:
            from ....utilities.pigar.reqs import get_installed_pkgs_detail
            from ....utilities.pigar.__main__ import GenerateReqs
            installed_pkgs = get_installed_pkgs_detail()
            gr = GenerateReqs(save_path='', project_path=self._root_folder, installed_pkgs=installed_pkgs,
                              ignores=['.git', '.hg', '.idea', '__pycache__', '.ipynb_checkpoints',
                                       'site-packages', 'dist-packages'])
            reqs, try_imports, guess, local_pks = gr.extract_reqs(
                module_callback=ScriptRequirements.add_trains_used_packages, entry_point_filename=entry_point_filename)
            return self.create_requirements_txt(reqs, local_pks)
        except Exception:
            return '', ''

    @staticmethod
    def add_trains_used_packages(modules):
        # hack: forcefully insert storage modules if we have them
        # noinspection PyBroadException
        try:
            import boto3
            modules.add('boto3', 'trains.storage', 0)
        except Exception:
            pass
        # noinspection PyBroadException
        try:
            from google.cloud import storage
            modules.add('google_cloud_storage', 'trains.storage', 0)
        except Exception:
            pass
        # noinspection PyBroadException
        try:
            from azure.storage.blob import ContentSettings
            modules.add('azure_storage_blob', 'trains.storage', 0)
        except Exception:
            pass

        # bugfix, replace sklearn with scikit-learn name
        if 'sklearn' in modules:
            sklearn = modules.pop('sklearn', {})
            for fname, lines in sklearn.items():
                modules.add('scikit_learn', fname, lines)

        # if we have torch and it supports tensorboard, we should add that as well
        # (because it will not be detected automatically)
        if 'torch' in modules and 'tensorboard' not in modules:
            # noinspection PyBroadException
            try:
                # see if this version of torch support tensorboard
                import torch.utils.tensorboard
                import tensorboard
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
            for package, version in Task._force_requirements.items():
                modules.add(package, 'trains', 0)
        except Exception:
            pass

        return modules

    @staticmethod
    def create_requirements_txt(reqs, local_pks=None):
        # write requirements.txt
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
                    # check if this is a pypi package, if it is, leave it outside
                    if not r.get('channel') or r.get('channel') == 'pypi':
                        continue
                    # check if we have it in our required packages
                    name = r['name'].lower().replace('-', '_')
                    # hack support pytorch/torch different naming convention
                    if name == 'pytorch':
                        name = 'torch'
                    k, v = reqs_lower.get(name, (None, None))
                    if k:
                        conda_requirements += '{0} {1} {2}\n'.format(k, '==', v.version)
        except:
            conda_requirements = ''

        # add forced requirements:
        # noinspection PyBroadException
        try:
            from ..task import Task
            forced_packages = copy(Task._force_requirements)
        except Exception:
            forced_packages = {}

        # python version header
        requirements_txt = '# Python ' + sys.version.replace('\n', ' ').replace('\r', ' ') + '\n'

        if local_pks:
            requirements_txt += '\n# Local modules found - skipping:\n'
            for k, v in local_pks.sorted_items():
                requirements_txt += '# {0} == {1}\n'.format(k, v.version)

        # requirement summary
        requirements_txt += '\n'
        for k, v in reqs.sorted_items():
            version = v.version
            if k in forced_packages:
                forced_version = forced_packages.pop(k, None)
                if forced_version:
                    version = forced_version
            # requirements_txt += ''.join(['# {0}\n'.format(c) for c in v.comments.sorted_items()])
            if k == '-e':
                requirements_txt += '{0} {1}\n'.format(k, version)
            elif v:
                requirements_txt += '{0} {1} {2}\n'.format(k, '==', version)
            else:
                requirements_txt += '{0}\n'.format(k)

        # add forced requirements that we could not find installed on the system
        for k in sorted(forced_packages.keys()):
            if forced_packages[k]:
                requirements_txt += '{0} {1} {2}\n'.format(k, '==', forced_packages[k])
            else:
                requirements_txt += '{0}\n'.format(k)

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


class _JupyterObserver(object):
    _thread = None
    _exit_event = Event()
    _sync_event = Event()
    _sample_frequency = 30.
    _first_sample_frequency = 3.

    @classmethod
    def observer(cls, jupyter_notebook_filename):
        if cls._thread is not None:
            # order of signaling is important!
            cls._exit_event.set()
            cls._sync_event.set()
            cls._thread.join()

        cls._sync_event.clear()
        cls._exit_event.clear()
        cls._thread = Thread(target=cls._daemon, args=(jupyter_notebook_filename, ))
        cls._thread.daemon = True
        cls._thread.start()

    @classmethod
    def signal_sync(cls, *_):
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
        from trains import Task

        # load jupyter notebook package
        # noinspection PyBroadException
        try:
            from nbconvert.exporters.script import ScriptExporter
            _script_exporter = ScriptExporter()
        except Exception:
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

        try:
            from ....version import __version__
            our_module = cls.__module__.split('.')[0], __version__
        except:
            our_module = None

        try:
            import re
            replace_ipython_pattern = re.compile('\\n([ \\t]*)get_ipython\(\)')
        except:
            replace_ipython_pattern = None

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
                    # noinspection PyBroadException
                    try:
                        get_ipython().run_line_magic('notebook', local_jupyter_filename)
                    except Exception as ex:
                        continue

                # get notebook python script
                script_code, resources = _script_exporter.from_filename(local_jupyter_filename)
                current_script_hash = hash(script_code)
                if prev_script_hash and prev_script_hash == current_script_hash:
                    continue

                # remove ipython direct access from the script code
                # we will not be able to run them anyhow
                if replace_ipython_pattern:
                    script_code = replace_ipython_pattern.sub('\n# \g<1>get_ipython()', script_code)

                requirements_txt = ''
                conda_requirements = ''
                # parse jupyter python script and prepare pip requirements (pigar)
                # if backend supports requirements
                if file_import_modules and Session.check_min_api_version('2.2'):
                    fmodules, _ = file_import_modules(notebook.parts[-1], script_code)
                    fmodules = ScriptRequirements.add_trains_used_packages(fmodules)
                    installed_pkgs = get_installed_pkgs_detail()
                    # make sure we are in installed packages
                    if our_module and (our_module[0] not in installed_pkgs):
                        installed_pkgs[our_module[0]] = our_module

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
                task._update_script(script=data_script)
                # update requirements
                task._update_requirements(requirements=requirements_txt)
            except Exception:
                pass


class ScriptInfo(object):

    plugins = [GitEnvDetector(), HgEnvDetector(), HgDetector(), GitDetector()]
    """ Script info detection plugins, in order of priority """

    @classmethod
    def _jupyter_install_post_store_hook(cls, jupyter_notebook_filename):
        # noinspection PyBroadException
        try:
            if 'IPython' in sys.modules:
                from IPython import get_ipython
                if get_ipython():
                    _JupyterObserver.observer(jupyter_notebook_filename)
                    get_ipython().events.register('pre_run_cell', _JupyterObserver.signal_sync)
        except Exception:
            pass

    @classmethod
    def _get_jupyter_notebook_filename(cls):
        if not (sys.argv[0].endswith(os.path.sep + 'ipykernel_launcher.py') or
                sys.argv[0].endswith(os.path.join(os.path.sep, 'ipykernel', '__main__.py'))) \
                or len(sys.argv) < 3 or not sys.argv[2].endswith('.json'):
            return None

        # we can safely assume that we can import the notebook package here
        # noinspection PyBroadException
        try:
            from notebook.notebookapp import list_running_servers
            import requests
            current_kernel = sys.argv[2].split(os.path.sep)[-1].replace('kernel-', '').replace('.json', '')
            try:
                server_info = next(list_running_servers())
            except Exception:
                # on some jupyter notebook versions this function can crash on parsing the json file,
                # we will parse it manually here
                import ipykernel
                from glob import glob
                import json
                for f in glob(os.path.join(os.path.dirname(ipykernel.get_connection_file()), 'nbserver-*.json')):
                    try:
                        with open(f, 'r') as json_data:
                            server_info = json.load(json_data)
                    except:
                        server_info = None
                    if server_info:
                        break
            try:
                r = requests.get(
                    url=server_info['url'] + 'api/sessions',
                    headers={'Authorization': 'token {}'.format(server_info.get('token', '')), })
            except requests.exceptions.SSLError:
                # disable SSL check warning
                from urllib3.exceptions import InsecureRequestWarning
                requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
                # fire request
                r = requests.get(
                    url=server_info['url'] + 'api/sessions',
                    headers={'Authorization': 'token {}'.format(server_info.get('token', '')), }, verify=False)
                # enable SSL check warning
                import warnings
                warnings.simplefilter('default', InsecureRequestWarning)

            r.raise_for_status()
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
                from IPython import get_ipython
                if get_ipython() and 'google.colab' in get_ipython().extension_manager.loaded:
                    is_google_colab = True
            except Exception:
                pass

            if is_google_colab:
                script_entry_point = notebook_name
                local_ipynb_file = None
            else:
                # always slash, because this is from uri (so never backslash not even oon windows)
                entry_point_filename = notebook_path.split('/')[-1]

                # now we should try to find the actual file
                entry_point = (Path.cwd() / entry_point_filename).absolute()
                if not entry_point.is_file():
                    entry_point = (Path.cwd() / notebook_path).absolute()

                # get local ipynb for observer
                local_ipynb_file = entry_point.as_posix()

                # now replace the .ipynb with .py
                # we assume we will have that file available with the Jupyter notebook plugin
                entry_point = entry_point.with_suffix('.py')

                script_entry_point = entry_point.as_posix()

            # install the post store hook,
            # notice that if we do not have a local file we serialize/write every time the entire notebook
            cls._jupyter_install_post_store_hook(local_ipynb_file)

            return script_entry_point
        except Exception:
            return None

    @classmethod
    def _get_entry_point(cls, repo_root, script_path):
        repo_root = Path(repo_root).absolute()

        try:
            # Use os.path.relpath as it calculates up dir movements (../)
            entry_point = os.path.relpath(str(script_path), str(Path.cwd()))
        except ValueError:
            # Working directory not under repository root
            entry_point = script_path.relative_to(repo_root)

        return Path(entry_point).as_posix()

    @classmethod
    def _get_working_dir(cls, repo_root):
        repo_root = Path(repo_root).absolute()

        try:
            return Path.cwd().relative_to(repo_root).as_posix()
        except ValueError:
            # Working directory not under repository root
            return os.path.curdir

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
    def _get_script_info(cls, filepaths, check_uncommitted=True, create_requirements=True, log=None):
        jupyter_filepath = cls._get_jupyter_notebook_filename()
        if jupyter_filepath:
            scripts_path = [Path(os.path.normpath(jupyter_filepath)).absolute()]
        else:
            scripts_path = [Path(os.path.normpath(f)).absolute() for f in filepaths if f]
            if all(not f.is_file() for f in scripts_path):
                raise ScriptInfoError(
                    "Script file {} could not be found".format(scripts_path)
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

        plugin = next((p for p in cls.plugins if any(p.exists(d) for d in scripts_dir)), None)
        repo_info = DetectionResult()
        script_dir = scripts_dir[0]
        script_path = scripts_path[0]
        if not plugin:
            log.info("No repository found, storing script code instead")
        else:
            try:
                for i, d in enumerate(scripts_dir):
                    repo_info = plugin.get_info(str(d), include_diff=check_uncommitted)
                    if not repo_info.is_empty():
                        script_dir = d
                        script_path = scripts_path[i]
                        break
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
            working_dir = cls._get_working_dir(repo_root)
            entry_point = cls._get_entry_point(repo_root, script_path)

        if check_uncommitted:
            diff = cls._get_script_code(script_path.as_posix()) \
                if not plugin or not repo_info.commit else repo_info.diff
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
                requirements, conda_requirements = script_requirements.get_requirements()
        else:
            script_requirements = None

        script_info = dict(
            repository=furl(repo_info.url).remove(username=True, password=True).tostr(),
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

        messages = []
        if repo_info.modified:
            messages.append(
                "======> WARNING! UNCOMMITTED CHANGES IN REPOSITORY {} <======".format(
                    script_info.get("repository", "")
                )
            )

        if not any(script_info.values()):
            script_info = None

        return (ScriptInfoResult(script=script_info, warning_messages=messages),
                script_requirements)

    @classmethod
    def get(cls, filepaths=None, check_uncommitted=True, create_requirements=True, log=None):
        try:
            if not filepaths:
                filepaths = [sys.argv[0], ]
            return cls._get_script_info(
                filepaths=filepaths, check_uncommitted=check_uncommitted,
                create_requirements=create_requirements, log=log)
        except Exception as ex:
            if log:
                log.warning("Failed auto-detecting task repository: {}".format(ex))
        return ScriptInfoResult(), None

    @classmethod
    def detect_running_module(cls, script_dict):
        # noinspection PyBroadException
        try:
            # If this is jupyter, do not try to detect the running module, we know what we have.
            if script_dict.get('jupyter_filepath'):
                return script_dict

            if '__main__' in sys.modules and vars(sys.modules['__main__'])['__package__']:
                argvs = ''
                git_root = os.path.abspath(script_dict['repo_root']) if script_dict['repo_root'] else None
                for a in sys.argv[1:]:
                    if git_root and os.path.exists(a):
                        # check if common to project:
                        a_abs = os.path.abspath(a)
                        if os.path.commonpath([a_abs, git_root]) == git_root:
                            # adjust path relative to working dir inside git repo
                            a = ' ' + os.path.relpath(a_abs, os.path.join(git_root, script_dict['working_dir']))
                    argvs += ' {}'.format(a)
                # update the script entry point to match the real argv and module call
                script_dict['entry_point'] = '-m {}{}'.format(
                    vars(sys.modules['__main__'])['__package__'], (' ' + argvs) if argvs else '')
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
