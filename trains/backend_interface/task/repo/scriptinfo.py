import os
import sys
from tempfile import mkstemp

import attr
import collections
import logging
from furl import furl
from pathlib2 import Path
from threading import Thread, Event

from ....backend_api import Session
from ....debugging import get_logger
from .detectors import GitEnvDetector, GitDetector, HgEnvDetector, HgDetector, Result as DetectionResult

_logger = get_logger("Repository Detection")


class ScriptInfoError(Exception):
    pass


class ScriptRequirements(object):
    def __init__(self, root_folder):
        self._root_folder = root_folder

    def get_requirements(self):
        try:
            from pigar import reqs
            reqs.project_import_modules = ScriptRequirements._patched_project_import_modules
            from pigar.__main__ import GenerateReqs
            from pigar.log import logger
            logger.setLevel(logging.WARNING)
            installed_pkgs = reqs.get_installed_pkgs_detail()
            gr = GenerateReqs(save_path='', project_path=self._root_folder, installed_pkgs=installed_pkgs,
                              ignores=['.git', '.hg', '.idea', '__pycache__', '.ipynb_checkpoints'])
            reqs, try_imports, guess = gr.extract_reqs()
            return self.create_requirements_txt(reqs)
        except Exception:
            return ''

    @staticmethod
    def _patched_project_import_modules(project_path, ignores):
        """
        copied form pigar req.project_import_modules
        patching, os.getcwd() is incorrectly used
        """
        from pigar.modules import ImportedModules
        from pigar.reqs import file_import_modules
        modules = ImportedModules()
        try_imports = set()
        local_mods = list()
        cur_dir = project_path  # os.getcwd()
        ignore_paths = collections.defaultdict(set)
        if not ignores:
            ignore_paths[project_path].add('.git')
        else:
            for path in ignores:
                parent_dir = os.path.dirname(path)
                ignore_paths[parent_dir].add(os.path.basename(path))

        for dirpath, dirnames, files in os.walk(project_path, followlinks=True):
            if dirpath in ignore_paths:
                dirnames[:] = [d for d in dirnames
                               if d not in ignore_paths[dirpath]]
            py_files = list()
            for fn in files:
                # C extension.
                if fn.endswith('.so'):
                    local_mods.append(fn[:-3])
                # Normal Python file.
                if fn.endswith('.py'):
                    local_mods.append(fn[:-3])
                    py_files.append(fn)
            if '__init__.py' in files:
                local_mods.append(os.path.basename(dirpath))
            for file in py_files:
                fpath = os.path.join(dirpath, file)
                fake_path = fpath.split(cur_dir)[1][1:]
                with open(fpath, 'rb') as f:
                    fmodules, try_ipts = file_import_modules(fake_path, f.read())
                    modules |= fmodules
                    try_imports |= try_ipts

        return modules, try_imports, local_mods

    @staticmethod
    def create_requirements_txt(reqs):
        # write requirements.txt

        # python version header
        requirements_txt = '# Python ' + sys.version.replace('\n', ' ').replace('\r', ' ') + '\n'

        # requirement summary
        requirements_txt += '\n'
        for k, v in reqs.sorted_items():
            # requirements_txt += ''.join(['# {0}\n'.format(c) for c in v.comments.sorted_items()])
            if k == '-e':
                requirements_txt += '{0} {1}\n'.format(k, v.version)
            elif v:
                requirements_txt += '{0} {1} {2}\n'.format(k, '==', v.version)
            else:
                requirements_txt += '{0}\n'.format(k)

        # requirements details (in comments)
        requirements_txt += '\n' + \
                            '# Detailed import analysis\n' \
                            '# **************************\n'
        for k, v in reqs.sorted_items():
            requirements_txt += '\n'
            if k == '-e':
                requirements_txt += '# IMPORT PACKAGE {0} {1}\n'.format(k, v.version)
            else:
                requirements_txt += '# IMPORT PACKAGE {0}\n'.format(k)
            requirements_txt += ''.join(['# {0}\n'.format(c) for c in v.comments.sorted_items()])

        return requirements_txt


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
            from pigar.reqs import get_installed_pkgs_detail, file_import_modules
            from pigar.modules import ReqsModules
            from pigar.log import logger
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
        # main observer loop
        while True:
            # wait for timeout or sync event
            cls._sync_event.wait(cls._sample_frequency if counter else cls._first_sample_frequency)
            # check if we need to exit
            if cls._exit_event.wait(timeout=0.):
                return
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
                requirements_txt = ''
                # parse jupyter python script and prepare pip requirements (pigar)
                # if backend supports requirements
                if file_import_modules and Session.api_version > '2.1':
                    fmodules, _ = file_import_modules(notebook.parts[-1], script_code)
                    installed_pkgs = get_installed_pkgs_detail()
                    reqs = ReqsModules()
                    for name in fmodules:
                        if name in installed_pkgs:
                            pkg_name, version = installed_pkgs[name]
                            reqs.add(pkg_name, version, fmodules[name])
                    requirements_txt = ScriptRequirements.create_requirements_txt(reqs)

                # update script
                prev_script_hash = current_script_hash
                data_script = task.data.script
                data_script.diff = script_code
                data_script.requirements = {'pip': requirements_txt}
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
        if not (sys.argv[0].endswith(os.path.sep+'ipykernel_launcher.py') or
                sys.argv[0].endswith(os.path.join(os.path.sep, 'ipykernel', '__main__.py'))) \
                or len(sys.argv) < 3 or not sys.argv[2].endswith('.json'):
            return None

        # we can safely assume that we can import the notebook package here
        # noinspection PyBroadException
        try:
            from notebook.notebookapp import list_running_servers
            import requests
            current_kernel = sys.argv[2].split(os.path.sep)[-1].replace('kernel-', '').replace('.json', '')
            server_info = next(list_running_servers())
            r = requests.get(
                url=server_info['url'] + 'api/sessions',
                headers={'Authorization': 'token {}'.format(server_info.get('token', '')), })
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
    def _get_script_info(cls, filepath, check_uncommitted=True, create_requirements=True, log=None):
        jupyter_filepath = cls._get_jupyter_notebook_filename()
        if jupyter_filepath:
            script_path = Path(os.path.normpath(jupyter_filepath)).absolute()
        else:
            script_path = Path(os.path.normpath(filepath)).absolute()
            if not script_path.is_file():
                raise ScriptInfoError(
                    "Script file [{}] could not be found".format(filepath)
                )

        script_dir = script_path.parent

        def _log(msg, *args, **kwargs):
            if not log:
                return
            log.warning(
                "Failed auto-detecting task repository: {}".format(
                    msg.format(*args, **kwargs)
                )
            )

        plugin = next((p for p in cls.plugins if p.exists(script_dir)), None)
        repo_info = DetectionResult()
        if not plugin:
            _log("expected one of: {}", ", ".join((p.name for p in cls.plugins)))
        else:
            try:
                repo_info = plugin.get_info(str(script_dir), include_diff=check_uncommitted)
            except Exception as ex:
                _log("no info for {} ({})", script_dir, ex)
            else:
                if repo_info.is_empty():
                    _log("no info for {}", script_dir)

        repo_root = repo_info.root or script_dir
        working_dir = cls._get_working_dir(repo_root)
        entry_point = cls._get_entry_point(repo_root, script_path)
        if check_uncommitted:
            diff = cls._get_script_code(script_path.as_posix()) \
                if not plugin or not repo_info.commit else repo_info.diff
        else:
            diff = ''
            # if this is not jupyter, get the requirements.txt
        requirements = ''
        # create requirements if backend supports requirements
        if create_requirements and not jupyter_filepath and Session.api_version > '2.1':
            script_requirements = ScriptRequirements(Path(repo_root).as_posix())
            requirements = script_requirements.get_requirements()

        script_info = dict(
            repository=furl(repo_info.url).remove(username=True, password=True).tostr(),
            branch=repo_info.branch,
            version_num=repo_info.commit,
            entry_point=entry_point,
            working_dir=working_dir,
            diff=diff,
            requirements={'pip': requirements} if requirements else None,
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

        return ScriptInfoResult(script=script_info, warning_messages=messages)

    @classmethod
    def get(cls, filepath=sys.argv[0], check_uncommitted=True, create_requirements=True, log=None):
        try:
            return cls._get_script_info(
                filepath=filepath, check_uncommitted=check_uncommitted,
                create_requirements=create_requirements, log=log)
        except Exception as ex:
            if log:
                log.warning("Failed auto-detecting task repository: {}".format(ex))
        return ScriptInfoResult()


@attr.s
class ScriptInfoResult(object):
    script = attr.ib(default=None)
    warning_messages = attr.ib(factory=list)
