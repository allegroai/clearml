import os
import sys

import attr
from furl import furl
from pathlib2 import Path

from ....debugging import get_logger
from .detectors import GitEnvDetector, GitDetector, HgEnvDetector, HgDetector, Result as DetectionResult

_logger = get_logger("Repository Detection")


class ScriptInfoError(Exception):
    pass


class ScriptInfo(object):

    plugins = [GitEnvDetector(), HgEnvDetector(), HgDetector(), GitDetector()]
    """ Script info detection plugins, in order of priority """

    @classmethod
    def _get_jupyter_notebook_filename(cls):
        if not sys.argv[0].endswith(os.path.sep+'ipykernel_launcher.py') or len(sys.argv) < 3 or not sys.argv[2].endswith('.json'):
            return None

        # we can safely assume that we can import the notebook package here
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

            notebook_path = cur_notebook['notebook']['path']
            entry_point_filename = notebook_path.split(os.path.sep)[-1]

            # now we should try to find the actual file
            entry_point = (Path.cwd() / entry_point_filename).absolute()
            if not entry_point.is_file():
                entry_point = (Path.cwd() / notebook_path).absolute()

            # now replace the .ipynb with .py
            # we assume we will have that file available with the Jupyter notebook plugin
            entry_point = entry_point.with_suffix('.py')

            return entry_point.as_posix()
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
    def _get_script_info(cls, filepath, check_uncommitted=False, log=None):
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

        script_info = dict(
            repository=furl(repo_info.url).remove(username=True, password=True).tostr(),
            branch=repo_info.branch,
            version_num=repo_info.commit,
            entry_point=entry_point,
            working_dir=working_dir,
            diff=repo_info.diff,
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
    def get(cls, filepath=sys.argv[0], check_uncommitted=False, log=None):
        try:
            return cls._get_script_info(
                filepath=filepath, check_uncommitted=check_uncommitted, log=log
            )
        except Exception as ex:
            if log:
                log.warning("Failed auto-detecting task repository: {}".format(ex))
        return ScriptInfoResult()


@attr.s
class ScriptInfoResult(object):
    script = attr.ib(default=None)
    warning_messages = attr.ib(factory=list)
