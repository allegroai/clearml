import json
import os
import re
from functools import reduce
from logging import getLogger
from typing import Optional, Sequence, Union, Tuple, List

from six.moves.urllib.parse import urlparse

from pathlib2 import Path

from ...task import Task
from .repo import ScriptInfo


class CreateAndPopulate(object):
    def __init__(
            self,
            project_name=None,  # Optional[str]
            task_name=None,  # Optional[str]
            task_type=None,  # Optional[str]
            repo=None,  # Optional[str]
            branch=None,  # Optional[str]
            commit=None,  # Optional[str]
            script=None,  # Optional[str]
            working_directory=None,  # Optional[str]
            packages=None,  # Optional[Union[bool, Sequence[str]]]
            requirements_file=None,  # Optional[Union[str, Path]]
            docker=None,  # Optional[str]
            docker_args=None,  # Optional[str]
            docker_bash_setup_script=None,  # Optional[str]
            base_task_id=None,  # Optional[str]
            add_task_init_call=True,  # bool
            raise_on_missing_entries=False,  # bool
            verbose=False,  # bool
    ):
        # type: (...) -> None
        """
        Create a new Task from an existing code base.
        If the code does not already contain a call to Task.init, pass add_task_init_call=True,
        and the code will be patched in remote execution (i.e. when executed by `clearml-agent`

        :param project_name: Set the project name for the task. Required if base_task_id is None.
        :param task_name: Set the name of the remote task. Required if base_task_id is None.
        :param task_type: Optional, The task type to be created. Supported values: 'training', 'testing', 'inference',
            'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc', 'custom'
        :param repo: Remote URL for the repository to use, OR path to local copy of the git repository
            Example: 'https://github.com/allegroai/clearml.git' or '~/project/repo'
        :param branch: Select specific repository branch/tag (implies the latest commit from the branch)
        :param commit: Select specific commit id to use (default: latest commit,
            or when used with local repository matching the local commit id)
        :param script: Specify the entry point script for the remote execution. When used in tandem with
            remote git repository the script should be a relative path inside the repository,
            for example: './source/train.py' . When used with local repository path it supports a
            direct path to a file inside the local repository itself, for example: '~/project/source/train.py'
        :param working_directory: Working directory to launch the script from. Default: repository root folder.
            Relative to repo root or local folder.
        :param packages: Manually specify a list of required packages. Example: ["tqdm>=2.1", "scikit-learn"]
            or `True` to automatically create requirements
            based on locally installed packages (repository must be local).
        :param requirements_file: Specify requirements.txt file to install when setting the session.
            If not provided, the requirements.txt from the repository will be used.
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environement
        :param base_task_id: Use a pre-existing task in the system, instead of a local repo/script.
            Essentially clones an existing task and overrides arguments/requirements.
        :param add_task_init_call: If True, a 'Task.init()' call is added to the script entry point in remote execution.
        :param raise_on_missing_entries: If True raise ValueError on missing entries when populating
        :param verbose: If True print verbose logging
        """
        if len(urlparse(repo).scheme) <= 1:
            folder = repo
            repo = None
        else:
            folder = None

        if raise_on_missing_entries and not base_task_id:
            if not script:
                raise ValueError("Entry point script not provided")
            if not repo and not folder and not Path(script).is_file():
                raise ValueError("Script file \'{}\' could not be found".format(script))
        if raise_on_missing_entries and commit and branch:
            raise ValueError(
                "Specify either a branch/tag or specific commit id, not both (either --commit or --branch)")
        if raise_on_missing_entries and not folder and working_directory and working_directory.startswith('/'):
            raise ValueError("working directory \'{}\', must be relative to repository root")

        if requirements_file and not Path(requirements_file).is_file():
            raise ValueError("requirements file could not be found \'{}\'")

        self.folder = folder
        self.commit = commit
        self.branch = branch
        self.repo = repo
        self.script = script
        self.cwd = working_directory
        assert not packages or isinstance(packages, (tuple, list, bool))
        self.packages = list(packages) if packages is not None and not isinstance(packages, bool) \
            else (packages or None)
        self.requirements_file = Path(requirements_file) if requirements_file else None
        self.base_task_id = base_task_id
        self.docker = dict(image=docker, args=docker_args, bash_script=docker_bash_setup_script)
        self.add_task_init_call = add_task_init_call
        self.project_name = project_name
        self.task_name = task_name
        self.task_type = task_type
        self.task = None
        self.raise_on_missing_entries = raise_on_missing_entries
        self.verbose = verbose

    def create_task(self):
        # type: () -> Task
        """
        Create the new populated Task

        :return: newly created Task object
        """
        local_entry_file = None
        repo_info = None
        if self.folder or (self.script and Path(self.script).is_file() and not self.repo):
            self.folder = os.path.expandvars(os.path.expanduser(self.folder)) if self.folder else None
            self.script = os.path.expandvars(os.path.expanduser(self.script)) if self.script else None
            self.cwd = os.path.expandvars(os.path.expanduser(self.cwd)) if self.cwd else None
            if Path(self.script).is_file():
                entry_point = self.script
            else:
                entry_point = (Path(self.folder) / self.script).as_posix()
            entry_point = os.path.abspath(entry_point)
            if not os.path.isfile(entry_point):
                raise ValueError("Script entrypoint file \'{}\' could not be found".format(entry_point))

            local_entry_file = entry_point
            repo_info, requirements = ScriptInfo.get(
                filepaths=[entry_point],
                log=getLogger(),
                create_requirements=self.packages is True, uncommitted_from_remote=True,
                detect_jupyter_notebook=False)

        # check if we have no repository and no requirements raise error
        if self.raise_on_missing_entries and (not self.requirements_file and not self.packages) \
                and not self.repo and (
                not repo_info or not repo_info.script or not repo_info.script.get('repository')):
            raise ValueError("Standalone script detected \'{}\', but no requirements provided".format(self.script))

        if self.base_task_id:
            if self.verbose:
                print('Cloning task {}'.format(self.base_task_id))
            task = Task.clone(source_task=self.base_task_id, project=Task.get_project_id(self.project_name))
        else:
            # noinspection PyProtectedMember
            task = Task._create(
                task_name=self.task_name, project_name=self.project_name,
                task_type=self.task_type or Task.TaskTypes.training)

            # if there is nothing to populate, return
            if not any([
                self.folder, self.commit, self.branch, self.repo, self.script, self.cwd,
                self.packages, self.requirements_file, self.base_task_id] + (list(self.docker.values()))
            ):
                return task

        task_state = task.export_task()
        if 'script' not in task_state:
            task_state['script'] = {}

        if repo_info:
            task_state['script']['repository'] = repo_info.script['repository']
            task_state['script']['version_num'] = repo_info.script['version_num']
            task_state['script']['branch'] = repo_info.script['branch']
            task_state['script']['diff'] = repo_info.script['diff'] or ''
            task_state['script']['working_dir'] = repo_info.script['working_dir']
            task_state['script']['entry_point'] = repo_info.script['entry_point']
            task_state['script']['binary'] = repo_info.script['binary']
            task_state['script']['requirements'] = repo_info.script.get('requirements') or {}
            if self.cwd:
                self.cwd = self.cwd
                cwd = self.cwd if Path(self.cwd).is_dir() else (
                            Path(repo_info.script['repo_root']) / self.cwd).as_posix()
                if not Path(cwd).is_dir():
                    raise ValueError("Working directory \'{}\' could not be found".format(cwd))
                cwd = Path(cwd).relative_to(repo_info.script['repo_root']).as_posix()
                entry_point = \
                    Path(repo_info.script['repo_root']) / repo_info.script['working_dir'] / repo_info.script[
                        'entry_point']
                entry_point = entry_point.relative_to(cwd).as_posix()
                task_state['script']['entry_point'] = entry_point or ""
                task_state['script']['working_dir'] = cwd or "."
        elif self.repo:
            # normalize backslashes and remove first one
            entry_point = '/'.join([p for p in self.script.split('/') if p and p != '.'])
            cwd = '/'.join([p for p in (self.cwd or '.').split('/') if p and p != '.'])
            if cwd and entry_point.startswith(cwd + '/'):
                entry_point = entry_point[len(cwd) + 1:]
            task_state['script']['repository'] = self.repo
            task_state['script']['version_num'] = self.commit or None
            task_state['script']['branch'] = self.branch or None
            task_state['script']['diff'] = ''
            task_state['script']['working_dir'] = cwd or '.'
            task_state['script']['entry_point'] = entry_point or ""
        else:
            # standalone task
            task_state['script']['entry_point'] = self.script or ""
            task_state['script']['working_dir'] = '.'

        # update requirements
        reqs = []
        if self.requirements_file:
            with open(self.requirements_file.as_posix(), 'rt') as f:
                reqs = [line.strip() for line in f.readlines()]
        if self.packages and self.packages is not True:
            reqs += self.packages
        if reqs:
            # make sure we have clearml.
            clearml_found = False
            for line in reqs:
                if line.strip().startswith('#'):
                    continue
                package = reduce(lambda a, b: a.split(b)[0], "#;@=~<>", line).strip()
                if package == 'clearml':
                    clearml_found = True
                    break
            if not clearml_found:
                reqs.append('clearml')
            task_state['script']['requirements'] = {'pip': '\n'.join(reqs)}
        elif not self.repo and repo_info and not repo_info.script.get('requirements'):
            # we are in local mode, make sure we have "requirements.txt" it is a must
            reqs_txt_file = Path(repo_info.script['repo_root']) / "requirements.txt"
            if self.raise_on_missing_entries and not reqs_txt_file.is_file():
                raise ValueError(
                    "requirements.txt not found [{}] "
                    "Use --requirements or --packages".format(reqs_txt_file.as_posix()))

        if self.add_task_init_call:
            script_entry = os.path.abspath('/' + task_state['script'].get('working_dir', '.') +
                                           '/' + task_state['script']['entry_point'])
            idx_a = 0
            # find the right entry for the patch if we have a local file (basically after __future__
            if local_entry_file:
                with open(local_entry_file, 'rt') as f:
                    lines = f.readlines()
                future_found = self._locate_future_import(lines)
                if future_found >= 0:
                    idx_a = future_found + 1

            task_init_patch = ''
            if self.repo or task_state.get('script', {}).get('repository'):
                # if we do not have requirements, add clearml to the requirements.txt
                if not reqs:
                    task_init_patch += \
                        "diff --git a/requirements.txt b/requirements.txt\n" \
                        "--- a/requirements.txt\n" \
                        "+++ b/requirements.txt\n" \
                        "@@ -0,0 +1,1 @@\n" \
                        "+clearml\n"

                # Add Task.init call
                task_init_patch += \
                    "diff --git a{script_entry} b{script_entry}\n" \
                    "--- a{script_entry}\n" \
                    "+++ b{script_entry}\n" \
                    "@@ -{idx_a},0 +{idx_b},3 @@\n" \
                    "+from clearml import Task\n" \
                    "+Task.init()\n" \
                    "+\n".format(
                        script_entry=script_entry, idx_a=idx_a, idx_b=idx_a + 1)
            else:
                # Add Task.init call
                task_init_patch += \
                    "from clearml import Task\n" \
                    "Task.init()\n\n"

            # make sure we add the dif at the end of the current diff
            task_state['script']['diff'] = task_state['script'].get('diff', '')
            if task_state['script']['diff'] and not task_state['script']['diff'].endswith('\n'):
                task_state['script']['diff'] += '\n'
            task_state['script']['diff'] += task_init_patch

        # set base docker image if provided
        if self.docker:
            task.set_base_docker(
                docker_cmd=self.docker.get('image'),
                docker_arguments=self.docker.get('args'),
                docker_setup_bash_script=self.docker.get('bash_script'),
            )

        if self.verbose:
            if task_state['script']['repository']:
                repo_details = {k: v for k, v in task_state['script'].items()
                                if v and k not in ('diff', 'requirements', 'binary')}
                print('Repository Detected\n{}'.format(json.dumps(repo_details, indent=2)))
            else:
                print('Standalone script detected\n  Script: {}'.format(self.script))

            if task_state['script'].get('requirements') and \
                    task_state['script']['requirements'].get('pip'):
                print('Requirements:{}{}'.format(
                    '\n  Using requirements.txt: {}'.format(
                        self.requirements_file.as_posix()) if self.requirements_file else '',
                    '\n  {}Packages: {}'.format('Additional ' if self.requirements_file else '', self.packages)
                    if self.packages else ''
                ))
            if self.docker:
                print('Base docker image: {}'.format(self.docker))

        # update the Task
        task.update_task(task_state)
        self.task = task
        return task

    def update_task_args(self, args=None):
        # type: (Optional[Union[Sequence[str], Sequence[Tuple[str, str]]]]) -> ()
        """
        Update the newly created Task argparse Arguments
        If called before Task created, used for argument verification

        :param args: Arguments to pass to the remote execution, list of string pairs (argument, value) or
            list of strings '<argument>=<value>'. Example: ['lr=0.003', (batch_size, 64)]
        """
        if not args:
            return

        # check args are in format <key>=<value>
        args_list = []
        for a in args:
            if isinstance(a, (list, tuple)):
                assert len(a) == 2
                args_list.append(a)
                continue
            try:
                parts = a.split('=', 1)
                assert len(parts) == 2
                args_list.append(parts)
            except Exception:
                raise ValueError(
                    "Failed parsing argument \'{}\', arguments must be in \'<key>=<value>\' format")

        if not self.task:
            return

        task_params = self.task.get_parameters()
        args_list = {'Args/{}'.format(k): v for k, v in args_list}
        task_params.update(args_list)
        self.task.set_parameters(task_params)

    def get_id(self):
        # type: () -> Optional[str]
        """
        :return: Return the created Task id (str)
        """
        return self.task.id if self.task else None

    @staticmethod
    def _locate_future_import(lines):
        # type: (List[str]) -> int
        """
        :param lines: string lines of a python file
        :return: line index of the last __future_ import. return -1 if no __future__ was found
        """
        # skip over the first two lines, they are ours
        # then skip over empty or comment lines
        lines = [(i, line.split('#', 1)[0].rstrip()) for i, line in enumerate(lines)
                 if line.strip('\r\n\t ') and not line.strip().startswith('#')]

        # remove triple quotes ' """ '
        nested_c = -1
        skip_lines = []
        for i, line_pair in enumerate(lines):
            for _ in line_pair[1].split('"""')[1:]:
                if nested_c >= 0:
                    skip_lines.extend(list(range(nested_c, i + 1)))
                    nested_c = -1
                else:
                    nested_c = i
        # now select all the
        lines = [pair for i, pair in enumerate(lines) if i not in skip_lines]

        from_future = re.compile(r"^from[\s]*__future__[\s]*")
        import_future = re.compile(r"^import[\s]*__future__[\s]*")
        # test if we have __future__ import
        found_index = -1
        for a_i, (_, a_line) in enumerate(lines):
            if found_index >= a_i:
                continue
            if from_future.match(a_line) or import_future.match(a_line):
                found_index = a_i
                # check the last import block
                i, line = lines[found_index]
                # wither we have \\ character at the end of the line or the line is indented
                parenthesized_lines = '(' in line and ')' not in line
                while line.endswith('\\') or parenthesized_lines:
                    found_index += 1
                    i, line = lines[found_index]
                    if ')' in line:
                        break

            else:
                break

        return found_index if found_index < 0 else lines[found_index][0]
