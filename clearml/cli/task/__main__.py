from argparse import ArgumentParser

from pathlib2 import Path

from clearml import Task
from clearml.backend_interface.task.populate import CreateAndPopulate


def setup_parser(parser):
    parser.add_argument('--version', action='store_true', default=None,
                        help='Display the clearml-task utility version')
    parser.add_argument('--project', type=str, default=None,
                        help='Required: set the project name for the task. '
                             'If --base-task-id is used, this arguments is optional.')
    parser.add_argument('--name', type=str, default=None, required=True,
                        help='Required: select a name for the remote task')
    parser.add_argument('--repo', type=str, default=None,
                        help='remote URL for the repository to use. '
                             'Example: --repo https://github.com/allegroai/clearml.git')
    parser.add_argument('--branch', type=str, default=None,
                        help='Select specific repository branch/tag (implies the latest commit from the branch)')
    parser.add_argument('--commit', type=str, default=None,
                        help='Select specific commit id to use (default: latest commit, '
                             'or when used with local repository matching the local commit id)')
    parser.add_argument('--folder', type=str, default=None,
                        help='Remotely execute the code in the local folder. '
                             'Notice! It assumes a git repository already exists. '
                             'Current state of the repo (commit id and uncommitted changes) is logged '
                             'and will be replicated on the remote machine')
    parser.add_argument('--script', type=str, default=None,
                        help='Specify the entry point script for the remote execution. '
                             'When used in tandem with --repo the script should be a relative path inside '
                             'the repository, for example: --script source/train.py .'
                             'When used with --folder it supports a direct path to a file inside the local '
                             'repository itself, for example: --script ~/project/source/train.py')
    parser.add_argument('--cwd', type=str, default=None,
                        help='Working directory to launch the script from. Default: repository root folder. '
                             'Relative to repo root or local folder')
    parser.add_argument('--args', default=None, nargs='*',
                        help='Arguments to pass to the remote execution, list of <argument>=<value> strings.'
                             'Currently only argparse arguments are supported. '
                             'Example: --args lr=0.003 batch_size=64')
    parser.add_argument('--queue', type=str, default=None,
                        help='Select the queue to launch the task. '
                             'If not provided a Task will be created but it will not be launched.')
    parser.add_argument('--requirements', type=str, default=None,
                        help='Specify requirements.txt file to install when setting the session. '
                             'If not provided, the requirements.txt from the repository will be used.')
    parser.add_argument('--packages', default=None, nargs='*',
                        help='Manually specify a list of required packages. '
                             'Example: --packages "tqdm>=2.1" "scikit-learn"')
    parser.add_argument('--docker', type=str, default=None,
                        help='Select the docker image to use in the remote session')
    parser.add_argument('--docker_args', type=str, default=None,
                        help='Add docker arguments, pass a single string')
    parser.add_argument('--docker_bash_setup_script', type=str, default=None,
                        help="Add bash script to be executed inside the docker before setting up "
                             "the Task's environment")
    parser.add_argument('--task-type', type=str, default=None,
                        help='Set the Task type, optional values: '
                             'training, testing, inference, data_processing, application, monitor, '
                             'controller, optimizer, service, qc, custom')
    parser.add_argument('--skip-task-init', action='store_true', default=None,
                        help='If set, Task.init() call is not added to the entry point, and is assumed '
                             'to be called in within the script. Default: add Task.init() call entry point script')
    parser.add_argument('--base-task-id', type=str, default=None,
                        help='Use a pre-existing task in the system, instead of a local repo/script. '
                             'Essentially clones an existing task and overrides arguments/requirements.')


def cli():
    title = 'ClearML launch - launch any codebase on remote machine running clearml-agent'
    print(title)
    parser = ArgumentParser(description=title)
    setup_parser(parser)

    # get the args
    args = parser.parse_args()

    if args.version:
        from ...version import __version__
        print('Version {}'.format(__version__))
        exit(0)

    if args.docker_bash_setup_script and Path(args.docker_bash_setup_script).is_file():
        with open(args.docker_bash_setup_script, "r") as bash_setup_script_file:
            bash_setup_script = bash_setup_script_file.readlines()
            # remove Bash Shebang
            if bash_setup_script and bash_setup_script[0].strip().startswith("#!"):
                bash_setup_script = bash_setup_script[1:]
    else:
        bash_setup_script = args.docker_bash_setup_script or None

    create_populate = CreateAndPopulate(
        project_name=args.project,
        task_name=args.name,
        task_type=args.task_type,
        repo=args.repo or args.folder,
        branch=args.branch,
        commit=args.commit,
        script=args.script,
        working_directory=args.cwd,
        packages=args.packages,
        requirements_file=args.requirements,
        docker=args.docker,
        docker_args=args.docker_args,
        docker_bash_setup_script=bash_setup_script,
        base_task_id=args.base_task_id,
        add_task_init_call=not args.skip_task_init,
        raise_on_missing_entries=True,
        verbose=True,
    )
    # verify args
    create_populate.update_task_args(args.args)

    print('Creating new task')
    create_populate.create_task()
    # update Task args
    create_populate.update_task_args(args.args)

    print('New task created id={}'.format(create_populate.get_id()))
    if not args.queue:
        print('Warning: No queue was provided, leaving task in draft-mode.')
        exit(0)

    Task.enqueue(create_populate.task, queue_name=args.queue)
    print('Task id={} sent for execution on queue {}'.format(create_populate.get_id(), args.queue))
    print('Execution log at: {}'.format(create_populate.task.get_output_log_web_page()))


def main():
    try:
        cli()
    except KeyboardInterrupt:
        print('\nUser aborted')
    except Exception as ex:
        print('\nError: {}'.format(ex))
        exit(1)


if __name__ == '__main__':
    main()
