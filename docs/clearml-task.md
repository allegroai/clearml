# `clearml-task` - Execute ANY python code on a remote machine

Using only your command line and __zero__ additional lines of code, you can easily integrate the ClearML platform
into your experiment. With the `clearml-task` command, you can create a [Task](https://allegro.ai/clearml/docs/docs/concepts_fundamentals/concepts_fundamentals_tasks.html)
using any script from **any python code or repository and launch it on a remote machine**.

The remote execution is fully monitored. All outputs - including console / tensorboard / matplotlib -
are logged in real-time into the ClearML UI.

## What does it do?

With the `clearml-task` command, you specify the details of your experiment including:
* Project and task name
* Repository / commit / branch
* [Queue](https://allegro.ai/clearml/docs/docs/concepts_fundamentals/concepts_fundamentals_workers_and_queues.html)
  name
* Optional: the base docker image to be used as underlying environment
* Optional: alternative python requirements, in case `requirements.txt` is not found inside the repository.

Then `clearml-task` does the rest of the heavy-lifting. It creates a new experiment or Task on your `clearml-server`
according to your specifications, and then, it will enqueue the experiment to the selected execution queue.

While the Task is executed on the remote machine (performed by an available `clearml-agent`), all the console outputs
will be logged in real-time, alongside your TensorBoard and matplotlib. During and after the Task execution, you can
track and visualize the results in the ClearML Web UI.

### Use-cases for `clearml-task` remote execution

- You have an off-the-shelf code, and you want to launch it on a remote machine with a specific resource (i.e., GPU)
- You want to run the [hyper-parameter optimization]() on a codebase that is still not connected to `clearml`
- You want to create a [pipeline]() from an assortment of scripts, and you need to create Tasks for those scripts
- Sometimes, you just want to run some code on a remote machine, either using an on-prem cluster or on the cloud...

### Prerequisites

- A single python script, or an up-to-date repository containing the codebase.
- `clearml` installed. `clearml` also has a [Task](https://allegro.ai/clearml/docs/rst/getting_started/index.html)
  feature but it requires two lines of code in order to integrate the platform.
- `clearml-agent` running on at least one machine (to execute the experiment)

## Tutorial

### Launching a job from a repository

You will be launching this [script](https://github.com/allegroai/events/blob/master/webinar-0620/keras_mnist.py)
on a remote machine. You will be using the following command-line options:
1. Give the experiment a name and select a project, for example: `--project keras_examples --name remote_test`. If the project
   doesn't exist, a new project will be created with the selected name.
2. Select the repository with your code. For example: `--repo https://github.com/allegroai/events.git` You can specify a
   branch and/or commit using `--branch <branch_name> --commit <commit_id>`. If you do not specify the
   branch / commit, it will use by default the latest commit from the master branch,
3. Specify which script in the repository needs to be run, for example: `--script /webinar-0620/keras_mnist.py`
By default, the execution working directory will be the root of the repository. If you need to change it,
   add `--cwd <folder>`
4. If you need, pass an argument to your scripts, use `--args`, followed by the arguments.
   The names of the arguments should match the argparse arguments, but without the '--' prefix. Instead
   of --key=value -> use `--args key=value`, for example `--args batch_size=64 epochs=1`
5. Select the queue for your Task's execution, for example: `--queue default`. If a queue isn't chosen, the Task
   will not be executed, it will be left in [draft mode](https://allegro.ai/clearml/docs/docs/concepts_fundamentals/concepts_fundamentals_tasks.html?highlight=draft#task-states-and-state-transitions),
   and you can enqueue and execute the Task at a later point.
6. Add required packages. If your repo has a requirements.txt file, you don't need to do anything; `clearml-task`
   will automatically find the file and put it in your Task. If your repo does __not__ have a requirements file and
there are packages that are necessary for the execution of your code, use --packages <package_name>. For example:
   `--packages "keras" "tensorflow>2.2"`.

``` bash
clearml-task --project keras_examples --name remote_test --repo https://github.com/allegroai/events.git
--script /webinar-0620/keras_mnist.py --args batch_size=64 epochs=1 --queue default
```


### Launching a job from a local script

You will be launching a single local script file (no git repo needed) on a remote machine:

1. Give the experiment a name and select a project (`--project examples --name remote_test`)
2. Select the script file on your machine, `--script /path/to/my/script.py`
3. If you require specific packages to run your code, you can specify them manually with `--packages "package_name" "package_name2`,
   for example: `packages "keras" "tensorflow>2.2"`
  or you can pass a requirements file `--requirements /path/to/my/requirements.txt`
4. If you need to pass arguments, like in the repo case, add `--args key=value` and make sure that the key names match
   the argparse arguments (`--args batch_size=64 epochs=1`)
5. If you have a docker container with an entire environment in which you want your script to run inside,
  add e.g. `--docker nvcr.io/nvidia/pytorch:20.11-py3`
6. Select the queue for your Task's execution, for example: `--queue dual_gpu`. If a queue isn't chosen, the Task
   will not be executed, it will be left in [draft mode](https://allegro.ai/clearml/docs/docs/concepts_fundamentals/concepts_fundamentals_tasks.html?highlight=draft#task-states-and-state-transitions),
   and you can enqueue and execute it at a later point.

``` bash
clearml-task --project examples --name remote_test --script /path/to/my/script.py
--packages "keras" "tensorflow>2.2" --args epochs=1 batch_size=64
--queue dual_gpu
```

### CLI options

``` bash
clearml-task --help
```

``` console
ClearML launch - launch any codebase on remote machine running clearml-agent

optional arguments:
  -h, --help            show this help message and exit
  --version             Display the clearml-task utility version
  --project PROJECT     Required: set the project name for the task. If
                        --base-task-id is used, this arguments is optional.
  --name NAME           Required: select a name for the remote task
  --repo REPO           remote URL for the repository to use. Example: --repo
                        https://github.com/allegroai/clearml.git
  --branch BRANCH       Select specific repository branch/tag (implies the
                        latest commit from the branch)
  --commit COMMIT       Select specific commit id to use (default: latest
                        commit, or when used with local repository matching
                        the local commit id)
  --folder FOLDER       Remotely execute the code in the local folder. Notice!
                        It assumes a git repository already exists. Current
                        state of the repo (commit id and uncommitted changes)
                        is logged and will be replicated on the remote machine
  --script SCRIPT       Specify the entry point script for the remote
                        execution. When used in tandem with --repo the script
                        should be a relative path inside the repository, for
                        example: --script source/train.py .When used with
                        --folder it supports a direct path to a file inside
                        the local repository itself, for example: --script
                        ~/project/source/train.py
  --cwd CWD             Working directory to launch the script from. Default:
                        repository root folder. Relative to repo root or local
                        folder
  --args [ARGS [ARGS ...]]
                        Arguments to pass to the remote execution, list of
                        <argument>=<value> strings.Currently only argparse
                        arguments are supported. Example: --args lr=0.003
                        batch_size=64
  --queue QUEUE         Select the queue to launch the task. If not provided a
                        Task will be created but it will not be launched.
  --requirements REQUIREMENTS
                        Specify requirements.txt file to install when setting
                        the session. If not provided, the requirements.txt
                        from the repository will be used.
  --packages [PACKAGES [PACKAGES ...]]
                        Manually specify a list of required packages. Example:
                        --packages "tqdm>=2.1" "scikit-learn"
  --docker DOCKER       Select the docker image to use in the remote session
  --task-type TASK_TYPE
                        Set the Task type, optional values: training, testing,
                        inference, data_processing, application, monitor,
                        controller, optimizer, service, qc, custom
  --skip-task-init      If set, Task.init() call is not added to the entry
                        point, and is assumed to be called in within the
                        script. Default: add Task.init() call entry point
                        script
  --base-task-id BASE_TASK_ID
                        Use a pre-existing task in the system, instead of a
                        local repo/script. Essentially clones an existing task
                        and overrides arguments/requirements.
                        
```
