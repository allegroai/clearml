# `clearml-task` - Execute ANY python code on a remote machine

If you are already familiar with `clearml`, then you can think of `clearml-task` as a way to create a Task/experiment 
from any script without the need to add even a single line of code to the original codebase.

`clearml-task` allows a user to **take any python code/repository and launch it on a remote machine**.

The remote execution is fully monitored, all outputs - including console / tensorboard / matplotlib 
are logged in real-time into the ClearML UI

## What does it do?

`clearml-task` creates a new experiment on your `clearml-server`; it populates the experiment's environment with:

* repository/commit/branch, as specified by the command-line invocation. 
* optional: the base docker image to be used as underlying environment
* optional: alternative python requirements, in case `requirements.txt` is not found inside the repository.

Once the new experiment is created and populated, it will enqueue the experiment to the selected execution queue.

When the experiment is executed on the remote machine (performed by an available `clearml-agent`), all the console outputs
will be logged in real-time, alongside your TensorBoard and matplotlib.

### Use-cases for `clearml-task` remote execution

- You have an off-the-shelf code, and you want to launch it on a remote machine with a specific resource (i.e., GPU)
- You want to run the [hyper-parameter optimization]() on a codebase that is still not connected with `clearml`
- You want to create a [pipeline]() from an assortment of scripts, and you need to create Tasks for those scripts
- Sometimes, you just want to run some code on a remote machine, either using an on-prem cluster or on the cloud... 

### Prerequisites

- A single python script, or an up-to-date repository containing the codebase.
- `clearml-agent` running on at least one machine (to execute the experiment) 

## Tutorial

### Launching a job from a repository

We will be launching this [script](https://github.com/allegroai/trains/blob/master/examples/frameworks/scikit-learn/sklearn_matplotlib_example.py) on a remote machine. The following are the command-line options we will be using:
- First, we have to give the experiment a name and select a project (`--project examples --name remote_test`)
- Then, we select the repository with our code. If we do not specify branch / commit, it will take the latest commit 
  from the master branch (`--repo https://github.com/allegroai/clearml.git`)
- Lastly, we need to specify which script in the repository needs to be run (`--script examples/frameworks/scikit-learn/sklearn_matplotlib_example.py`)
Notice that by default, the execution working directory will be the root of the repository. If we need to change it, add `--cwd <folder>`

If we additionally need to pass an argument to our scripts, use the `--args` switch. 
  The names of the arguments should match the argparse arguments, removing the '--' prefix 
  (e.g. instead of --key=value -> use `--args key=value` )   

``` bash
clearml-task --project examples --name remote_test --repo https://github.com/allegroai/clearml.git 
--script examples/frameworks/scikit-learn/sklearn_matplotlib_example.py
--queue single_gpu
```

### Launching a job from a local script

We will be launching a single local script file (no git repo needed) on a remote machine.

- First, we have to give the experiment a name and select a project (`--project examples --name remote_test`)
- Then, we select the script file on our machine, `--script /path/to/my/script.py`
- If we need specific packages, we can specify them manually with `--packages "tqdm>=4" "torch>1.0"` 
  or we can pass a requirements file `--requirements /path/to/my/requirements.txt`
- Same as in the repo case, if we need to pass arguments to `argparse` we can add `--args key=value`
- If we have a docker container with an entire environment we want our script to run inside, 
  add e.g., `--docker nvcr.io/nvidia/pytorch:20.11-py3`

Note: In this example, the exact version of PyTorch to install will be resolved by the `clearml-agent` depending on the CUDA environment available at runtime.

``` bash
clearml-task --project examples --name remote_test --script /path/to/my/script.py 
--packages "tqdm>=4" "torch>1.0" --args verbose=true
--queue dual_gpu
```

### CLI options

``` bash
clearml-task --help
```

``` console
ClearML launch - launch any codebase on remote machines running clearml-agent

optional arguments:
  -h, --help            show this help message and exit
  --version             Display the Allegro.ai utility version
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
  --skip-task-init      If set, Task.init() call is not added to the entry
                        point, and is assumed to be called in within the
                        script. Default: add Task.init() call entry point
                        script
  --base-task-id BASE_TASK_ID
                        Use a pre-existing task in the system, instead of a local repo/script.
                        Essentially clones an existing task and overrides arguments/requirements.
                        
```
