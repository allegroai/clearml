# TRAINS FAQ

* [Can I store more information on the models?](#store-more-model-info)
* [Can I store the model configuration file as well?](#store-model-configuration)
* [I want to add more graphs, not just with Tensorboard. Is this supported?](#more-graph-types)
* [I noticed that all of my experiments appear as `Training`. Are there other options?](#other-experiment-types)
* [I noticed I keep getting the message `warning: uncommitted code`. What does it mean?](#uncommitted-code-warning)
* [Is there something TRAINS can do about uncommitted code running?](#help-uncommitted-code)
* [I read there is a feature for centralized model storage. How do I use it?](#centralized-model-storage)
* [I am training multiple models at the same time, but I only see one of them. What happened?](#only-last-model-appears)
* [Can I log input and output models manually?](#manually-log-models)
* [I am using Jupyter Notebook. Is this supported?](#jupyter-notebook)
* [I do not use Argarser for hyper-parameters. Do you have a solution?](#dont-want-argparser)
* [Git is not well supported in Jupyter, so we just gave up on committing our code. Do you have a solution?](#commit-git-in-jupyter)
* [Can I use TRAINS with scikit-learn?](#use-scikit-learn)
* [When using PyCharm to remotely debug a machine, the git repo is not detected. Do you have a solution?](#pycharm-remote-debug-detect-git)
* [How do I know a new version came out?](#new-version-auto-update)
* [Sometimes I see experiments as running when in fact they are not. What's going on?](#experiment-running-but-stopped)
* [The first log lines are missing from the experiment log tab. Where did they go?](#first-log-lines-missing)


## Can I store more information on the models? <a name="store-more-model-info"></a>

#### For example, can I store enumeration of classes?

Yes! Use the `Task.set_model_label_enumeration()` method:

```python
Task.current_task().set_model_label_enumeration( {"label": int(0), } )
```

## Can I store the model configuration file as well?  <a name="store-model-configuration"></a>

Yes! Use the `Task.set_model_design()` method:

```python
Task.current_task().set_model_design("a very long text with the configuration file's content")
```

## I want to add more graphs, not just with Tensorboard. Is this supported? <a name="more-graph-types"></a>

Yes! Use a [Logger](https://github.com/allegroai/trains/blob/master/trains/logger.py) object. An instance can be always be retrieved using the `Task.current_task().get_logger()` method:

```python
# Get a logger object
logger = Task.current_task().get_logger()

# Report some scalar 
logger.report_scalar("loss", "classification", iteration=42, value=1.337)
```

#### **TRAINS supports:**
* Scalars
* Plots
* 2D/3D Scatter Diagrams
* Histograms
* Surface Diagrams
* Confusion Matrices
* Images
* Text logs

For a more detailed example, see [here](https://github.com/allegroai/trains/blob/master/examples/manual_reporting.py).


## I noticed that all of my experiments appear as `Training`. Are there other options? <a name="other-experiment-types"></a>

Yes! When creating experiments and calling `Task.init`, you can provide an experiment type.
The currently supported types are `Task.TaskTypes.training` and `Task.TaskTypes.testing`. For example:

```python
task = Task.init(project_name, task_name, Task.TaskTypes.testing)
```

If you feel we should add a few more, let us know in the [issues](https://github.com/allegroai/trains/issues) section.


## I noticed I keep getting the message `warning: uncommitted code`. What does it mean? <a name="uncommitted-code-warning"></a>

TRAINS not only detects your current repository and git commit,
but also warns you if you are using uncommitted code. TRAINS does this
because uncommitted code means this experiment will be difficult to reproduce.

If you still don't care, just ignore this message - it is merely a warning.  


## Is there something TRAINS can do about uncommitted code running?  <a name="help-uncommitted-code"></a>

Yes! TRAINS currently stores the git diff as part of the experiment's information.
The Web-App will soon present the git diff as well. This is coming very soon!


## I read there is a feature for centralized model storage. How do I use it? <a name="centralized-model-storage"></a>

When calling `Task.init()`, providing the `output_uri` parameter allows you to specify the location in which model snapshots will be stored.

For example, calling:

```python
task = Task.init(project_name, task_name, output_uri="/mnt/shared/folder")
```

Will tell TRAINS to copy all stored snapshots into a sub-folder under `/mnt/shared/folder`. 
The sub-folder's name will contain the experiment's ID. 
Assuming the experiment's ID in this example is `6ea4f0b56d994320a713aeaf13a86d9d`, the following folder will be used:

`/mnt/shared/folder/task_6ea4f0b56d994320a713aeaf13a86d9d/models/`

TRAINS supports more storage types for `output_uri`:

```python
# AWS S3 bucket
task = Task.init(project_name, task_name, output_uri="s3://bucket-name/folder")
```

```python
# Google Cloud Storage bucket
taks = Task.init(project_name, task_name, output_uri="gs://bucket-name/folder")
```

**NOTE:** These require configuring the storage credentials in `~/trains.conf`.
For a more detailed example, see [here](https://github.com/allegroai/trains/blob/master/docs/trains.conf#L51).


## I am training multiple models at the same time, but I only see one of them. What happened? <a name="only-last-model-appears"></a>

Although all models can be found under the project's **Models** tab, TRAINS currently shows only the last model associated with an experiment in the experiment's information panel. 

This will be fixed in a future version.

## Can I log input and output models manually? <a name="manually-log-models"></a>

Yes! For example:

```python
input_model = InputModel.import_model(link_to_initial_model_file)
Task.current_task().connect(input_model)

OutputModel(Task.current_task()).update_weights(link_to_new_model_file_here)
```

See [InputModel](https://github.com/allegroai/trains/blob/master/trains/model.py#L319) and [OutputModel](https://github.com/allegroai/trains/blob/master/trains/model.py#L539) for more information.


## I am using Jupyter Notebook. Is this supported? <a name="jupyter-notebook"></a>

Yes! Jupyter Notebook is supported. See [TRAINS Jupyter Plugin](https://github.com/allegroai/trains-jupyter-plugin).


## I do not use Argarser for hyper-parameters. Do you have a solution? <a name="dont-want-argparser"></a>

Yes! TRAINS supports using a Python dictionary for hyper-parameter logging. Just call:

```python
parameters_dict = Task.current_task().connect(parameters_dict)
```

From this point onward, not only are the dictionary key/value pairs stored as part of the experiment, but any changes to the dictionary will be automatically updated in the task's information.


## Git is not well supported in Jupyter, so we just gave up on committing our code. Do you have a solution? <a name="commit-git-in-jupyter"></a>

Yes! Check our [TRAINS Jupyter Plugin](https://github.com/allegroai/trains-jupyter-plugin). This plugin allows you to commit your notebook directly from Jupyter. It also saves the Python version of your code and creates an updated `requirements.txt` so you know which packages you were using.


## Can I use TRAINS with scikit-learn? <a name="use-scikit-learn"></a>

Yes! `scikit-learn` is supported. Everything you do is logged.

**NOTE**: Models are not automatically logged because in most cases, scikit-learn will simply pickle the object to files so there is no underlying frame we can connect to.


## When using PyCharm to remotely debug a machine, the git repo is not detected. Do you have a solution? <a name="pycharm-remote-debug-detect-git"></a>

Yes! Since this is such a common occurrence, we created a PyCharm plugin that allows a remote debugger to grab your local repository / commit ID. See our [TRAINS PyCharm Plugin](https://github.com/allegroai/trains-pycharm-plugin) repository for instructions and [latest release](https://github.com/allegroai/trains-pycharm-plugin/releases).


## How do I know a new version came out? <a name="new-version-auto-update"></a>

TRAINS does not yet support auto-update checks. We hope to add this feature soon.


## Sometimes I see experiments as running when in fact they are not. What's going on? <a name="experiment-running-but-stopped"></a>

TRAINS monitors your Python process. When the process exits in an orderly fashion, TRAINS closes the experiment.

When the process crashes and terminates abnormally, the stop signal is sometimes missed. In such a case, you can safely right click the experiment in the Web-App and stop it.


## The first log lines are missing from the experiment log tab. Where did they go? <a name="first-log-lines-missing"></a>

Due to speed/optimization issues, we opted to display only the last several hundred log lines.

You can always downloaded the full log as a file using the Web-App.

