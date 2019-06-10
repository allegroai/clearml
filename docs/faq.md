# FAQ

**Can I store more information on the models? For example, can I store enumeration of classes?**
 
YES!

Use the SDK `set_model_label_enumeration` method:
 
```python
Task.current_task().set_model_label_enumeration( {‘label’: int(0), } )
```

**Can I store the model configuration file as well?**
 
YES!

Use the SDK `set_model_design` method:

```python
Task.current_task().set_model_design( ‘a very long text of the configuration file content’ )
```

**I want to add more graphs, not just with Tensorboard. Is this supported?**

YES!

Use an SDK [Logger](link to git) object. An instance can be always be retrieved with `Task.current_task().get_logger()`:

```python
logger = Task.current_task().get_logger()
logger.report_scalar("loss", "classification", iteration=42, value=1.337)
```

TRAINS supports scalars, plots, 2d/3d scatter diagrams, histograms, surface diagrams, confusion matrices, images, and text logging.

An example can be found [here](docs/manual_log.py).

**I noticed that all of my experiments appear as “Training”. Are there other options?**

YES! 

When creating experiments and calling `Task.init`, you can pass an experiment type.
The currently supported types are `Task.TaskTypes.training` and `Task.TaskTypes.testing`:

```python
task = Task.init(project_name, task_name, Task.TaskTypes.testing)
```

If you feel we should add a few more, let us know in the [issues]() section.

**I noticed I keep getting a message “warning: uncommitted code”. What does it mean?**

TRAINS not only detects your current repository and git commit, 
but it also warns you if you are using uncommitted code. TRAINS does this
because uncommitted code means it will be difficult to reproduce this experiment.

**Is there something you can do about uncommitted code running?**

YES! 

TRAINS currently stores the git diff together with the project. 
The Web-App will soon present the git diff as well. This is coming very soon! 

**I read that there is a feature for centralized model storage. How do I use it?**

Pass the `output_uri` parameter to `Task.init`, for example:

```python
Task.init(project_name, task_name, output_uri=’/mnt/shared/folder’)
```

All of the stored snapshots are copied into a subfolder whose name contains the task ID, for example:
 
`/mnt/shared/folder/task_6ea4f0b56d994320a713aeaf13a86d9d/models/`

Other options include:

```python
Task.init(project_name, task_name, output_uri=’s3://bucket/folder’)
```

```python
Task.init(project_name, task_name, output_uri=’gs://bucket/folder’)
```

These require configuring the cloud storage credentials in `~/trains.conf` (see an [example](v)).

**I am training multiple models at the same time, but I only see one of them. What happened?**

This will be fixed in a future version. Currently, TRAINS does support multiple models 
from the same task/experiment so you can find all the models in the project Models tab.
In the Task view, we only present the last one.

**Can I log input and output models manually?**

YES!

See  [InputModel]() and [OutputModel]().

For example:

```python
input_model = InputModel.import_model(link_to_initial_model_file)
Task.current_task().connect(input_model)
OutputModel(Task.current_task()).update_weights(link_to_new_model_file_here) 
```

**I am using Jupyter Notebook. Is this supported?**

YES! 

Jupyter Notebook is supported.

**I do not use ArgParser for hyper-parameters. Do you have a solution?**

YES! 

TRAINS supports using a Python dictionary for hyper-parameter logging.

```python
parameters_dict = Task.current_task().connect(parameters_dict)
```

From this point onward, not only are the dictionary key/value pairs stored, but also any change to the dictionary is automatically stored.

**Git is not well supported in Jupyter. We just gave up on properly committing our code. Do you have a solution?**

YES! 

Check our [trains-jupyter-plugin](). It is a Jupyter plugin that allows you to commit your notebook directly from Jupyter. It also saves the Python version of the code and creates an updated `requirements.txt` so you know which packages you were using.

**Can I use TRAINS with scikit-learn?**

YES! 

scikit-learn is supported. Everything you do is logged, with the caveat that models are not logged automatically. 
 Models are not logged automatically because, in most cases, scikit-learn is simply pickling the object to files so there is no underlying frame to connect to.

**I am working with PyCharm and remotely debugging a machine, but the git repo is not detected. Do you have a solution?**

YES! 

This is such a common occurrence that we created a PyCharm plugin that allows for a remote debugger to grab your local repository / commit ID. See our [trains-pycharm-plugin]() repository for instructions and [latest release]().

**How do I know a new version came out?**

Unfortunately, TRAINS currently does not support auto-update checks. We hope to add this soon.

**Sometimes I see experiments as running while they are not. What is it?**

When the Python process exits in an orderly fashion, TRAINS closes the experiment. 
If a process crashes, then sometimes the stop signal is missed. You can safely right click on the experiment in the Web-App and stop it.

**In the experiment log tab, I’m missing the first log lines. Where are they?**
 
Unfortunately, due to speed/optimization issues, we opted to display only the last several hundreds. The full log can be downloaded from the Web-App.




