# ClearML - example code for logging configuration files to Task":
#
import json
from pathlib import Path
import yaml

from clearml import Task


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='FirstTrial', task_name='config_files_example')


# -----------------------------------------------
#  Log config file
#  Notice any file format i supported
#  In the Web UI you could edit the configuration file directly as text
#  and launch on a remote worker with the new configuration automatically applied
# -----------------------------------------------

config_file = task.connect_configuration(Path("data_samples") / "sample.json", name='json config file')

with open(config_file.as_posix(), "rt") as f:
    config_json = json.load(f)

print(config_json)

config_file = task.connect_configuration(Path("data_samples") / "config_yaml.yaml", name='yaml config file')

with open(config_file.as_posix(), "rt") as f:
    config_yaml = yaml.load(f, Loader=yaml.SafeLoader)

print(config_yaml)

print("done")
