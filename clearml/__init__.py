""" ClearML open SDK """
from six import PY2

from .version import __version__
from .task import Task
from .model import InputModel, OutputModel, Model
from .logger import Logger
from .storage import StorageManager
from .errors import UsageError
from .datasets import Dataset

TaskTypes = Task.TaskTypes

if not PY2:
    from .automation.controller import PipelineController, PipelineDecorator  # noqa: F401

    __all__ = [
        "__version__",
        "Task",
        "TaskTypes",
        "InputModel",
        "OutputModel",
        "Model",
        "Logger",
        "StorageManager",
        "UsageError",
        "Dataset",
        "PipelineController",
        "PipelineDecorator",
    ]
else:
    __all__ = [
        "__version__",
        "Task",
        "TaskTypes",
        "InputModel",
        "OutputModel",
        "Model",
        "Logger",
        "StorageManager",
        "UsageError",
        "Dataset",
    ]
