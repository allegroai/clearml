""" ClearML open SDK """
from six import PY2

from .version import __version__
from .task import Task
from .model import InputModel, OutputModel, Model
from .logger import Logger
from .storage import StorageManager
from .errors import UsageError
from .datasets import Dataset

if not PY2:
    from .automation.controller import PipelineController

    __all__ = [
        "__version__",
        "Task",
        "InputModel",
        "OutputModel",
        "Model",
        "Logger",
        "StorageManager",
        "UsageError",
        "Dataset",
        "PipelineController",
    ]
else:
    __all__ = [
        "__version__",
        "Task",
        "InputModel",
        "OutputModel",
        "Model",
        "Logger",
        "StorageManager",
        "UsageError",
        "Dataset",
    ]
