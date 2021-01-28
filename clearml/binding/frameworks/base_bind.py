from abc import ABCMeta, abstractmethod

import six


@six.add_metaclass(ABCMeta)
class PatchBaseModelIO(object):
    """
    Base class for patched models

    :param __main_task: Task to run (Experiment)
    :type __main_task: Task
    :param __patched: True if the model is patched
    :type __patched: bool
    """
    @property
    @abstractmethod
    def __main_task(self):
        pass

    @property
    @abstractmethod
    def __patched(self):
        pass

    @staticmethod
    @abstractmethod
    def update_current_task(task, **kwargs):
        """
        Update the model task to run
        :param task: the experiment to do
        :type task: Task
        """
        pass

    @staticmethod
    @abstractmethod
    def _patch_model_io():
        """
        Patching the load and save functions
        """
        pass

    @staticmethod
    @abstractmethod
    def _save(original_fn, obj, f, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def _load(original_fn, f, *args, **kwargs):
        pass
