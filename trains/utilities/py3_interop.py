""" Convenience classes supporting python3-like concepts """
import abc

import six


@six.add_metaclass(abc.ABCMeta)
class AbstractContextManager(object):
    """An abstract base class for context managers. Supported in contextlib from python 3.6 and up """

    def __enter__(self):
        """Return `self` upon entering the runtime context."""
        return self

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractContextManager:
            if (any("__enter__" in B.__dict__ for B in C.__mro__) and any("__exit__" in B.__dict__ for B in C.__mro__)):
                return True
        return NotImplemented


try:

    from abc import abstractclassmethod

except ImportError:

    class abstractclassmethod(classmethod):
        __isabstractmethod__ = True

        def __init__(self, callable):
            callable.__isabstractmethod__ = True
            super(abstractclassmethod, self).__init__(callable)
