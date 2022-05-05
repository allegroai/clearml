from threading import Thread
from typing import Any, Callable, Optional, Type


class FutureCaller(object):
    """
    FutureCaller is used to create a class via a functions async, in another thread.

    For example:

    .. code-block:: py

        future = FutureCaller().call(func=max, func_cb=None, override_cls=None, 1, 2)
        print('Running other code')
        print(future.result())  # will print '2'
    """
    __slots__ = ('__object', '__object_cls', '__executor')

    @property
    def __class__(self):
        return self.__object_cls

    def __init__(self, func, func_cb, override_cls, *args, **kwargs):
        # type: (Callable, Optional[Callable], Type, *Any, **Any) -> None
        """
        __init__(*args, **kwargs) in another thread

        :return: This FutureCaller instance
        """
        self.__object = None
        self.__object_cls = override_cls

        self.__executor = Thread(target=self.__submit__, args=(func, func_cb, args, kwargs))
        self.__executor.daemon = True
        self.__executor.start()

    def __submit__(self, fn, fn_cb, args, kwargs):
        self.__object = fn(*args, **kwargs)
        if fn_cb is not None:
            fn_cb(self.__object)

    def __getattr__(self, item):
        # if we get here, by definition this is not a __slot__ entry, pass to the object
        return getattr(self.__result__(), item)

    def __setattr__(self, item, value):
        # make sure we can set the slots
        if item in ["_FutureCaller__executor", "_FutureCaller__object", "_FutureCaller__object_cls"]:
            return super(FutureCaller, self).__setattr__(item, value)

        setattr(self.__result__(), item, value)

    def __result__(self, timeout=None):
        # type: (Optional[float]) -> Any
        """
        Wait and get the result of the function called with self.call()

        :param timeout: The maximum number of seconds to wait for the result. If None,
            there is no limit for the wait time.

        :return: The result of the called function
        """
        if self.__executor:
            self.__executor.join(timeout=timeout)
            self.__executor = None
        return self.__object
