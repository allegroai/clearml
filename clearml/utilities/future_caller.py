from copy import deepcopy
from time import sleep

from six.moves.queue import Queue, Empty
from threading import Thread
from typing import Any, Callable, Optional, Type


class _DeferredClass(object):
    __slots__ = ('__queue', '__future_caller', '__future_func')

    def __init__(self, a_future_caller, future_func):
        self.__queue = Queue()
        self.__future_caller = a_future_caller
        self.__future_func = future_func

    def __nested_caller(self, item, args, kwargs):
        # wait until object is constructed
        getattr(self.__future_caller, "id")  # noqa

        future_func = getattr(self.__future_caller, self.__future_func)
        the_object = future_func()
        the_object_func = getattr(the_object, item)
        return the_object_func(*args, **kwargs)

    def _flush_into_logger(self, a_future_object=None, a_future_func=None):
        self.__close_queue(a_future_object=a_future_object, a_future_func=a_future_func)

    def __close_queue(self, a_future_object=None, a_future_func=None):
        # call this function when we Know the object is initialization is completed
        if self.__queue is None:
            return

        _queue = self.__queue
        self.__queue = None
        while True:
            # noinspection PyBroadException
            try:
                item, args, kwargs = _queue.get(block=False)
                if a_future_object:
                    future_func = getattr(a_future_object, self.__future_func)
                    the_object = future_func()
                    the_object_func = getattr(the_object, item)
                    the_object_func(*args, **kwargs)
                elif a_future_func:
                    the_object_func = getattr(a_future_func, item)
                    the_object_func(*args, **kwargs)
                else:
                    self.__nested_caller(item, args, kwargs)
            except Empty:
                break
            except Exception:
                # import traceback
                # stdout_print(''.join(traceback.format_exc()))
                pass

    def __getattr__(self, item):
        def _caller(*args, **kwargs):
            # if we already completed the background initialization, call functions immediately
            # noinspection PyProtectedMember
            if not self.__queue or self.__future_caller._FutureTaskCaller__executor is None:
                return self.__nested_caller(item, args, kwargs)

            # noinspection PyBroadException
            try:
                # if pool is still active call async
                self.__queue.put((item, deepcopy(args) if args else args, deepcopy(kwargs) if kwargs else kwargs))
            except Exception:
                # assume we wait only if self.__pool was nulled between the if and now, so just call directly
                return self.__nested_caller(item, args, kwargs)

            # let's hope it is the right one
            return True

        return _caller


class FutureTaskCaller(object):
    """
    FutureTaskCaller is used to create a class via a functions async, in another thread.

    For example:

    .. code-block:: py

        future = FutureTaskCaller().call(func=max, func_cb=None, override_cls=None, 1, 2)
        print('Running other code')
        print(future.result())  # will print '2'
    """
    __slots__ = ('__object', '__object_cls', '__executor', '__deferred_bkg_class')

    @property
    def __class__(self):
        return self.__object_cls

    def __init__(self, func, func_cb, override_cls, *args, **kwargs):
        # type: (Callable, Optional[Callable], Type, *Any, **Any) -> None
        """
        __init__(*args, **kwargs) in another thread

        :return: This FutureTaskCaller instance
        """
        self.__object = None
        self.__object_cls = override_cls
        self.__deferred_bkg_class = _DeferredClass(self, "get_logger")

        self.__executor = Thread(target=self.__submit__, args=(func, func_cb, args, kwargs))
        self.__executor.daemon = True
        self.__executor.start()

    def __submit__(self, fn, fn_cb, args, kwargs):
        # background initialization call
        _object = fn(*args, **kwargs)

        # push all background calls (now that the initialization is complete)
        if self.__deferred_bkg_class:
            _deferred_bkg_class = self.__deferred_bkg_class
            self.__deferred_bkg_class = None
            # noinspection PyProtectedMember
            _deferred_bkg_class._flush_into_logger(a_future_object=_object)

        # store the initialized object
        self.__object = _object
        # callback function
        if fn_cb is not None:
            fn_cb(self.__object)

    def __getattr__(self, item):
        # if we get here, by definition this is not a __slot__ entry, pass to the object
        return getattr(self.__result__(), item)

    def __setattr__(self, item, value):
        # make sure we can set the slots
        if item in ["_FutureTaskCaller__executor", "_FutureTaskCaller__object",
                    "_FutureTaskCaller__object_cls", "_FutureTaskCaller__deferred_bkg_class"]:
            return super(FutureTaskCaller, self).__setattr__(item, value)

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
            # since the test is not atomic, we assume that if we failed joining
            # it is because someone else joined before us
            # noinspection PyBroadException
            try:
                self.__executor.join(timeout=timeout)
            except RuntimeError:
                # this is probably calling ourselves from the same thread
                raise
            except Exception:
                # wait until that someone else updated the __object
                while self.__object is None:
                    sleep(1)
            self.__executor = None
        return self.__object

    # This is the part where we are no longer generic, but __slots__
    # inheritance is too cumbersome to actually inherit and make sure it works optimally
    def get_logger(self):
        if self.__object is not None:
            return self.__object.get_logger()

        if self.__deferred_bkg_class is None:
            # we are shutting down, wait until object is available
            return self.__result__().get_logger()

        return self.__deferred_bkg_class
