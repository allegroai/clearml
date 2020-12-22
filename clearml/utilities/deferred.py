import threading
from functools import wraps

import attr
import six


class DeferredExecutionPool(object):
    @attr.s
    class _DeferredAction(object):
        method = attr.ib()
        args = attr.ib()
        kwargs = attr.ib()

    def __init__(self, instance):
        self._instance = instance
        self._pool = []
        self._lock = threading.Lock()

    def add(self, callable_, *args, **kwargs):
        self._pool.append(self._DeferredAction(callable_, args, kwargs))

    def clear(self):
        with self._lock:
            pool = self._pool
            self._pool = []
            return pool

    def apply(self):
        pool = self.clear()
        for action in pool:
            action.method(self._instance, *action.args, **action.kwargs)

    def copy_from(self, other):
        if not isinstance(self._instance, type(other._instance)):
            raise ValueError("Copy deferred actions must be with the same instance type")

        self._pool = other._pool[:]


class ParameterizedDefaultDict(dict):
    def __init__(self, factory, *args, **kwargs):
        super(ParameterizedDefaultDict, self).__init__(*args, **kwargs)
        self._factory = factory

    def __missing__(self, key):
        self[key] = self._factory(key)
        return self[key]


class DeferredExecution(object):
    def __init__(self, pool_cls=DeferredExecutionPool):
        self._pools = ParameterizedDefaultDict(pool_cls)

    def __get__(self, instance, owner):
        if not instance:
            return self

        return self._pools[instance]

    def defer_execution(self, condition_or_attr_name=True):
        """
        Deferred execution decorator, designed to wrap class functions for classes containing a deferred execution pool.
        :param condition_or_attr_name: Condition controlling whether wrapped function should be deferred.
            True by default. If a callable is provided, it will be called with the class instance (self)
            as first argument. If a string is provided, a class instance (self) attribute by that name is evaluated.
        :return:
        """
        def decorator(func):
            @wraps(func)
            def wrapper(instance, *args, **kwargs):
                if self._resolve_condition(instance, condition_or_attr_name):
                    self._pools[instance].add(func, *args, **kwargs)
                else:
                    return func(instance, *args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def _resolve_condition(instance, condition_or_attr_name):
        if callable(condition_or_attr_name):
            return condition_or_attr_name(instance)
        elif isinstance(condition_or_attr_name, six.string_types):
            return getattr(instance, condition_or_attr_name)
        return condition_or_attr_name

    def _apply(self, instance, condition_or_attr_name):
        if self._resolve_condition(instance, condition_or_attr_name):
            self._pools[instance].apply()

    def apply_after(self, condition_or_attr_name=True):
        """
        Decorator for applying deferred execution pool after wrapped function has completed
        :param condition_or_attr_name: Condition controlling whether deferred pool should be applied. True by default.
            If a callable is provided, it will be called with the class instance (self) as first argument.
            If a string is provided, a class instance (self) attribute by that name is evaluated.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(instance, *args, **kwargs):
                res = func(instance, *args, **kwargs)
                self._apply(instance, condition_or_attr_name)
                return res
            return wrapper
        return decorator

    def apply_before(self, condition_or_attr_name=True):
        """
        Decorator for applying deferred execution pool before wrapped function is executed
        :param condition_or_attr_name: Condition controlling whether deferred pool should be applied. True by default.
            If a callable is provided, it will be called with the class instance (self) as first argument.
            If a string is provided, a class instance (self) attribute by that name is evaluated.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(instance, *args, **kwargs):
                self._apply(instance, condition_or_attr_name)
                return func(instance, *args, **kwargs)
            return wrapper
        return decorator
