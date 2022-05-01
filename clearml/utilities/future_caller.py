from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from ..errors import UsageError


class FutureCaller:
    """
    FutureCaller is used to call functions async, in another thread.

    For example:

    .. code-block:: py

        future = FutureCaller().call(max, 1, 2)
        print('Running other code')
        print(future.result())  # will print '2'
    """

    def __init__(self):
        self._executor = None
        self._future = None

    def call(self, fn, *args, **kwargs):
        # type: (Callable, *Any, **Any) -> FutureCaller
        """
        Call fn(*args, **kwargs) in another thread

        :return: This FutureCaller instance
        """
        self._executor = ThreadPoolExecutor(max_workers=1)
        if self._future:
            raise UsageError("A function is currently running in this FutureCaller instance")
        self._future = self._executor.submit(fn, *args, **kwargs)
        return self

    def result(self, timeout=None):
        # type: (Optional[float]) -> Any
        """
        Wait and get the result of the function called with self.call()

        :param timeout: The maximum number of seconds to wait for the result. If None,
            there is no limit for the wait time.

        :return: The result of the called function
        """
        if not self._executor:
            raise UsageError("No function has been called in this FutureCaller instance")
        result_ = self._future.result(timeout=timeout)
        self._future = None
        self._executor.shutdown(wait=False)
        return result_
