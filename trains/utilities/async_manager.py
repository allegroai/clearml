from threading import Lock
import time


class AsyncManagerMixin(object):
    _async_results_lock = Lock()
    _async_results = []

    @classmethod
    def _add_async_result(cls, result, wait_on_max_results=None, wait_time=30, wait_cb=None):
        while True:
            try:
                cls._async_results_lock.acquire()
                # discard completed results
                cls._async_results = [r for r in cls._async_results if not r.ready()]
                num_results = len(cls._async_results)
                if wait_on_max_results is not None and num_results >= wait_on_max_results:
                    # At least max_results results are still pending, wait
                    if wait_cb:
                        wait_cb(num_results)
                    if wait_time:
                        time.sleep(wait_time)
                    continue
                # add result
                if result and not result.ready():
                    cls._async_results.append(result)
                break
            finally:
                cls._async_results_lock.release()

    @classmethod
    def wait_for_results(cls, timeout=None, max_num_uploads=None):
        remaining = timeout
        count = 0
        for r in cls._async_results:
            if r.ready():
                continue
            t = time.time()
            r.wait(timeout=remaining)
            count += 1
            if max_num_uploads is not None and max_num_uploads - count <= 0:
                break
            if timeout is not None:
                remaining = max(0, remaining - max(0, time.time() - t))
                if not remaining:
                    break

    @classmethod
    def get_num_results(cls):
        if cls._async_results is not None:
            return len([r for r in cls._async_results if not r.ready()])
        else:
            return 0
