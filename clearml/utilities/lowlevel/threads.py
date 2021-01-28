import ctypes
import threading
import six
import sys
import time


def get_current_thread_id():
    return threading._get_ident() if six.PY2 else threading.get_ident()


# Nasty hack to raise exception for other threads
def _lowlevel_async_raise(thread_obj, exception=None):
    NULL = 0
    found = False
    target_tid = 0
    for tid, tobj in threading._active.items():
        if tobj is thread_obj:
            found = True
            target_tid = tid
            break

    if not found:
        # raise ValueError("Invalid thread object")
        return False

    if not exception:
        exception = SystemExit()

    if sys.version_info.major >= 3 and sys.version_info.minor >= 7:
        target_tid = ctypes.c_ulong(target_tid)
        NULL = ctypes.c_ulong(NULL)
    else:
        target_tid = ctypes.c_long(target_tid)
        NULL = ctypes.c_long(NULL)

    # noinspection PyBroadException
    try:
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, ctypes.py_object(exception))
    except Exception:
        ret = 0

    # ref: http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc
    if ret == 0:
        # raise ValueError("Invalid thread ID")
        return False
    elif ret > 1:
        # Huh? Why would we notify more than one threads?
        # Because we punch a hole into C level interpreter.
        # So it is better to clean up the mess.
        # noinspection PyBroadException
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, NULL)
        except Exception:
            pass
        # raise SystemError("PyThreadState_SetAsyncExc failed")
        return False

    # print("Successfully set asynchronized exception for", target_tid)
    return True


def kill_thread(thread_obj, wait=False):
    if not _lowlevel_async_raise(thread_obj, SystemExit()):
        return False

    while wait and thread_obj.is_alive():
        time.sleep(0.1)
    return True


def __wait_thread(a_thread, a_event):
    # noinspection PyBroadException
    try:
        a_thread.join()
        a_event.set()
    except Exception:
        pass


def threadpool_waited_join(thread_object, timeout):
    """
    Call threadpool.join() with timeout. If join completed return True, otherwise False
    Notice: This function creates another daemon thread and kills it, use with care.

    :param thread_object: Thread to join
    :param float timeout: timeout in seconds for the join operation to complete
    :return: True os join() completed
    """
    if not thread_object:
        return True

    if isinstance(thread_object, threading.Thread):
        thread_object.join(timeout=timeout)
        return not thread_object.is_alive()

    done_signal = threading.Event()
    waitable = threading.Thread(target=__wait_thread, args=(thread_object, done_signal,))
    waitable.daemon = True
    waitable.start()

    if not done_signal.wait(timeout=timeout):
        kill_thread(waitable)
        return False
    return True


if __name__ == '__main__':
    def demo_thread(*_, **__):
        from time import sleep
        for i in range(5):
            print('.')
            sleep(1.)

    t = threading.Thread(target=demo_thread)
    t.daemon = True
    t.start()
    print(threadpool_waited_join(t, 2.0))
