import ctypes
import threading
import sys
import time


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

    try:
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, ctypes.py_object(exception))
    except:
        ret = 0

    # ref: http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc
    if ret == 0:
        # raise ValueError("Invalid thread ID")
        return False
    elif ret > 1:
        # Huh? Why would we notify more than one threads?
        # Because we punch a hole into C level interpreter.
        # So it is better to clean up the mess.
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, NULL)
        except:
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
