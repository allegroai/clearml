import os
import psutil
import sys
from multiprocessing import Process, Lock, Event as ProcessEvent
from multiprocessing.pool import ThreadPool
from threading import Thread, Event as TrEvent
from time import sleep
from typing import List, Dict

from ..py3_interop import AbstractContextManager

try:
    from multiprocessing import SimpleQueue
except ImportError:  # noqa
    from multiprocessing.queues import SimpleQueue


class SingletonThreadPool(object):
    __lock = None
    __thread_pool = None
    __thread_pool_pid = None

    @classmethod
    def get(cls):
        if os.getpid() != cls.__thread_pool_pid:
            cls.__thread_pool = ThreadPool(1)
            cls.__thread_pool_pid = os.getpid()
        return cls.__thread_pool


class SafeQueue(object):
    __thread_pool = SingletonThreadPool()

    def __init__(self, *args, **kwargs):
        self._q = SimpleQueue(*args, **kwargs)

    def empty(self):
        return self._q.empty()

    def get(self):
        return self._q.get()

    def put(self, obj):
        # make sure the block put is done in the thread pool i.e. in the background
        SafeQueue.__thread_pool.get().apply_async(self._q.put, args=(obj, ))


class SafeEvent(object):
    __thread_pool = SingletonThreadPool()

    def __init__(self):
        self._event = ProcessEvent()

    def is_set(self):
        return self._event.is_set()

    def set(self):
        if not BackgroundMonitor.is_subprocess_enabled() or BackgroundMonitor.is_subprocess_alive():
            self._event.set()
        # SafeEvent.__thread_pool.get().apply_async(func=self._event.set, args=())

    def clear(self):
        return self._event.clear()

    def wait(self, timeout=None):
        return self._event.wait(timeout=timeout)


class SingletonLock(AbstractContextManager):
    _instances = []

    def __init__(self):
        self._lock = None
        SingletonLock._instances.append(self)

    def acquire(self, *args, **kwargs):
        self.create()
        return self._lock.acquire(*args, **kwargs)

    def release(self, *args, **kwargs):
        if self._lock is None:
            return None
        return self._lock.release(*args, **kwargs)

    def create(self):
        if self._lock is None:
            self._lock = Lock()

    @classmethod
    def instantiate(cls):
        for i in cls._instances:
            i.create()

    def __enter__(self):
        """Return `self` upon entering the runtime context."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        # Do whatever cleanup.
        self.release()
        if any((exc_type, exc_value, traceback,)):
            raise (exc_type, exc_value, traceback)


class BackgroundMonitor(object):
    # If we will need multiple monitoring contexts (i.e. subprocesses) this will become a dict
    _main_process = None
    _parent_pid = None
    _sub_process_started = None
    _instances = {}  # type: Dict[int, List[BackgroundMonitor]]

    def __init__(self, task, wait_period):
        self._event = TrEvent()
        self._done_ev = TrEvent()
        self._start_ev = TrEvent()
        self._task_pid = os.getpid()
        self._thread = None
        self._wait_timeout = wait_period
        self._subprocess = None if task.is_main_task() else False
        self._task_obj_id = id(task)

    def start(self):
        if not self._thread:
            self._thread = True
        self._event.clear()
        self._done_ev.clear()
        if self._subprocess is False:
            # start the thread we are in threading mode.
            self._start()
        else:
            # append to instances
            if self not in self._get_instances():
                self._get_instances().append(self)

    def wait(self, timeout=None):
        if not self._thread:
            return
        self._done_ev.wait(timeout=timeout)

    def _start(self):
        # if we already started do nothing
        if isinstance(self._thread, Thread):
            return
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        if not self._thread:
            return

        if not self.is_subprocess() or self.is_subprocess_alive():
            self._event.set()

        if isinstance(self._thread, Thread):
            try:
                self._get_instances().remove(self)
            except ValueError:
                pass
            self._thread = None

    def daemon(self):
        while True:
            if self._event.wait(self._wait_timeout):
                break
            self._daemon_step()

    def _daemon(self):
        self._start_ev.set()
        self.daemon()
        self.post_execution()

    def post_execution(self):
        self._done_ev.set()

    def set_subprocess_mode(self):
        # called just before launching the daemon in a subprocess
        if not self._subprocess:
            self._subprocess = True
        if not isinstance(self._done_ev, SafeEvent):
            self._done_ev = SafeEvent()
        if not isinstance(self._start_ev, SafeEvent):
            self._start_ev = SafeEvent()
        if not isinstance(self._event, SafeEvent):
            self._event = SafeEvent()

    def _daemon_step(self):
        pass

    @classmethod
    def start_all(cls, task, wait_for_subprocess=False):
        # noinspection PyProtectedMember
        execute_in_subprocess = task._report_subprocess_enabled

        if not execute_in_subprocess:
            for d in BackgroundMonitor._instances.get(id(task), []):
                d._start()
        elif not BackgroundMonitor._main_process:
            cls._parent_pid = os.getpid()
            cls._sub_process_started = SafeEvent()
            cls._sub_process_started.clear()
            # setup
            for d in BackgroundMonitor._instances.get(id(task), []):
                d.set_subprocess_mode()
            BackgroundMonitor._main_process = Process(target=cls._background_process_start, args=(id(task), ))
            BackgroundMonitor._main_process.daemon = True
            BackgroundMonitor._main_process.start()
            # wait until subprocess is up
            if wait_for_subprocess:
                cls._sub_process_started.wait()

    @classmethod
    def _background_process_start(cls, task_obj_id):
        is_debugger_running = bool(getattr(sys, 'gettrace', None) and sys.gettrace())
        # restore original signal, this will prevent any deadlocks
        # Do not change the exception we need to catch base exception as well
        # noinspection PyBroadException
        try:
            from ... import Task
            # noinspection PyProtectedMember
            Task.current_task()._remove_at_exit_callbacks()
        except:  # noqa
            pass

        # if a debugger is running, wait for it to attach to the subprocess
        if is_debugger_running:
            sleep(3)

        # launch all the threads
        for d in cls._instances.get(task_obj_id, []):
            d._start()

        if cls._sub_process_started:
            cls._sub_process_started.set()

        # wait until we are signaled
        for i in BackgroundMonitor._instances.get(task_obj_id, []):
            # noinspection PyBroadException
            try:
                if i._thread and i._thread.is_alive():
                    # DO Not change, we need to catch base exception, if the process gte's killed
                    try:
                        i._thread.join()
                    except:  # noqa
                        break
                else:
                    pass
            except:  # noqa
                pass
        # we are done, leave process
        return

    def is_alive(self):
        if self.is_subprocess():
            return self.is_subprocess_alive() and self._thread \
                   and self._start_ev.is_set() and not self._done_ev.is_set()
        else:
            return isinstance(self._thread, Thread) and self._thread.is_alive()

    @classmethod
    def is_subprocess_alive(cls):
        if not cls._main_process:
            return False
        # noinspection PyBroadException
        try:
            return \
                cls._main_process.is_alive() and \
                psutil.Process(cls._main_process.pid).status() != psutil.STATUS_ZOMBIE
        except Exception:
            current_pid = cls._main_process.pid
            if not current_pid:
                return False
            try:
                parent = psutil.Process(cls._parent_pid)
            except psutil.Error:
                # could not find parent process id
                return
            for child in parent.children(recursive=True):
                # kill ourselves last (if we need to)
                if child.pid == current_pid:
                    return child.status() != psutil.STATUS_ZOMBIE
            return False

    def is_subprocess(self):
        return self._subprocess is not False and bool(self._main_process)

    def _get_instances(self):
        return self._instances.setdefault(self._task_obj_id, [])

    @classmethod
    def is_subprocess_enabled(cls):
        return bool(cls._main_process)
