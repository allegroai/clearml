import os
import pickle
import struct
import sys
from functools import partial
from multiprocessing import Process, Lock, Event as ProcessEvent
from multiprocessing.pool import ThreadPool
from threading import Thread, Event as TrEvent
from time import sleep
from typing import List, Dict

import psutil
from six.moves.queue import Empty, Queue as TrQueue

from ..py3_interop import AbstractContextManager

try:
    from multiprocessing import SimpleQueue
except ImportError:  # noqa
    from multiprocessing.queues import SimpleQueue


class SingletonThreadPool(object):
    __thread_pool = None
    __thread_pool_pid = None

    @classmethod
    def get(cls):
        if os.getpid() != cls.__thread_pool_pid:
            cls.__thread_pool = ThreadPool(1)
            cls.__thread_pool_pid = os.getpid()
        return cls.__thread_pool

    @classmethod
    def clear(cls):
        if cls.__thread_pool:
            cls.__thread_pool.close()
        cls.__thread_pool = None
        cls.__thread_pool_pid = None


class SafeQueue(object):
    """
    Many writers Single Reader multiprocessing safe Queue
    """
    __thread_pool = SingletonThreadPool()

    def __init__(self, *args, **kwargs):
        self._reader_thread = None
        self._q = SimpleQueue(*args, **kwargs)
        # Fix the simple queue write so it uses a single OS write, making it atomic message passing
        # noinspection PyBroadException
        try:
            self._q._writer._send_bytes = partial(SafeQueue._pipe_override_send_bytes, self._q._writer)
        except Exception:
            pass
        self._internal_q = None
        self._q_size = 0

    def empty(self):
        return self._q.empty() and (not self._internal_q or self._internal_q.empty())

    def is_pending(self):
        # only call from main put process
        return self._q_size > 0 or not self.empty()

    def close(self, event):
        # wait until all pending requests pushed
        while self.is_pending():
            if event:
                event.set()
            sleep(0.1)

    def get(self, *args, **kwargs):
        return self._get_internal_queue(*args, **kwargs)

    def batch_get(self, max_items=1000, timeout=0.2, throttle_sleep=0.1):
        buffer = []
        timeout_count = int(timeout/throttle_sleep)
        empty_count = timeout_count
        while len(buffer) < max_items:
            while not self.empty() and len(buffer) < max_items:
                try:
                    buffer.append(self._get_internal_queue(block=False))
                    empty_count = 0
                except Empty:
                    break
            empty_count += 1
            if empty_count > timeout_count or len(buffer) >= max_items:
                break
            sleep(throttle_sleep)
        return buffer

    def put(self, obj):
        # GIL will make sure it is atomic
        self._q_size += 1
        # make sure the block put is done in the thread pool i.e. in the background
        obj = pickle.dumps(obj)
        self.__thread_pool.get().apply_async(self._q_put, args=(obj, ))

    def _q_put(self, obj):
        self._q.put(obj)
        # GIL will make sure it is atomic
        self._q_size -= 1

    def _get_internal_queue(self, *args, **kwargs):
        if not self._internal_q:
            self._internal_q = TrQueue()
        if not self._reader_thread:
            self._reader_thread = Thread(target=self._reader_daemon)
            self._reader_thread.daemon = True
            self._reader_thread.start()
        obj = self._internal_q.get(*args, **kwargs)
        # deserialize
        return pickle.loads(obj)

    def _reader_daemon(self):
        # pull from process queue and push into thread queue
        while True:
            # noinspection PyBroadException
            try:
                obj = self._q.get()
                if obj is None:
                    break
            except Exception:
                break
            self._internal_q.put(obj)

    @staticmethod
    def _pipe_override_send_bytes(self, buf):
        n = len(buf)
        # For wire compatibility with 3.2 and lower
        header = struct.pack("!i", n)
        # Issue #20540: concatenate before sending, to avoid delays due
        # to Nagle's algorithm on a TCP socket.
        # Also note we want to avoid sending a 0-length buffer separately,
        # to avoid "broken pipe" errors if the other end closed the pipe.
        self._send(header + buf)


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
        if not self._done_ev:
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
        self._thread = None

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
            # Hack allow to create daemon subprocesses (even though python doesn't like it)
            un_daemonize = False
            # noinspection PyBroadException
            try:
                from multiprocessing import current_process
                if current_process()._config.get('daemon'):  # noqa
                    un_daemonize = current_process()._config.get('daemon')  # noqa
                    current_process()._config['daemon'] = False  # noqa
            except BaseException:
                pass
            # try to start the background process, if we fail retry again, or crash
            for i in range(4):
                try:
                    BackgroundMonitor._main_process.start()
                    break
                except BaseException:
                    if i < 3:
                        sleep(1)
                        continue
                    raise
            if un_daemonize:
                # noinspection PyBroadException
                try:
                    from multiprocessing import current_process
                    current_process()._config['daemon'] = un_daemonize  # noqa
                except BaseException:
                    pass
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

    @classmethod
    def clear_main_process(cls):
        BackgroundMonitor._main_process = None
        BackgroundMonitor._parent_pid = None
        BackgroundMonitor._sub_process_started = None
        BackgroundMonitor._instances = {}
        SingletonThreadPool.clear()
