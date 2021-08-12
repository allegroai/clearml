import os
import pickle
import struct
import sys
from functools import partial
from multiprocessing import Lock, Event as ProcessEvent
from threading import Thread, Event as TrEvent
from time import sleep, time
from typing import List, Dict, Optional
from multiprocessing import Process, get_context

import psutil
from six.moves.queue import Empty, Queue as TrQueue

from ..py3_interop import AbstractContextManager

try:
    from multiprocessing import SimpleQueue
except ImportError:
    from multiprocessing.queues import SimpleQueue

try:
    from multiprocessing.context import ForkContext  # noqa
except ImportError:
    ForkContext = None


class ThreadCalls(object):
    def __init__(self):
        self._queue = TrQueue()
        self._thread = Thread(target=self._worker)
        self._thread.daemon = True
        self._thread.start()

    def is_alive(self):
        return bool(self._thread)

    def apply_async(self, func, args=None):
        if not func:
            return False
        self._queue.put((func, args))
        return True

    def close(self):
        if not self._thread:
            return
        t = self._thread
        # push something into queue so it knows this is the end
        self._queue.put(None)
        # wait fot thread
        t.join()
        # mark thread is done
        self._thread = None

    def _worker(self):
        while True:
            try:
                request = self._queue.get(block=True, timeout=1.0)
                if not request:
                    break
            except Empty:
                continue
            # noinspection PyBroadException
            try:
                if request[1]:
                    request[0](*request[1])
                else:
                    request[0]()
            except Exception:
                pass


class SingletonThreadPool(object):
    __thread_pool = None
    __thread_pool_pid = None

    @classmethod
    def get(cls):
        if os.getpid() != cls.__thread_pool_pid:
            cls.__thread_pool = ThreadCalls()
            cls.__thread_pool_pid = os.getpid()
        return cls.__thread_pool

    @classmethod
    def clear(cls):
        if cls.__thread_pool:
            cls.__thread_pool.close()
        cls.__thread_pool = None
        cls.__thread_pool_pid = None

    @classmethod
    def is_active(cls):
        return cls.__thread_pool and cls.__thread_pool.is_alive()


class SafeQueue(object):
    """
    Many writers Single Reader multiprocessing safe Queue
    """
    __thread_pool = SingletonThreadPool()

    def __init__(self, *args, **kwargs):
        self._reader_thread = None
        self._reader_thread_started = False
        self._q = SimpleQueue(*args, **kwargs)
        # Fix the simple queue write so it uses a single OS write, making it atomic message passing
        # noinspection PyBroadException
        try:
            # noinspection PyUnresolvedReferences,PyProtectedMember
            self._q._writer._send_bytes = partial(SafeQueue._pipe_override_send_bytes, self._q._writer)
        except Exception:
            pass
        self._internal_q = None
        self._q_size = 0

    def empty(self):
        return self._q.empty() and (not self._internal_q or self._internal_q.empty())

    def is_pending(self):
        # check if we have pending requests to be pushed (it does not mean they were pulled)
        # only call from main put process
        return self._q_size > 0

    def close(self, event, timeout=100.0):
        # wait until all pending requests pushed
        tic = time()
        while self.is_pending():
            if event:
                event.set()
            if not self.__thread_pool.is_active():
                break
            sleep(0.1)
            if timeout and (time()-tic) > timeout:
                break

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

    def _init_reader_thread(self):
        if not self._internal_q:
            self._internal_q = TrQueue()
        if not self._reader_thread or not self._reader_thread.is_alive():
            # read before we start the thread
            self._reader_thread = Thread(target=self._reader_daemon)
            self._reader_thread.daemon = True
            self._reader_thread.start()
            # if we have waiting results
            # wait until thread is up and pushed some results
            while not self._reader_thread_started:
                sleep(0.2)
            # just in case make sure we pulled some stuff if we had any
            # todo: wait until a queue is not empty, but for some reason that might fail
            sleep(1.0)

    def _get_internal_queue(self, *args, **kwargs):
        self._init_reader_thread()
        obj = self._internal_q.get(*args, **kwargs)
        # deserialize
        return pickle.loads(obj)

    def _reader_daemon(self):
        self._reader_thread_started = True
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
    _main_process_task_id = None
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
        self._task_id = task.id
        self._task_obj_id = id(task.id)

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
    def start_all(cls, task, wait_for_subprocess=True):
        # noinspection PyProtectedMember
        execute_in_subprocess = task._report_subprocess_enabled

        if not execute_in_subprocess:
            for d in BackgroundMonitor._instances.get(id(task.id), []):
                d._start()
        elif not BackgroundMonitor._main_process:
            cls._parent_pid = os.getpid()
            cls._sub_process_started = SafeEvent()
            cls._sub_process_started.clear()
            cls._main_process_task_id = task.id
            # setup
            for d in BackgroundMonitor._instances.get(id(task.id), []):
                d.set_subprocess_mode()

            # todo: solve for standalone spawn subprocess
            if ForkContext is not None and isinstance(get_context(), ForkContext):
                cls.__start_subprocess_forkprocess(task_obj_id=id(task.id))
            else:
                cls.__start_subprocess_os_fork(task_obj_id=id(task.id))

            # wait until subprocess is up
            if wait_for_subprocess:
                cls._sub_process_started.wait()

    @classmethod
    def __start_subprocess_os_fork(cls, task_obj_id):
        process_args = (task_obj_id, cls._sub_process_started, os.getpid())
        BackgroundMonitor._main_process = os.fork()
        # check if we are the child process
        if BackgroundMonitor._main_process == 0:
            # update to the child process pid
            BackgroundMonitor._main_process = os.getpid()
            cls._background_process_start(*process_args)
            # force to leave the subprocess
            leave_process(0)
            return

    @classmethod
    def __start_subprocess_forkprocess(cls, task_obj_id):
        _main_process = Process(
            target=cls._background_process_start,
            args=(task_obj_id, cls._sub_process_started, os.getpid())
        )
        _main_process.daemon = True
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
                _main_process.start()
                break
            except BaseException:
                if i < 3:
                    sleep(1)
                    continue
                raise
        BackgroundMonitor._main_process = _main_process.pid
        if un_daemonize:
            # noinspection PyBroadException
            try:
                from multiprocessing import current_process
                current_process()._config['daemon'] = un_daemonize  # noqa
            except BaseException:
                pass

    @classmethod
    def _background_process_start(cls, task_obj_id, event_start=None, parent_pid=None):
        # type: (int, Optional[SafeEvent], Optional[int]) -> None
        is_debugger_running = bool(getattr(sys, 'gettrace', None) and sys.gettrace())
        # make sure we update the pid to our own
        cls._main_process = os.getpid()
        # restore original signal, this will prevent any deadlocks
        # Do not change the exception we need to catch base exception as well
        # noinspection PyBroadException
        try:
            from ... import Task
            # make sure we do not call Task.current_task() it will create a Task object for us on a subprocess!
            # noinspection PyProtectedMember
            if Task._has_current_task_obj():
                # noinspection PyProtectedMember
                Task.current_task()._remove_at_exit_callbacks()
        except:  # noqa
            pass

        # if a debugger is running, wait for it to attach to the subprocess
        if is_debugger_running:
            sleep(3)

        instances = BackgroundMonitor._instances.get(task_obj_id, [])
        # launch all the threads
        for d in instances:
            d._start()

        if cls._sub_process_started:
            cls._sub_process_started.set()

        if event_start:
            event_start.set()

        # wait until we are signaled
        for i in instances:
            # DO NOT CHANGE, we need to catch base exception, if the process gte's killed
            try:
                while i._thread and i._thread.is_alive():
                    # noinspection PyBroadException
                    try:
                        p = psutil.Process(parent_pid)
                        parent_alive = p.is_running() and p.status() != psutil.STATUS_ZOMBIE
                    except Exception:
                        parent_alive = False

                    # if parent process is not here we should just leave!
                    if not parent_alive:
                        return

                    # DO NOT CHANGE, we need to catch base exception, if the process gte's killed
                    try:
                        # timeout so we can detect if the parent process got killed.
                        i._thread.join(timeout=30.)
                    except:  # noqa
                        break
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
    def is_subprocess_alive(cls, task=None):
        if not cls._main_process or (task and cls._main_process_task_id != task.id):
            return False
        # noinspection PyBroadException
        try:
            p = psutil.Process(cls._main_process)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except Exception:
            current_pid = cls._main_process
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
                    return child.is_running() and child.status() != psutil.STATUS_ZOMBIE
            return False

    def is_subprocess(self):
        return self._subprocess is not False and \
               bool(self._main_process) and self._task_id == self._main_process_task_id

    def _get_instances(self):
        return self._instances.setdefault(self._task_obj_id, [])

    @classmethod
    def is_subprocess_enabled(cls, task=None):
        return bool(cls._main_process) and (not task or task.id == cls._main_process_task_id)

    @classmethod
    def clear_main_process(cls, task):
        if BackgroundMonitor._main_process_task_id != task.id:
            return
        cls.wait_for_sub_process(task)
        BackgroundMonitor._main_process = None
        BackgroundMonitor._main_process_task_id = None
        BackgroundMonitor._parent_pid = None
        BackgroundMonitor._sub_process_started = None
        BackgroundMonitor._instances = {}
        SingletonThreadPool.clear()

    @classmethod
    def wait_for_sub_process(cls, task, timeout=None):
        if not cls.is_subprocess_enabled(task=task):
            return

        for d in BackgroundMonitor._instances.get(id(task.id), []):
            d.stop()

        tic = time()
        while cls.is_subprocess_alive(task=task) and (not timeout or time()-tic < timeout):
            sleep(0.03)


def leave_process(status=0):
    # type: (int) -> None
    """
    Exit current process with status-code (status)
    :param status: int exit code
    """
    try:
        sys.exit(status or 0)
    except:   # noqa
        # ipython/jupyter notebook will not allow to call sys.exit
        # we have to call the low level function
        os._exit(status or 0)  # noqa
