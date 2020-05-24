import os
import sys
import time
from logging import LogRecord, getLogger, basicConfig, getLevelName, INFO, WARNING, Formatter, makeLogRecord
from logging.handlers import BufferingHandler
from threading import Thread, Event
from six.moves.queue import Queue

from ...backend_api.services import events
from ...backend_api.session.session import MaxRequestSizeError
from ...config import config

buffer_capacity = config.get('log.task_log_buffer_capacity', 100)


class TaskHandler(BufferingHandler):
    __flush_max_history_seconds = 30.
    __wait_for_flush_timeout = 10.
    __max_event_size = 1024 * 1024
    __once = False

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value

    def __init__(self, session, task_id, capacity=buffer_capacity):
        super(TaskHandler, self).__init__(capacity)
        self.task_id = task_id
        self.session = session
        self.last_timestamp = 0
        self.counter = 1
        self._last_event = None
        self._exit_event = None
        self._queue = None
        self._thread = None
        self._pending = 0

    def shouldFlush(self, record):
        """
        Should the handler flush its buffer?

        Returns true if the buffer is up to capacity. This method can be
        overridden to implement custom flushing strategies.
        """
        if self._task_id is None:
            return False

        # if we need to add handlers to the base_logger,
        # it will not automatically create stream one when first used, so we must manually configure it.
        if not TaskHandler.__once:
            base_logger = getLogger()
            if len(base_logger.handlers) == 1 and isinstance(base_logger.handlers[0], TaskHandler):
                if record.name != 'console' and not record.name.startswith('trains.'):
                    base_logger.removeHandler(self)
                    basicConfig()
                    base_logger.addHandler(self)
                    TaskHandler.__once = True
            else:
                TaskHandler.__once = True

        # if we passed the max buffer
        if len(self.buffer) >= self.capacity:
            return True

        # if the first entry in the log was too long ago.
        try:
            if len(self.buffer) and (time.time() - self.buffer[0].created) > self.__flush_max_history_seconds:
                return True
        except:
            pass

        return False

    def _record_to_event(self, record):
        # type: (LogRecord) -> events.TaskLogEvent
        if self._task_id is None:
            return None
        timestamp = int(record.created * 1000)
        if timestamp == self.last_timestamp:
            timestamp += self.counter
            self.counter += 1
        else:
            self.last_timestamp = timestamp
            self.counter = 1

        # ignore backspaces (they are often used)
        full_msg = record.getMessage().replace('\x08', '')

        return_events = []
        while full_msg:
            msg = full_msg[:self.__max_event_size]
            full_msg = full_msg[self.__max_event_size:]
            # unite all records in a single second
            if self._last_event and timestamp - self._last_event.timestamp < 1000 and \
                    len(self._last_event.msg) + len(msg) < self.__max_event_size and \
                    record.levelname.lower() == str(self._last_event.level):
                # ignore backspaces (they are often used)
                self._last_event.msg += '\n' + msg
                continue

            # if we have a previous event and it timed out, return it.
            new_event = events.TaskLogEvent(
                task=self.task_id,
                timestamp=timestamp,
                level=record.levelname.lower(),
                worker=self.session.worker,
                msg=msg
            )
            if self._last_event:
                return_events.append(self._last_event)

            self._last_event = new_event

        return return_events

    def flush(self):
        if self._task_id is None:
            return

        if not self.buffer:
            return

        self.acquire()
        if self.buffer:
            buffer = self.buffer
            self.buffer = []
        self.release()

        if not buffer:
            return

        try:
            record_events = [r for record in buffer for r in self._record_to_event(record)] + [self._last_event]
            self._last_event = None
            batch_requests = events.AddBatchRequest(requests=[events.AddRequest(e) for e in record_events if e])
        except Exception:
            self.__log_stderr("WARNING: trains.log - Failed logging task to backend ({:d} lines)".format(len(buffer)))
            batch_requests = None

        if batch_requests:
            self._pending += 1
            self._add_to_queue(batch_requests)

    def _create_thread_queue(self):
        if self._queue:
            return

        self._queue = Queue()
        self._exit_event = Event()
        self._exit_event.clear()
        # multiple workers could be supported as well
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()

    def _add_to_queue(self, request):
        self._create_thread_queue()
        self._queue.put(request)

    def close(self, wait=False):
        # self.__log_stderr('Closing {} wait={}'.format(os.getpid(), wait))
        # flush pending logs
        if not self._task_id:
            return
        # avoid deadlocks just skip the lock, we are shutting down anyway
        self.lock = None

        self.flush()
        # shut down the TaskHandler, from this point onwards. No events will be logged
        _thread = self._thread
        self._thread = None
        if self._queue:
            self._exit_event.set()
            self._queue.put(None)
        self._task_id = None

        if wait and _thread:
            try:
                timeout = 1. if self._queue.empty() else self.__wait_for_flush_timeout
                _thread.join(timeout=timeout)
                if not self._queue.empty():
                    self.__log_stderr('Flush timeout {}s exceeded, dropping last {} lines'.format(
                        timeout, self._queue.qsize()))
                # self.__log_stderr('Closing {} wait done'.format(os.getpid()))
            except:
                pass
        # call super and remove the handler
        super(TaskHandler, self).close()

    def _send_events(self, a_request):
        try:
            # if self._thread is None:
            #     self.__log_stderr('Task.close() flushing remaining logs ({})'.format(self._pending))
            self._pending -= 1
            res = self.session.send(a_request)
            if not res.ok():
                self.__log_stderr("failed logging task to backend ({:d} lines, {})".format(
                    len(a_request.requests), str(res.meta)), level=WARNING)
        except MaxRequestSizeError:
            self.__log_stderr("failed logging task to backend ({:d} lines) log size exceeded limit".format(
                len(a_request.requests)), level=WARNING)
        except Exception as ex:
            self.__log_stderr("Retrying, failed logging task to backend ({:d} lines): {}".format(
                len(a_request.requests), ex))
            # we should push ourselves back into the thread pool
            if self._queue:
                self._pending += 1
                self._queue.put(a_request)

    def _daemon(self):
        # multiple daemons are supported
        leave = self._exit_event.wait(0)
        request = True
        while not leave or request:
            # pull from queue
            request = None
            if self._queue:
                try:
                    request = self._queue.get(block=not leave)
                except:
                    pass
            if request:
                self._send_events(request)
            leave = self._exit_event.wait(0)
        # self.__log_stderr('leaving {}'.format(os.getpid()))

    @staticmethod
    def __log_stderr(msg, level=INFO):
        # output directly to stderr, make sure we do not catch it.
        write = sys.stderr._original_write if hasattr(sys.stderr, '_original_write') else sys.stderr.write
        write('{asctime} - {name} - {levelname} - {message}\n'.format(
            asctime=Formatter().formatTime(makeLogRecord({})),
            name='trains.log', levelname=getLevelName(level), message=msg))
