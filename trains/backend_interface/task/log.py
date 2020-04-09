import sys
import time
from logging import LogRecord, getLogger, basicConfig
from logging.handlers import BufferingHandler
from multiprocessing.pool import ThreadPool

from ...backend_api.services import events
from ...config import config

buffer_capacity = config.get('log.task_log_buffer_capacity', 100)


class TaskHandler(BufferingHandler):
    __flush_max_history_seconds = 30.
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
        self._thread_pool = None
        self._pending = 0

    def shouldFlush(self, record):
        """
        Should the handler flush its buffer?

        Returns true if the buffer is up to capacity. This method can be
        overridden to implement custom flushing strategies.
        """
        if self._task_id is None:
            return False
        # Notice! protect against infinite loops, i.e. flush while sending previous records
        # if self.lock._is_owned():
        #     return False

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
        if len(self.buffer) and (time.time() - self.buffer[0].created) > self.__flush_max_history_seconds:
            return True

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

        # unite all records in a single second
        if self._last_event and timestamp - self._last_event.timestamp < 1000 and \
                record.levelname.lower() == str(self._last_event.level):
            # ignore backspaces (they are often used)
            self._last_event.msg += '\n' + record.getMessage().replace('\x08', '')
            return None

        self._last_event = events.TaskLogEvent(
            task=self.task_id,
            timestamp=timestamp,
            level=record.levelname.lower(),
            worker=self.session.worker,
            msg=record.getMessage().replace('\x08', '')  # ignore backspaces (they are often used)
        )
        return self._last_event

    def flush(self):
        if self._task_id is None:
            return

        if not self.buffer:
            return

        self.acquire()
        if not self.buffer:
            self.release()
            return
        buffer = self.buffer
        self.buffer = []
        try:
            record_events = [self._record_to_event(record) for record in buffer]
            self._last_event = None
            batch_requests = events.AddBatchRequest(requests=[events.AddRequest(e) for e in record_events if e])
        except Exception:
            # print("Failed logging task to backend ({:d} lines)".format(len(buffer)))
            batch_requests = None

        if batch_requests:
            if not self._thread_pool:
                self._thread_pool = ThreadPool(processes=1)
            self._pending += 1
            self._thread_pool.apply_async(self._send_events, args=(batch_requests, ))

        self.release()

    def wait_for_flush(self, shutdown=False):
        msg = 'Task.log.wait_for_flush: %d'
        ll = self.__log_stderr
        ll(msg % 0)
        self.acquire()
        ll(msg % 1)
        if self._thread_pool:
            ll(msg % 2)
            t = self._thread_pool
            ll(msg % 3)
            self._thread_pool = None
            ll(msg % 4)
            try:
                ll(msg % 5)
                t.close()
                ll(msg % 6)
                t.join()
                ll(msg % 7)
            except Exception:
                ll(msg % 8)
                pass
        if shutdown:
            ll(msg % 9)
            self._task_id = None
        ll(msg % 10)
        self.release()
        ll(msg % 11)

    def close(self, wait=True):
        # super already calls self.flush()
        super(TaskHandler, self).close()
        # shut down the TaskHandler, from this point onwards. No events will be logged
        if not wait:
            self.acquire()
            self._thread_pool = None
            self._task_id = None
            self.release()
        else:
            self.wait_for_flush(shutdown=True)

    def _send_events(self, a_request):
        try:
            if self._thread_pool is None:
                self.__log_stderr('WARNING: trains.Task - '
                                  'Task.close() flushing remaining logs ({})'.format(self._pending))
            self._pending -= 1
            res = self.session.send(a_request)
            if not res.ok():
                self.__log_stderr("WARNING: trains.log._send_events: failed logging task to backend "
                                  "({:d} lines, {})".format(len(a_request.requests), str(res.meta)))
        except Exception as ex:
            self.__log_stderr("WARNING: trains.log._send_events: Retrying, "
                              "failed logging task to backend ({:d} lines): {}".format(len(a_request.requests), ex))
            # we should push ourselves back into the thread pool
            if self._thread_pool:
                self._pending += 1
                self._thread_pool.apply_async(self._send_events, args=(a_request, ))

    @staticmethod
    def __log_stderr(t):
        if hasattr(sys.stderr, '_original_write'):
            sys.stderr._original_write(t + '\n')
        else:
            sys.stderr.write(t + '\n')
