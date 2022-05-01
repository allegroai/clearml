import json
import sys
from pathlib2 import Path
from logging import LogRecord, getLogger, basicConfig, getLevelName, INFO, WARNING, Formatter, makeLogRecord, warning
from logging.handlers import BufferingHandler

from .development.worker import DevWorker
from ...backend_api.services import events
from ...backend_api.session.session import MaxRequestSizeError
from ...config import config
from ...utilities.process.mp import BackgroundMonitor, ForkEvent, ForkQueue
from ...utilities.process.mp import SafeQueue as PrQueue, SafeEvent


class BackgroundLogService(BackgroundMonitor):
    __max_event_size = 1024 * 1024

    def __init__(self, session, wait_period, worker=None, task=None, offline_log_filename=None):
        super(BackgroundLogService, self).__init__(task=task, wait_period=wait_period)
        self._worker = worker
        self._task_id = task.id
        self._queue = ForkQueue()
        self._flush = ForkEvent()
        self._last_event = None
        self._offline_log_filename = offline_log_filename
        self.session = session
        self.counter = 1
        self._last_timestamp = 0

    def stop(self):
        # make sure we signal the flush event before closing the queue (send everything)
        self.flush()
        if isinstance(self._queue, PrQueue):
            self._queue.close(self._event)
        super(BackgroundLogService, self).stop()

    def daemon(self):
        # multiple daemons are supported
        while not self._event.wait(0):
            self._flush.wait(self._wait_timeout)
            self._flush.clear()
            self.send_all_records()

        # flush all leftover events
        self.send_all_records()

    def _send_events(self, a_request):
        if not a_request or not a_request.requests:
            return

        try:
            if self._offline_log_filename:
                with open(self._offline_log_filename.as_posix(), 'at') as f:
                    f.write(json.dumps([b.to_dict() for b in a_request.requests]) + '\n')
                return

            # if self._thread is None:
            #     self._log_stderr('Task.close() flushing remaining logs ({})'.format(self.pending))
            res = self.session.send(a_request)
            if res and not res.ok():
                # noinspection PyProtectedMember
                TaskHandler._log_stderr("failed logging task to backend ({:d} lines, {})".format(
                    len(a_request.requests), str(res.meta)), level=WARNING)
        except MaxRequestSizeError:
            # noinspection PyProtectedMember
            TaskHandler._log_stderr("failed logging task to backend ({:d} lines) log size exceeded limit".format(
                len(a_request.requests)), level=WARNING)
        except Exception as ex:
            # noinspection PyProtectedMember
            TaskHandler._log_stderr("Retrying, failed logging task to backend ({:d} lines): {}".format(
                len(a_request.requests), ex))
            # we should push ourselves back into the thread pool
            if self._queue:
                self._queue.put(a_request)

    def set_subprocess_mode(self):
        if isinstance(self._queue, ForkQueue):
            self.send_all_records()
            self._queue = PrQueue()
        super(BackgroundLogService, self).set_subprocess_mode()
        self._flush = SafeEvent()

    def add_to_queue(self, record):
        # check that we did not loose the reporter sub-process
        if self.is_subprocess_mode() and not self._fast_is_subprocess_alive() and not self.get_at_exit_state():  # HANGS IF RACE HOLDS!
            # we lost the reporting subprocess, let's switch to thread mode
            # gel all data, work on local queue:
            self.send_all_records()
            # replace queue:
            self._queue = ForkQueue()
            self._flush = ForkEvent()
            self._event = ForkEvent()
            self._done_ev = ForkEvent()
            self._start_ev = ForkEvent()
            # set thread mode
            self._subprocess = False
            # start background thread
            self._thread = None
            self._start()
            getLogger('clearml.log').warning(
                'Event reporting sub-process lost, switching to thread based reporting')

        self._queue.put(record)

    def empty(self):
        return self._queue.empty() if self._queue else True

    def send_all_records(self):
        buffer = []
        while self._queue and not self._queue.empty():
            # noinspection PyBroadException
            try:
                request = self._queue.get(block=False)
                if request:
                    buffer.append(request)
            except Exception:
                break
        if buffer:
            self._send_records(buffer)

    def _record_to_event(self, record):
        # type: (LogRecord) -> events.TaskLogEvent
        timestamp = int(record.created * 1000)
        if timestamp == self._last_timestamp:
            timestamp += self.counter
            self.counter += 1
        else:
            self._last_timestamp = timestamp
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
                task=self._task_id,
                timestamp=timestamp,
                level=record.levelname.lower(),
                worker=self._worker,
                msg=msg
            )
            if self._last_event:
                return_events.append(self._last_event)

            self._last_event = new_event

        return return_events

    def _send_records(self, records):
        # if we have previous batch requests first send them
        buffer = []
        for r in records:
            if isinstance(r, events.AddBatchRequest):
                self._send_events(r)
            else:
                buffer.append(r)

        # noinspection PyBroadException
        try:
            record_events = [r for record in buffer for r in self._record_to_event(record)] + [self._last_event]
            self._last_event = None
            batch_requests = events.AddBatchRequest(requests=[events.AddRequest(e) for e in record_events if e])
            self._send_events(batch_requests)
        except Exception as ex:
            # noinspection PyProtectedMember
            TaskHandler._log_stderr(
                "{}\nWARNING: clearml.log - Failed logging task to backend ({:d} lines)".format(ex, len(buffer)))

    def flush(self):
        if self.is_alive():
            self._flush.set()


class TaskHandler(BufferingHandler):
    __flush_max_history_seconds = 30.
    __wait_for_flush_timeout = 10.
    __once = False
    __offline_filename = 'log.jsonl'

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value

    def __init__(self, task, capacity=None, use_subprocess=False):
        capacity = capacity or config.get('log.task_log_buffer_capacity', 100)
        super(TaskHandler, self).__init__(capacity)
        self.task_id = task.id
        self.worker = task.session.worker
        self.counter = 0
        self._offline_log_filename = None
        if task.is_offline():
            offline_folder = Path(task.get_offline_mode_folder())
            offline_folder.mkdir(parents=True, exist_ok=True)
            self._offline_log_filename = offline_folder / self.__offline_filename
        self._background_log = BackgroundLogService(
            worker=task.session.worker, task=task,
            session=task.session, wait_period=float(DevWorker.report_period),
            offline_log_filename=self._offline_log_filename)
        self._background_log_size = 0
        if use_subprocess:
            self._background_log.set_subprocess_mode()
        self._background_log.start()

    def emit(self, record):
        self.counter += 1
        if self._background_log:
            self._background_log.add_to_queue(record)
            self._background_log_size += 1

    def shouldFlush(self, record):
        """
        Should the handler flush its buffer

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
        return (self.counter >= self.capacity and self._background_log and
                self._background_log_size >= self.capacity)

    def flush(self):
        if self._task_id is None:
            return
        self.counter = 0
        if self._background_log:
            self._background_log.flush()
            self._background_log_size = 0

    def close(self, wait=False):
        # self._log_stderr('Closing {} wait={}'.format(os.getpid(), wait))
        # flush pending logs
        if not self._task_id:
            return
        # avoid deadlocks just skip the lock, we are shutting down anyway
        self.lock = None
        self._task_id = None

        # shut down the TaskHandler, from this point onwards. No events will be logged
        _background_log = self._background_log
        self._background_log = None
        if _background_log:
            if not _background_log.is_subprocess_mode() or _background_log.is_alive():
                _background_log.stop()
                if wait and (not _background_log.is_subprocess_mode() or
                             _background_log.is_subprocess_mode_and_parent_process()):
                    # noinspection PyBroadException
                    try:
                        timeout = 1. if _background_log.empty() else self.__wait_for_flush_timeout
                        _background_log.wait(timeout=timeout)
                        if not _background_log.empty():
                            self._log_stderr('Flush timeout {}s exceeded, dropping last {} lines'.format(
                                timeout, self._background_log_size))
                        # self._log_stderr('Closing {} wait done'.format(os.getpid()))
                    except Exception:
                        pass
            else:
                _background_log.send_all_records()

        # call super and remove the handler
        super(TaskHandler, self).close()

    @classmethod
    def report_offline_session(cls, task, folder, iteration_offset=0):
        filename = Path(folder) / cls.__offline_filename
        if not filename.is_file():
            return False
        with open(filename.as_posix(), 'rt') as f:
            i = 0
            while True:
                try:
                    line = f.readline()
                    if not line:
                        break
                    list_requests = json.loads(line)
                    for r in list_requests:
                        r.pop('task', None)
                        # noinspection PyBroadException
                        try:
                            r["iter"] += iteration_offset
                        except Exception:
                            pass
                    i += 1
                except StopIteration:
                    break
                except Exception as ex:
                    warning('Failed reporting log, line {} [{}]'.format(i, ex))
                batch_requests = events.AddBatchRequest(
                    requests=[events.TaskLogEvent(task=task.id, **r) for r in list_requests])
                if batch_requests.requests:
                    res = task.session.send(batch_requests)
                    if res and not res.ok():
                        warning("failed logging task to backend ({:d} lines, {})".format(
                            len(batch_requests.requests), str(res.meta)))
        return True

    @staticmethod
    def _log_stderr(msg, level=INFO):
        # output directly to stderr, make sure we do not catch it.
        # noinspection PyProtectedMember
        write = sys.stderr._original_write if hasattr(sys.stderr, '_original_write') else sys.stderr.write
        write('{asctime} - {name} - {levelname} - {message}\n'.format(
            asctime=Formatter().formatTime(makeLogRecord({})),
            name='clearml.log', levelname=getLevelName(level), message=msg))
