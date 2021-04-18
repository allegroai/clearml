import logging
import sys
import threading
from time import time

from ..binding.frameworks import _patched_call  # noqa
from ..config import running_remotely, config, DEBUG_SIMULATE_REMOTE_TASK


class StdStreamPatch(object):
    _stdout_proxy = None
    _stderr_proxy = None
    _stdout_original_write = None
    _stderr_original_write = None

    @staticmethod
    def patch_std_streams(a_logger, connect_stdout=True, connect_stderr=True):
        if (connect_stdout or connect_stderr) and not PrintPatchLogger.patched and \
                (not running_remotely() or DEBUG_SIMULATE_REMOTE_TASK.get()):
            StdStreamPatch._stdout_proxy = PrintPatchLogger(sys.stdout, a_logger, level=logging.INFO) \
                if connect_stdout else None
            StdStreamPatch._stderr_proxy = PrintPatchLogger(sys.stderr, a_logger, level=logging.ERROR) \
                if connect_stderr else None

            if StdStreamPatch._stdout_proxy:
                # noinspection PyBroadException
                try:
                    if StdStreamPatch._stdout_original_write is None:
                        StdStreamPatch._stdout_original_write = sys.stdout.write

                    # this will only work in python 3, guard it with try/catch
                    if not hasattr(sys.stdout, '_original_write'):
                        sys.stdout._original_write = sys.stdout.write
                    sys.stdout.write = StdStreamPatch._stdout__patched__write__
                except Exception:
                    pass
                sys.stdout = StdStreamPatch._stdout_proxy
                # noinspection PyBroadException
                try:
                    sys.__stdout__ = sys.stdout
                except Exception:
                    pass

            if StdStreamPatch._stderr_proxy:
                # noinspection PyBroadException
                try:
                    if StdStreamPatch._stderr_original_write is None:
                        StdStreamPatch._stderr_original_write = sys.stderr.write
                    if not hasattr(sys.stderr, '_original_write'):
                        sys.stderr._original_write = sys.stderr.write
                    sys.stderr.write = StdStreamPatch._stderr__patched__write__
                except Exception:
                    pass
                sys.stderr = StdStreamPatch._stderr_proxy

                # patch the base streams of sys (this way colorama will keep its ANSI colors)
                # noinspection PyBroadException
                try:
                    sys.__stderr__ = sys.stderr
                except Exception:
                    pass

            # now check if we have loguru and make it re-register the handlers
            # because it stores internally the stream.write function, which we cant patch
            # noinspection PyBroadException
            try:
                from loguru import logger  # noqa
                register_stderr = None
                register_stdout = None
                for k, v in logger._handlers.items():  # noqa
                    if connect_stderr and v._name == '<stderr>':  # noqa
                        register_stderr = k
                    elif connect_stdout and v._name == '<stdout>':  # noqa
                        register_stderr = k
                if register_stderr is not None:
                    logger.remove(register_stderr)
                    logger.add(sys.stderr)
                if register_stdout is not None:
                    logger.remove(register_stdout)
                    logger.add(sys.stdout)
            except Exception:
                pass

        elif (connect_stdout or connect_stderr) and not running_remotely():
            if StdStreamPatch._stdout_proxy and connect_stdout:
                StdStreamPatch._stdout_proxy.connect(a_logger)
            if StdStreamPatch._stderr_proxy and connect_stderr:
                StdStreamPatch._stderr_proxy.connect(a_logger)

    @staticmethod
    def patch_logging_formatter(a_logger, logging_handler=None):
        if not logging_handler:
            import logging
            logging_handler = logging.Handler
        logging_handler.format = _patched_call(logging_handler.format, HandlerFormat(a_logger))

    @staticmethod
    def remove_patch_logging_formatter(logging_handler=None):
        if not logging_handler:
            import logging
            logging_handler = logging.Handler
        # remove the function, Hack calling patched logging.Handler.format() returns the original function
        # noinspection PyBroadException
        try:
            logging_handler.format = logging_handler.format()  # noqa
        except Exception:
            pass

    @staticmethod
    def remove_std_logger(logger=None):
        if isinstance(sys.stdout, PrintPatchLogger):
            # noinspection PyBroadException
            try:
                sys.stdout.disconnect(logger)
            except Exception:
                pass
        if isinstance(sys.stderr, PrintPatchLogger):
            # noinspection PyBroadException
            try:
                sys.stderr.disconnect(logger)
            except Exception:
                pass

    @staticmethod
    def stdout_original_write(*args, **kwargs):
        if StdStreamPatch._stdout_original_write:
            StdStreamPatch._stdout_original_write(*args, **kwargs)
        else:
            sys.stdout.write(*args, **kwargs)

    @staticmethod
    def stderr_original_write(*args, **kwargs):
        if StdStreamPatch._stderr_original_write:
            StdStreamPatch._stderr_original_write(*args, **kwargs)
        else:
            sys.stderr.write(*args, **kwargs)

    @staticmethod
    def _stdout__patched__write__(*args, **kwargs):
        if StdStreamPatch._stdout_proxy:
            return StdStreamPatch._stdout_proxy.write(*args, **kwargs)
        return sys.stdout._original_write(*args, **kwargs)  # noqa

    @staticmethod
    def _stderr__patched__write__(*args, **kwargs):
        if StdStreamPatch._stderr_proxy:
            return StdStreamPatch._stderr_proxy.write(*args, **kwargs)
        return sys.stderr._original_write(*args, **kwargs)  # noqa


class HandlerFormat(object):
    def __init__(self, logger):
        self._logger = logger

    def __call__(self, original_format_func, *args):
        # hack get back original function, so we can remove it
        if all(a is None for a in args):
            return original_format_func
        if len(args) == 1:
            record = args[0]
            msg = original_format_func(record)
        else:
            handler = args[0]
            record = args[1]
            msg = original_format_func(handler, record)

        self._logger.report_text(msg=msg, level=record.levelno, print_console=False)
        return msg


class PrintPatchLogger(object):
    """
    Allowed patching a stream into the logger.
    Used for capturing and logging stdin and stderr when running in development mode pseudo worker.
    """
    patched = False
    lock = threading.Lock()
    recursion_protect_lock = threading.RLock()
    cr_flush_period = config.get("development.worker.console_cr_flush_period", 0)

    def __init__(self, stream, logger=None, level=logging.INFO):
        PrintPatchLogger.patched = True
        self._terminal = stream
        self._log = logger
        self._log_level = level
        self._cur_line = ''
        self._force_lf_flush = False
        self._lf_last_flush = 0

    def write(self, message):
        # make sure that we do not end up in infinite loop (i.e. log.console ends up calling us)
        if self._log and not PrintPatchLogger.recursion_protect_lock._is_owned():  # noqa
            try:
                # make sure we flush from time to time on \r
                self._test_lr_flush()

                self.lock.acquire()
                with PrintPatchLogger.recursion_protect_lock:
                    if hasattr(self._terminal, '_original_write'):
                        self._terminal._original_write(message)  # noqa
                    else:
                        self._terminal.write(message)

                do_flush = '\n' in message
                do_cr = '\r' in message
                self._cur_line += message

                if not do_flush and do_cr and PrintPatchLogger.cr_flush_period and self._force_lf_flush:
                    self._cur_line += '\n'
                    do_flush = True

                if (not do_flush and (PrintPatchLogger.cr_flush_period or not do_cr)) or not message:
                    return

                if PrintPatchLogger.cr_flush_period and self._cur_line:
                    self._cur_line = '\n'.join(line.split('\r')[-1] for line in self._cur_line.split('\n'))

                last_lf = self._cur_line.rindex('\n' if do_flush else '\r')
                next_line = self._cur_line[last_lf + 1:]
                cur_line = self._cur_line[:last_lf + 1].rstrip()
                self._cur_line = next_line
            finally:
                self.lock.release()

            if cur_line:
                self._force_lf_flush = False
                with PrintPatchLogger.recursion_protect_lock:
                    # noinspection PyBroadException
                    try:
                        if self._log:
                            # noinspection PyProtectedMember
                            self._log._console(cur_line, level=self._log_level, omit_console=True)
                    except Exception:
                        # what can we do, nothing
                        pass
        else:
            if hasattr(self._terminal, '_original_write'):
                self._terminal._original_write(message)  # noqa
            else:
                self._terminal.write(message)

    def connect(self, logger):
        self._cur_line = ''
        self._log = logger

    def disconnect(self, logger=None):
        # disconnect the logger only if it was registered
        if not logger or self._log == logger:
            self.connect(None)

    def _test_lr_flush(self):
        if not self.cr_flush_period:
            return
        if time() - self._lf_last_flush > self.cr_flush_period:
            self._force_lf_flush = True
            self._lf_last_flush = time()

    def __getattr__(self, attr):
        if attr in ['_log', '_terminal', '_log_level', '_cur_line', '_cr_overwrite',
                    '_force_lf_flush', '_lf_last_flush']:
            return self.__dict__.get(attr)
        return getattr(self._terminal, attr)

    def __setattr__(self, key, value):
        if key in ['_log', '_terminal', '_log_level', '_cur_line', '_cr_overwrite',
                   '_force_lf_flush', '_lf_last_flush']:
            self.__dict__[key] = value
        else:
            return setattr(self._terminal, key, value)
