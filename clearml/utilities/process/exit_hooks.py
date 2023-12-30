import atexit
import os
import signal
import sys

import six

from ...logger import Logger


class ExitHooks(object):
    _orig_exit = None
    _orig_exc_handler = None
    remote_user_aborted = False

    def __init__(self, callback):
        self.exit_code = None
        self.exception = None
        self.signal = None
        self._exit_callback = callback
        self._org_handlers = {}
        self._signal_recursion_protection_flag = False
        self._except_recursion_protection_flag = False
        self._import_bind_path = os.path.join("clearml", "binding", "import_bind.py")

    def update_callback(self, callback):
        if self._exit_callback and not six.PY2:
            # noinspection PyBroadException
            try:
                atexit.unregister(self._exit_callback)
            except Exception:
                pass
        self._exit_callback = callback
        if callback:
            self.hook()
        else:
            # un register int hook
            if self._orig_exc_handler:
                sys.excepthook = self._orig_exc_handler
                self._orig_exc_handler = None
            for h in self._org_handlers:
                # noinspection PyBroadException
                try:
                    signal.signal(h, self._org_handlers[h])
                except Exception:
                    pass
            self._org_handlers = {}

    def hook(self):
        if self._orig_exit is None:
            self._orig_exit = sys.exit
            sys.exit = self.exit

        if self._exit_callback:
            atexit.register(self._exit_callback)

    def register_signal_and_exception_hooks(self):
        if self._orig_exc_handler is None:
            self._orig_exc_handler = sys.excepthook

        sys.excepthook = self.exc_handler

        if not self._org_handlers:
            if sys.platform == "win32":
                catch_signals = [
                    signal.SIGINT,
                    signal.SIGTERM,
                    signal.SIGSEGV,
                    signal.SIGABRT,
                    signal.SIGILL,
                    signal.SIGFPE,
                ]
            else:
                catch_signals = [
                    signal.SIGINT,
                    signal.SIGTERM,
                    signal.SIGSEGV,
                    signal.SIGABRT,
                    signal.SIGILL,
                    signal.SIGFPE,
                    signal.SIGQUIT,
                ]
            for c in catch_signals:
                # noinspection PyBroadException
                try:
                    self._org_handlers[c] = signal.getsignal(c)
                    signal.signal(c, self.signal_handler)
                except Exception:
                    pass

    def remove_signal_hooks(self):
        for org_handler_k, org_handler_v in self._org_handlers.items():
            # noinspection PyBroadException
            try:
                signal.signal(org_handler_k, org_handler_v)
            except Exception:
                pass
        self._org_handlers = {}

    def remove_exception_hooks(self):
        if self._orig_exc_handler:
            sys.excepthook = self._orig_exc_handler
            self._orig_exc_handler = None

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exctype, value, traceback, *args, **kwargs):
        if self._except_recursion_protection_flag or not self._orig_exc_handler:
            # noinspection PyArgumentList
            return sys.__excepthook__(exctype, value, traceback, *args, **kwargs)

        self._except_recursion_protection_flag = True
        self.exception = value

        try:
            # remove us from import errors
            if six.PY3 and isinstance(exctype, type) and issubclass(exctype, ImportError):
                prev = cur = traceback
                while cur is not None:
                    tb_next = cur.tb_next
                    # if this is the import frame, we should remove it
                    if cur.tb_frame.f_code.co_filename.endswith(self._import_bind_path):
                        # remove this frame by connecting the previous one to the next one
                        prev.tb_next = tb_next
                        cur.tb_next = None
                        del cur
                        cur = prev

                    prev = cur
                    cur = tb_next
        except:  # noqa
            pass

        if self._orig_exc_handler:
            # noinspection PyArgumentList
            ret = self._orig_exc_handler(exctype, value, traceback, *args, **kwargs)
        else:
            # noinspection PyNoneFunctionAssignment, PyArgumentList
            ret = sys.__excepthook__(exctype, value, traceback, *args, **kwargs)
        self._except_recursion_protection_flag = False

        return ret

    def signal_handler(self, sig, frame):
        org_handler = self._org_handlers.get(sig)
        if not org_handler:
            return signal.SIG_DFL

        self.signal = sig
        signal.signal(sig, org_handler or signal.SIG_DFL)

        # if this is a sig term, we wait until __at_exit is called (basically do nothing)
        if sig == signal.SIGINT:
            # return original handler result
            return org_handler if not callable(org_handler) else org_handler(sig, frame)

        if self._signal_recursion_protection_flag:
            # call original
            os.kill(os.getpid(), sig)
            return org_handler if not callable(org_handler) else org_handler(sig, frame)

        self._signal_recursion_protection_flag = True

        # call exit callback
        if self._exit_callback:
            # noinspection PyBroadException
            try:
                self._exit_callback()
            except Exception:
                pass

        # remove stdout logger, just in case
        # noinspection PyBroadException
        try:
            # noinspection PyProtectedMember
            Logger._remove_std_logger()
        except Exception:
            pass

        # noinspection PyUnresolvedReferences
        os.kill(os.getpid(), sig)

        self._signal_recursion_protection_flag = False
        # return handler result
        return org_handler if not callable(org_handler) else org_handler(sig, frame)
