import os
import sys
import threading
import inspect
import time
import zipfile

__stream_write = None
__stream_flush = None
__patched_trace = False
__trace_level = 1
__trace_start = 0
__thread_id = None
__thread_so = None


def _thread_linux_id():
    # System dependent, see e.g. /usr/include/x86_64-linux-gnu/asm/unistd_64.h (system call 186)
    return __thread_so.syscall(186)


def _thread_py_id():
    # return threading.get_ident()
    return zipfile.crc32(int(threading.get_ident()).to_bytes(8, 'little'))


def _log_stderr(name, fnc, args, kwargs, is_return):
    global __stream_write, __stream_flush, __trace_level, __trace_start, __thread_id
    try:
        if is_return and __trace_level not in (-1, -2):
            return
        if __trace_level not in (1, 2, -1, -2):
            return
        fnc_address = str(fnc).split(' at ')
        fnc_address = '{}'.format(fnc_address[-1].replace('>', '')) if len(fnc_address) > 1 else ''
        if __trace_level == 1 or __trace_level == -1:
            t = '{:14} {}'.format(fnc_address, name)
        elif __trace_level == 2 or __trace_level == -2:
            a_args = str(args)[1:-1] if args else ''
            a_kwargs = ' {}'.format(kwargs) if kwargs else ''
            t = '{:14} {} ({}{})'.format(fnc_address, name, a_args, a_kwargs)
        # get a nicer thread id
        h = int(__thread_id())
        ts = time.time() - __trace_start
        __stream_write('{}{:<9.3f}:{:5}:{:8x}: [{}] {}\n'.format(
            '-' if is_return else '', ts, os.getpid(),
            h, threading.current_thread().name, t))
        if __stream_flush:
            __stream_flush()
    except:
        pass


def _traced_call_method(name, fnc):
    def _traced_call_int(self, *args, **kwargs):
        _log_stderr(name, fnc, args, kwargs, False)
        r = None
        try:
            ret = fnc(self, *args, **kwargs)
        except Exception as ex:
            r = ex
        _log_stderr(name, fnc, args, kwargs, True)
        if r:
            raise r
        return ret
    return _traced_call_int


def _traced_call_cls(name, fnc):
    class WrapperClass(object):
        @classmethod
        def _traced_call_int(cls, *args, **kwargs):
            _log_stderr(name, fnc, args, kwargs, False)
            r = None
            try:
                ret = fnc(*args, **kwargs)
            except Exception as ex:
                r = ex
            _log_stderr(name, fnc, args, kwargs, True)
            if r:
                raise r
            return ret

    return WrapperClass.__dict__['_traced_call_int']


def _traced_call_static(name, fnc):
    class WrapperStatic(object):
        @staticmethod
        def _traced_call_int(*args, **kwargs):
            _log_stderr(name, fnc, args, kwargs, False)
            r = None
            try:
                ret = fnc(*args, **kwargs)
            except Exception as ex:
                r = ex
            _log_stderr(name, fnc, args, kwargs, True)
            if r:
                raise r
            return ret
    return WrapperStatic.__dict__['_traced_call_int']


def _traced_call_func(name, fnc):
    def _traced_call_int(*args, **kwargs):
        _log_stderr(name, fnc, args, kwargs, False)
        r = None
        try:
            ret = fnc(*args, **kwargs)
        except Exception as ex:
            r = ex
        _log_stderr(name, fnc, args, kwargs, True)
        if r:
            raise r
        return ret
    return _traced_call_int


def _patch_module(module, prefix='', basepath=None, basemodule=None):
    if isinstance(module, str):
        if basemodule is None:
            basemodule = module + '.'
            import importlib
            importlib.import_module(module)
        module = sys.modules.get(module)
        if not module:
            return
        if not basepath:
            basepath = os.path.sep.join(module.__file__.split(os.path.sep)[:-1]) + os.path.sep

    # only sub modules
    if not hasattr(module, '__file__') or (inspect.ismodule(module) and not module.__file__.startswith(basepath)):
        if hasattr(module, '__module__') and module.__module__.startswith(basemodule):
            # this is one of ours
            pass
        else:
            # print('Skipping: {}'.format(module))
            return

    # Do not patch ourselves
    if hasattr(module, '__file__') and module.__file__ == __file__:
        return

    prefix += module.__name__.split('.')[-1] + '.'

    # Do not patch low level network layer
    if prefix.startswith('trains.backend_api.session.') and prefix != 'trains.backend_api.session.':
        if not prefix.endswith('.Session.') and '.token_manager.' not in prefix:
            # print('SKIPPING: {}'.format(prefix))
            return
    if prefix.startswith('trains.backend_api.services.'):
        return

    for fn in (m for m in dir(module) if not m.startswith('__')):
        if fn in ('schema_property') or fn.startswith('_PostImportHookPatching__'):
            continue
        try:
            fnc = getattr(module, fn)
        except:
            continue
        if inspect.ismodule(fnc):
            _patch_module(fnc, prefix=prefix, basepath=basepath, basemodule=basemodule)
        elif inspect.isclass(fnc):
            _patch_module(fnc, prefix=prefix, basepath=basepath, basemodule=basemodule)
        elif inspect.isroutine(fnc):
            pass  # _log_stderr('Patching: {}'.format(prefix+fn))
            if inspect.isclass(module):
                # check if this is even in our module
                if hasattr(fnc, '__module__') and fnc.__module__ != module.__module__:
                    pass  # print('not ours {} {}'.format(module, fnc))
                elif hasattr(fnc, '__qualname__') and fnc.__qualname__.startswith(module.__name__ + '.'):
                    if isinstance(module.__dict__[fn], classmethod):
                        setattr(module, fn, _traced_call_cls(prefix + fn, fnc))
                    elif isinstance(module.__dict__[fn], staticmethod):
                        setattr(module, fn, _traced_call_static(prefix + fn, fnc))
                    else:
                        setattr(module, fn, _traced_call_method(prefix + fn, fnc))
                else:
                    # probably not ours hopefully static function
                    if hasattr(fnc, '__qualname__') and not fnc.__qualname__.startswith(module.__name__ + '.'):
                        pass  # print('not ours {} {}'.format(module, fnc))
                    else:
                        # we should not get here
                        setattr(module, fn, _traced_call_static(prefix + fn, fnc))
            elif inspect.ismodule(module):
                setattr(module, fn, _traced_call_func(prefix + fn, fnc))
            else:
                # we should not get here
                setattr(module, fn, _traced_call_func(prefix + fn, fnc))


def trace_trains(stream=None, level=1):
    """
    DEBUG ONLY - Add full Trains package code trace
    Output trace to filename or stream, default is sys.stderr
    Trace level
        -2: Trace function and arguments and returned call
        -1: Trace function call (no arguments) and returned call
        0: Trace disabled
        1: Trace function call (no arguments). This is the default
        2: Trace function and arguments

    :param stream: stream or filename for trace log (default stderr)
    :param int level: Trace level
    """
    global __patched_trace, __stream_write, __stream_flush, __trace_level, __trace_start, __thread_id, __thread_so
    __trace_level = level
    if __patched_trace:
        return
    __patched_trace = True
    if not __thread_id:
        if sys.platform == 'linux':
            import ctypes
            __thread_so = ctypes.cdll.LoadLibrary('libc.so.6')
            __thread_id = _thread_linux_id
        else:
            __thread_id = _thread_py_id

    stderr_write = sys.stderr._original_write if hasattr(sys.stderr, '_original_write') else sys.stderr.write
    if stream:
        if isinstance(stream, str):
            stream = open(stream, 'w')
        __stream_write = stream.write
        __stream_flush = stream.flush
    else:
        __stream_write = stderr_write
        __stream_flush = None

    from ..version import __version__
    msg = 'Trains v{} - Starting Trace\n\n'.format(__version__)
    # print to actual stderr
    stderr_write(msg)
    # store to stream
    __stream_write(msg)
    __stream_write('{:9}:{:5}:{:8}: {:14}\n'.format('seconds', 'pid', 'tid', 'self'))
    __stream_write('{:9}:{:5}:{:8}:{:15}\n'.format('-' * 9, '-' * 5, '-' * 8, '-' * 15))
    __trace_start = time.time()

    _patch_module('trains')


def trace_level(level=1):
    """
    Set trace level
        -2: Trace function and arguments and returned call
        -1: Trace function call (no arguments) and returned call
        0: Trace disabled
        1: Trace function call (no arguments). This is the default
        2: Trace function and arguments

    :param int level: Trace level
    :return: True if trace level changed
    """
    global __patched_trace, __trace_level
    if not __patched_trace:
        return False
    __trace_level = level
    return True


def print_traced_files(glob_mask, lines_per_tid=5, stream=sys.stdout, specify_pids=None):
    """
    Collect trace lines from files (glob mask), sort by pid/tid and print ordered by time

    :param glob_mask: file list to process ('*.txt')
    :param lines_per_tid: number of lines per pid/tid to print
    :param stream: output file stream, can accept file stream or filename(str). default is sys.stdout
    :param specify_pids: optional list of pids to include
    """
    from glob import glob

    def hash_line(a_line):
        return hash(':'.join(a_line.split(':')[1:]))

    pids = {}
    orphan_calls = set()
    print_orphans = False
    for fname in glob(glob_mask, recursive=False):
        with open(fname, 'rt') as fd:
            lines = fd.readlines()
        for l in lines:
            try:
                _, pid, tid = l.split(':')[:3]
                pid = int(pid)
            except:
                continue
            if specify_pids and pid not in specify_pids:
                continue

            if l.startswith('-'):
                print_orphans = True
                l = l[1:]
                h = hash_line(l)
                if h in orphan_calls:
                    orphan_calls.remove(h)
                    continue
            else:
                h = hash_line(l)
                orphan_calls.add(h)

            tids = pids.get(pid) if pid in pids else {}
            tids[tid] = (tids.get(tid, []) + [l])[-lines_per_tid:]
            pids[pid] = tids

    # sort by time stamp
    by_time = {}
    for p, tids in pids.items():
        for t, lines in tids.items():
            ts = float(lines[-1].split(':')[0].strip()) + 0.000001 * len(by_time)
            if print_orphans:
                for i, l in enumerate(lines):
                    if i > 0 and hash_line(l) in orphan_calls:
                        lines[i] = ' ### Orphan ### {}'.format(l)
            by_time[ts] = ''.join(lines) + '\n'

    out_stream = open(stream, 'w') if isinstance(stream, str) else stream
    for k in sorted(by_time.keys()):
        out_stream.write(by_time[k] + '\n')
    if isinstance(stream, str):
        out_stream.close()


def end_of_program():
    # stub
    pass


if __name__ == '__main__':
    # from trains import Task
    # task = Task.init(project_name="examples", task_name="trace test")
    # trace_trains('_trace.txt', level=2)
    print_traced_files('_trace_*.txt', lines_per_tid=10)
