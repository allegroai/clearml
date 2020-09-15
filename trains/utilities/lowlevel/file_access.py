import os
import sys
from functools import partial

import six
from typing import Optional, Any, Callable


def __buffer_writer_close_patch(self):
    self._trains_org_close()
    # noinspection PyBroadException
    try:
        self._trains_close_cb(self)
    except Exception:
        pass


def buffer_writer_close_cb(bufferwriter, callback, overwrite=False):
    # type: (Any, Callable[[Any], None], bool) -> ()
    # noinspection PyBroadException
    try:
        if not hasattr(bufferwriter, '_trains_org_close'):
            bufferwriter._trains_org_close = bufferwriter.close
            bufferwriter.close = partial(__buffer_writer_close_patch, bufferwriter)
        elif not overwrite and hasattr(bufferwriter, '_trains_close_cb'):
            return

        bufferwriter._trains_close_cb = callback
    except Exception:
        pass


def get_filename_from_file_object(file_object, flush=False, analyze_file_handle=False):
    # type: (object, bool, bool) -> Optional[str]
    """
    Return a string of the file location, extracted from any file object
    :param file_object: str, file, stream, FileIO etc.
    :param flush: If True, flush file object before returning (default: False)
    :param analyze_file_handle: If True try to retrieve filename from file handler object (default: False)
    :return: string full path of file location or None if filename cannot be extract
    """
    if isinstance(file_object, six.string_types):
        # noinspection PyBroadException
        try:
            return os.path.abspath(file_object) if file_object else file_object
        except Exception:
            return file_object
    elif hasattr(file_object, 'name'):
        filename = file_object.name
        if flush:
            # noinspection PyBroadException
            try:
                file_object.flush()
            except Exception:
                pass
        return os.path.abspath(filename)
    elif analyze_file_handle and isinstance(file_object, int) or hasattr(file_object, 'fileno'):
        # noinspection PyBroadException
        try:
            fileno = file_object if isinstance(file_object, int) else file_object.fileno()
            if sys.platform == 'win32':
                import msvcrt
                from ctypes import windll, create_string_buffer
                handle = msvcrt.get_osfhandle(fileno)
                name = create_string_buffer(2050)
                windll.kernel32.GetFinalPathNameByHandleA(handle, name, 2048, 0)
                filename = name.value.decode('utf-8')
                if filename.startswith('\\\\?\\'):
                    filename = filename[4:]
                if flush:
                    os.fsync(fileno)
                return os.path.abspath(filename)
            elif sys.platform == 'linux':
                filename = os.readlink('/proc/self/fd/{}'.format(fileno))
                if flush:
                    os.fsync(fileno)
                return os.path.abspath(filename)
            elif sys.platform == 'darwin':
                import fcntl
                name = b' ' * 1024
                # F_GETPATH = 50
                name = fcntl.fcntl(fileno, 50, name)
                filename = name.split(b'\x00')[0].decode()
                if flush:
                    os.fsync(fileno)
                return os.path.abspath(filename)
        except Exception:
            return None
    return None
