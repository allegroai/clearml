# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import tarfile
import zipfile
import re
import string
import io


class Archive(object):
    """Archive provides a consistent interface for unpacking
    compressed file.
    """

    def __init__(self, filename, fileobj):
        self._filename = filename
        self._fileobj = fileobj
        self._file = None
        self._names = None
        self._read = None

    @property
    def filename(self):
        return self._filename

    @property
    def names(self):
        """If name list is not required, do not get it."""
        if self._file is None:
            self._prepare()
        if not hasattr(self, '_namelist'):
            self._namelist = self._names()
        return self._namelist

    def close(self):
        """Close file object."""
        if self._file is not None:
            self._file.close()
        if hasattr(self, '_namelist'):
            del self._namelist
        self._filename = self._fileobj = None
        self._file = self._names = self._read = None

    def read(self, filename):
        """Read one file from archive."""
        if self._file is None:
            self._prepare()
        return self._read(filename)

    def unpack(self, to_path):
        """Unpack compressed files to path."""
        if self._file is None:
            self._prepare()
        self._safe_extractall(to_path)

    def _prepare(self):
        if self._filename.endswith(('.tar.gz', '.tar.bz2', '.tar.xz')):
            self._prepare_tarball()
        # An .egg file is actually just a .zip file
        # with a different extension, .whl too.
        elif self._filename.endswith(('.zip', '.egg', '.whl')):
            self._prepare_zip()
        else:
            raise ValueError("unreadable: {0}".format(self._filename))

    def _safe_extractall(self, to_path='.'):
        unsafe = []
        for name in self.names:
            if not self.is_safe(name):
                unsafe.append(name)
        if unsafe:
            raise ValueError("unsafe to unpack: {}".format(unsafe))
        self._file.extractall(to_path)

    def _prepare_zip(self):
        self._file = zipfile.ZipFile(self._fileobj)
        self._names = self._file.namelist
        self._read = self._file.read

    def _prepare_tarball(self):
        # tarfile has no read method
        def _read(filename):
            f = self._file.extractfile(filename)
            return f.read()

        self._file = tarfile.open(mode='r:*', fileobj=self._fileobj)
        self._names = self._file.getnames
        self._read = _read

    def is_safe(self, filename):
        return not (filename.startswith(("/", "\\")) or
                    (len(filename) > 1 and filename[1] == ":" and
                     filename[0] in string.ascii_letter) or
                    re.search(r"[.][.][/\\]", filename))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def top_level(url, data):
    """Read top level names from compressed file."""
    sb = io.BytesIO(data)
    txt = None
    with Archive(url, sb) as archive:
        file = None
        for name in archive.names:
            if name.lower().endswith('top_level.txt'):
                file = name
                break
        if file:
            txt = archive.read(file).decode('utf-8')
    sb.close()
    return [name.replace('/', '.') for name in txt.splitlines()] if txt else []
