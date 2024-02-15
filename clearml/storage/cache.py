import atexit
import hashlib
import os
import shutil

from collections import OrderedDict
from threading import RLock
from typing import Union, Optional, Tuple, Dict

from pathlib2 import Path

from .helper import StorageHelper
from .util import quote_url
from ..config import get_cache_dir, deferred_config
from ..debugging.log import LoggerRoot
from ..utilities.locks.utils import Lock as FileLock
from ..utilities.locks.exceptions import LockException
from ..utilities.files import get_filename_max_length


class CacheManager(object):
    __cache_managers = {}
    _default_cache_file_limit = deferred_config(
        "storage.cache.default_cache_manager_size", 100
    )
    _storage_manager_folder = "storage_manager"
    _default_context = "global"
    _local_to_remote_url_lookup = OrderedDict()
    __local_to_remote_url_lookup_max_size = 1024
    _context_to_folder_lookup = dict()
    _default_context_folder_template = "{0}_artifacts_archive_{1}"
    _lockfile_prefix = ".lock."
    _lockfile_suffix = ".clearml"

    class CacheContext(object):
        _folder_locks = dict()  # type: Dict[str, FileLock]
        _lockfile_at_exit_cb = None

        def __init__(self, cache_context, default_cache_file_limit=10):
            # type: (str, int) -> None
            self._context = str(cache_context)
            self._file_limit = int(default_cache_file_limit)
            self._rlock = RLock()
            self._max_file_name_length = None

        def set_cache_limit(self, cache_file_limit):
            # type: (int) -> int
            self._file_limit = max(self._file_limit, int(cache_file_limit))
            return self._file_limit

        def get_local_copy(
            self, remote_url, force_download, skip_zero_size_check=False
        ):
            # type: (str, bool, bool) -> Optional[str]
            helper = StorageHelper.get(remote_url)

            if helper.base_url == "file://":
                remote_url = os.path.expanduser(remote_url)

            if not helper:
                raise ValueError("Storage access failed: {}".format(remote_url))
            # check if we need to cache the file
            try:
                # noinspection PyProtectedMember
                direct_access = helper.get_driver_direct_access(remote_url)
            except (OSError, ValueError):
                LoggerRoot.get_base_logger().debug(
                    "Failed accessing local file: {}".format(remote_url)
                )
                return None

            if direct_access:
                return direct_access

            # check if we already have the file in our cache
            cached_file, cached_size = self.get_cache_file(remote_url)
            if cached_size is not None and not force_download:
                CacheManager._add_remote_url(remote_url, cached_file)
                return cached_file
            # we need to download the file:
            downloaded_file = helper.download_to_file(
                remote_url,
                cached_file,
                overwrite_existing=force_download,
                skip_zero_size_check=skip_zero_size_check,
            )
            if downloaded_file != cached_file:
                # something happened
                return None
            CacheManager._add_remote_url(remote_url, cached_file)
            return cached_file

        @staticmethod
        def upload_file(local_file, remote_url, wait_for_upload=True, retries=3):
            # type: (str, str, bool, int) -> Optional[str]
            helper = StorageHelper.get(remote_url)
            result = helper.upload(
                local_file,
                remote_url,
                async_enable=not wait_for_upload,
                retries=retries,
            )
            CacheManager._add_remote_url(remote_url, local_file)
            return result

        @classmethod
        def get_hashed_url_file(cls, url):
            # type: (str) -> str
            str_hash = hashlib.md5(url.encode()).hexdigest()
            filename = url.split("/")[-1]
            return "{}.{}".format(str_hash, quote_url(filename))

        def _conform_filename(self, file_name):
            # type: (str) -> str
            """
            Renames very long filename by reducing characters from the end
            without the extensions from 2 floating point.
            :param file_name: base file name
            :return: new_file name (if it has very long name) or original
            """
            if self._max_file_name_length is None:
                self._max_file_name_length = get_filename_max_length(self.get_cache_folder())

            # Maximum character supported for filename
            # (FS limit) - (32 for temporary file name addition)
            allowed_length = self._max_file_name_length - 32

            if len(file_name) <= allowed_length:
                return file_name  # File name size is in limit

            file_ext = "".join(Path(file_name).suffixes[-2:])
            file_ext = file_ext.rstrip(" ")

            file_basename = file_name[:-len(file_ext)]
            file_basename = file_basename.strip()

            # Omit characters from extensionss
            if len(file_ext) > allowed_length:
                file_ext = file_ext[-(allowed_length - 1):]
                file_ext = "." + file_ext.lstrip(".")

            # Updating maximum character length
            allowed_length -= len(file_ext)

            # Omit characters from filename (without extension)
            if len(file_basename) > allowed_length:
                file_basename = file_basename[:allowed_length].strip()

            new_file_name = file_basename + file_ext

            LoggerRoot.get_base_logger().warning(
                'Renaming file to "{}" due to filename length limit'.format(new_file_name)
            )

            return new_file_name

        def get_cache_folder(self):
            # type: () -> str
            """
            :return: full path to current contexts cache folder
            """
            folder = Path(
                get_cache_dir() / CacheManager._storage_manager_folder / self._context
            )
            return folder.as_posix()

        def get_cache_file(self, remote_url=None, local_filename=None):
            # type: (Optional[str], Optional[str]) -> Tuple[str, Optional[int]]
            """
            :param remote_url: check if we have the remote url in our cache
            :param local_filename: if local_file is given, search for the local file/directory in the cache folder
            :return: full path to file name, current file size or None
            """

            def safe_time(x):
                # noinspection PyBroadException
                try:
                    return x.stat().st_mtime
                except Exception:
                    return 0

            def sort_max_access_time(x):
                atime = safe_time(x)
                # noinspection PyBroadException
                try:
                    if x.is_dir():
                        dir_files = list(x.iterdir())
                        atime = (
                            max(atime, max(safe_time(s) for s in dir_files))
                            if dir_files
                            else atime
                        )
                except Exception:
                    pass
                return atime

            folder = Path(
                get_cache_dir() / CacheManager._storage_manager_folder / self._context
            )
            folder.mkdir(parents=True, exist_ok=True)
            local_filename = local_filename or self.get_hashed_url_file(remote_url)
            local_filename = self._conform_filename(local_filename)
            new_file = folder / local_filename
            new_file_exists = new_file.exists()
            if new_file_exists:
                # noinspection PyBroadException
                try:
                    new_file.touch(exist_ok=True)
                except Exception:
                    pass
            # if file doesn't exist, return file size None
            # noinspection PyBroadException
            try:
                new_file_size = new_file.stat().st_size if new_file_exists else None
            except Exception:
                new_file_size = None

            folder_files = list(folder.iterdir())
            if len(folder_files) <= self._file_limit:
                return new_file.as_posix(), new_file_size

            # first exclude lock files
            lock_files = dict()
            files = []
            for f in sorted(folder_files, reverse=True, key=sort_max_access_time):
                if f.name.startswith(CacheManager._lockfile_prefix) and f.name.endswith(
                    CacheManager._lockfile_suffix
                ):
                    # parse the lock filename
                    name = f.name[
                        len(CacheManager._lockfile_prefix):-len(
                            CacheManager._lockfile_suffix
                        )
                    ]
                    num, _, name = name.partition(".")
                    lock_files[name] = lock_files.get(name, []) + [f.as_posix()]
                else:
                    files.append(f)

            # remove new lock files from the list (we will delete them when time comes)
            for f in files[: self._file_limit]:
                lock_files.pop(f.name, None)

            # delete old files
            files = files[self._file_limit:]
            for f in files:
                # check if the file is in the lock folder list:
                folder_lock = self._folder_locks.get(f.absolute().as_posix())
                if folder_lock:
                    # pop from lock files
                    lock_files.pop(f.name, None)
                    continue

                # check if someone else holds the lock file
                locks = lock_files.get(f.name, [])
                for lck in locks:
                    try:
                        a_lock = FileLock(filename=lck)
                        a_lock.acquire(timeout=0)
                        a_lock.release()
                        a_lock.delete_lock_file()
                        del a_lock
                    except LockException:
                        # someone have the lock skip the file
                        continue

                # if we got here we need to pop from the lock_files, later we will delete the leftover entries
                lock_files.pop(f.name, None)

                # if we are here we can delete the file
                if not f.is_dir():
                    # noinspection PyBroadException
                    try:
                        f.unlink()
                    except Exception:
                        pass
                else:
                    try:
                        shutil.rmtree(f.as_posix(), ignore_errors=False)
                    except Exception as e:
                        # failed deleting folder
                        LoggerRoot.get_base_logger().debug(
                            "Exception {}\nFailed deleting folder {}".format(e, f)
                        )

            # cleanup old lock files
            for lock_files in lock_files.values():
                for f in lock_files:
                    # noinspection PyBroadException
                    try:
                        os.unlink(f)
                    except BaseException:
                        pass

            return new_file.as_posix(), new_file_size

        def lock_cache_folder(self, local_path):
            # type: (Union[str, Path]) -> ()
            """
            Lock a specific cache folder, making sure it will not be deleted in the next
            cache cleanup round
            :param local_path: Path (str/Path) to a sub-folder inside the instance cache folder
            """
            local_path = Path(local_path).absolute()
            self._rlock.acquire()
            if self._lockfile_at_exit_cb is None:
                self._lockfile_at_exit_cb = True
                atexit.register(self._lock_file_cleanup_callback)

            lock = self._folder_locks.get(local_path.as_posix())
            i = 0
            # try to create a lock if we do not already have one (if we do, we assume it is locked)
            while not lock:
                lock_path = local_path.parent / "{}{:03d}.{}{}".format(
                    CacheManager._lockfile_prefix,
                    i,
                    local_path.name,
                    CacheManager._lockfile_suffix,
                )
                lock = FileLock(filename=lock_path)

                # try to lock folder (if we failed to create lock, try nex number)
                try:
                    lock.acquire(timeout=0)
                    break
                except LockException:
                    # failed locking, maybe someone else already locked it.
                    del lock
                    lock = None
                    i += 1

            # store lock
            self._folder_locks[local_path.as_posix()] = lock
            self._rlock.release()

        def unlock_cache_folder(self, local_path):
            # type: (Union[str, Path]) -> ()
            """
            Lock a specific cache folder, making sure it will not be deleted in the next
            cache cleanup round
            :param local_path: Path (str/Path) to a sub-folder inside the instance cache folder
            """
            local_path = Path(local_path).absolute()
            self._rlock.acquire()
            # pop lock
            lock = self._folder_locks.pop(local_path.as_posix(), None)
            if lock:
                lock.release()
                lock.delete_lock_file()
                del lock

            self._rlock.release()

        @classmethod
        def _lock_file_cleanup_callback(cls):
            for lock in cls._folder_locks.values():
                lock.release()
                lock.delete_lock_file()

    @classmethod
    def get_cache_manager(cls, cache_context=None, cache_file_limit=None):
        # type: (Optional[str], Optional[int]) -> CacheManager.CacheContext
        cache_context = cache_context or cls._default_context
        if cache_context not in cls.__cache_managers:
            cls.__cache_managers[cache_context] = cls.CacheContext(
                cache_context, cache_file_limit or cls._default_cache_file_limit
            )
        if cache_file_limit:
            cls.__cache_managers[cache_context].set_cache_limit(cache_file_limit)

        return cls.__cache_managers[cache_context]

    @staticmethod
    def get_remote_url(local_copy_path):
        # type: (str) -> str
        if not CacheManager._local_to_remote_url_lookup:
            return local_copy_path

        # noinspection PyBroadException
        try:
            conform_local_copy_path = StorageHelper.conform_url(str(local_copy_path))
        except Exception:
            return local_copy_path

        return CacheManager._local_to_remote_url_lookup.get(
            hash(conform_local_copy_path), local_copy_path
        )

    @staticmethod
    def _add_remote_url(remote_url, local_copy_path):
        # type: (str, str) -> ()
        # so that we can disable the cache lookup altogether
        if CacheManager._local_to_remote_url_lookup is None:
            return

        # noinspection PyBroadException
        try:
            remote_url = StorageHelper.conform_url(str(remote_url))
        except Exception:
            return

        if remote_url.startswith("file://"):
            return

        local_copy_path = str(local_copy_path)

        # noinspection PyBroadException
        try:
            local_copy_path = StorageHelper.conform_url(local_copy_path)
        except Exception:
            pass
        CacheManager._local_to_remote_url_lookup[hash(local_copy_path)] = remote_url
        # protect against overuse, so we do not blowup the memory
        if (
            len(CacheManager._local_to_remote_url_lookup)
            > CacheManager.__local_to_remote_url_lookup_max_size
        ):
            # pop the first item (FIFO)
            CacheManager._local_to_remote_url_lookup.popitem(last=False)

    @classmethod
    def set_context_folder_lookup(cls, context, name_template):
        # type: (str, str) -> str
        cls._context_to_folder_lookup[str(context)] = str(name_template)
        return str(name_template)

    @classmethod
    def get_context_folder_lookup(cls, context):
        # type: (Optional[str]) -> str
        if not context:
            return cls._default_context_folder_template
        return cls._context_to_folder_lookup.get(
            str(context), cls._default_context_folder_template
        )
