import hashlib
import shutil

from collections import OrderedDict
from pathlib2 import Path

from .helper import StorageHelper
from .util import quote_url
from ..config import get_cache_dir
from ..debugging.log import LoggerRoot


class CacheManager(object):
    __cache_managers = {}
    _default_cache_file_limit = 100
    _storage_manager_folder = "storage_manager"
    _default_context = "global"
    _local_to_remote_url_lookup = OrderedDict()
    __local_to_remote_url_lookup_max_size = 1024

    class CacheContext(object):
        def __init__(self, cache_context, default_cache_file_limit=10):
            self._context = str(cache_context)
            self._file_limit = int(default_cache_file_limit)

        def set_cache_limit(self, cache_file_limit):
            self._file_limit = max(self._file_limit, int(cache_file_limit))
            return self._file_limit

        def get_local_copy(self, remote_url):
            helper = StorageHelper.get(remote_url)
            if not helper:
                raise ValueError("Storage access failed: {}".format(remote_url))
            # check if we need to cache the file
            try:
                # noinspection PyProtectedMember
                direct_access = helper._driver.get_direct_access(remote_url)
            except (OSError, ValueError):
                LoggerRoot.get_base_logger().warning("Failed accessing local file: {}".format(remote_url))
                return None

            if direct_access:
                return direct_access

            # check if we already have the file in our cache
            cached_file, cached_size = self._get_cache_file(remote_url)
            if cached_size is not None:
                CacheManager._add_remote_url(remote_url, cached_file)
                return cached_file
            # we need to download the file:
            downloaded_file = helper.download_to_file(remote_url, cached_file)
            if downloaded_file != cached_file:
                # something happened
                return None
            CacheManager._add_remote_url(remote_url, cached_file)
            return cached_file

        @staticmethod
        def upload_file(local_file, remote_url, wait_for_upload=True):
            helper = StorageHelper.get(remote_url)
            result = helper.upload(
                local_file, remote_url, async_enable=not wait_for_upload
            )
            CacheManager._add_remote_url(remote_url, local_file)
            return result

        @classmethod
        def _get_hashed_url_file(cls, url):
            str_hash = hashlib.md5(url.encode()).hexdigest()
            filename = url.split("/")[-1]
            return "{}.{}".format(str_hash, quote_url(filename))

        def _get_cache_file(self, remote_url):
            """
            :param remote_url: check if we have the remote url in our cache
            :return: full path to file name, current file size or None
            """
            folder = Path(
                get_cache_dir() / CacheManager._storage_manager_folder / self._context
            )
            folder.mkdir(parents=True, exist_ok=True)
            local_filename = self._get_hashed_url_file(remote_url)
            new_file = folder / local_filename
            if new_file.exists():
                new_file.touch(exist_ok=True)

            # delete old files
            def sort_max_access_time(x):
                atime = x.stat().st_atime
                # noinspection PyBroadException
                try:
                    if x.is_dir():
                        dir_files = list(x.iterdir())
                        atime = (
                            max(atime, max(s.stat().st_atime for s in dir_files))
                            if dir_files
                            else atime
                        )
                except Exception:
                    pass
                return atime

            files = sorted(folder.iterdir(), reverse=True, key=sort_max_access_time)
            files = files[self._file_limit:]
            for f in files:
                if not f.is_dir():
                    f.unlink()
                else:
                    try:
                        shutil.rmtree(f)
                    except Exception as e:
                        # failed deleting folder
                        LoggerRoot.get_base_logger().warning(
                            "Exception {}\nFailed deleting folder {}".format(e, f)
                        )

            # if file doesn't exist, return file size None
            return (
                new_file.as_posix(),
                new_file.stat().st_size if new_file.exists() else None,
            )

    @classmethod
    def get_cache_manager(cls, cache_context=None, cache_file_limit=None):
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
        if not CacheManager._local_to_remote_url_lookup:
            return local_copy_path

        # noinspection PyBroadException
        try:
            conform_local_copy_path = StorageHelper.conform_url(str(local_copy_path))
        except Exception:
            return local_copy_path

        return CacheManager._local_to_remote_url_lookup.get(hash(conform_local_copy_path), local_copy_path)

    @staticmethod
    def _add_remote_url(remote_url, local_copy_path):
        # so that we can disable the cache lookup altogether
        if CacheManager._local_to_remote_url_lookup is None:
            return

        # noinspection PyBroadException
        try:
            remote_url = StorageHelper.conform_url(str(remote_url))
        except Exception:
            return

        if remote_url.startswith('file://'):
            return

        local_copy_path = str(local_copy_path)

        # noinspection PyBroadException
        try:
            local_copy_path = StorageHelper.conform_url(local_copy_path)
        except Exception:
            pass
        CacheManager._local_to_remote_url_lookup[hash(local_copy_path)] = remote_url
        # protect against overuse, so we do not blowup the memory
        if len(CacheManager._local_to_remote_url_lookup) > CacheManager.__local_to_remote_url_lookup_max_size:
            # pop the first item (FIFO)
            CacheManager._local_to_remote_url_lookup.popitem(last=False)
