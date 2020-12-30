import hashlib
import shutil

from collections import OrderedDict
from pathlib2 import Path

from .helper import StorageHelper
from .util import quote_url
from ..config import get_cache_dir, config
from ..debugging.log import LoggerRoot


class CacheManager(object):
    __cache_managers = {}
    _default_cache_file_limit = config.get("storage.cache.default_cache_manager_size", 100)
    _storage_manager_folder = "storage_manager"
    _default_context = "global"
    _local_to_remote_url_lookup = OrderedDict()
    __local_to_remote_url_lookup_max_size = 1024
    _context_to_folder_lookup = dict()
    _default_context_folder_template = "{0}_artifacts_archive_{1}"

    class CacheContext(object):
        def __init__(self, cache_context, default_cache_file_limit=10):
            self._context = str(cache_context)
            self._file_limit = int(default_cache_file_limit)

        def set_cache_limit(self, cache_file_limit):
            self._file_limit = max(self._file_limit, int(cache_file_limit))
            return self._file_limit

        def get_local_copy(self, remote_url, force_download):
            helper = StorageHelper.get(remote_url)
            if not helper:
                raise ValueError("Storage access failed: {}".format(remote_url))
            # check if we need to cache the file
            try:
                # noinspection PyProtectedMember
                direct_access = helper._driver.get_direct_access(remote_url)
            except (OSError, ValueError):
                LoggerRoot.get_base_logger().debug("Failed accessing local file: {}".format(remote_url))
                return None

            if direct_access:
                return direct_access

            # check if we already have the file in our cache
            cached_file, cached_size = self.get_cache_file(remote_url)
            if cached_size is not None and not force_download:
                CacheManager._add_remote_url(remote_url, cached_file)
                return cached_file
            # we need to download the file:
            downloaded_file = helper.download_to_file(remote_url, cached_file, overwrite_existing=force_download)
            if downloaded_file != cached_file:
                # something happened
                return None
            CacheManager._add_remote_url(remote_url, cached_file)
            return cached_file

        @staticmethod
        def upload_file(local_file, remote_url, wait_for_upload=True, retries=1):
            helper = StorageHelper.get(remote_url)
            result = helper.upload(
                local_file, remote_url, async_enable=not wait_for_upload, retries=retries,
            )
            CacheManager._add_remote_url(remote_url, local_file)
            return result

        @classmethod
        def get_hashed_url_file(cls, url):
            str_hash = hashlib.md5(url.encode()).hexdigest()
            filename = url.split("/")[-1]
            return "{}.{}".format(str_hash, quote_url(filename))

        def get_cache_folder(self):
            """
            :return: full path to current contexts cache folder
            """
            folder = Path(
                get_cache_dir() / CacheManager._storage_manager_folder / self._context
            )
            return folder.as_posix()

        def get_cache_file(self, remote_url=None, local_filename=None):
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
            new_file = folder / local_filename
            new_file_exists = new_file.exists()
            if new_file_exists:
                # noinspection PyBroadException
                try:
                    new_file.touch(exist_ok=True)
                except Exception:
                    pass

            # delete old files
            files = sorted(folder.iterdir(), reverse=True, key=sort_max_access_time)
            files = files[self._file_limit:]
            for f in files:
                if not f.is_dir():
                    # noinspection PyBroadException
                    try:
                        f.unlink()
                    except Exception:
                        pass
                else:
                    try:
                        shutil.rmtree(f)
                    except Exception as e:
                        # failed deleting folder
                        LoggerRoot.get_base_logger().debug(
                            "Exception {}\nFailed deleting folder {}".format(e, f)
                        )

            # if file doesn't exist, return file size None
            # noinspection PyBroadException
            try:
                size = new_file.stat().st_size if new_file_exists else None
            except Exception:
                size = None
            return new_file.as_posix(), size

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

    @classmethod
    def set_context_folder_lookup(cls, context, name_template):
        cls._context_to_folder_lookup[str(context)] = str(name_template)
        return str(name_template)

    @classmethod
    def get_context_folder_lookup(cls, context):
        if not context:
            return cls._default_context_folder_template
        return cls._context_to_folder_lookup.get(str(context), cls._default_context_folder_template)
