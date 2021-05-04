import fnmatch
import os
import shutil
import tarfile
from multiprocessing.pool import ThreadPool
from random import random
from time import time
from typing import List, Optional
from zipfile import ZipFile

from pathlib2 import Path

from .cache import CacheManager
from .helper import StorageHelper
from .util import encode_string_to_filename
from ..debugging.log import LoggerRoot


class StorageManager(object):
    """
    StorageManager is helper interface for downloading & uploading files to supported remote storage
    Support remote servers: http(s)/S3/GS/Azure/File-System-Folder
    Cache is enabled by default for all downloaded remote urls/files
    """

    @classmethod
    def get_local_copy(
        cls, remote_url, cache_context=None, extract_archive=True, name=None, force_download=False,
    ):
        # type: (str, Optional[str], bool, Optional[str], bool) -> str
        """
        Get a local copy of the remote file. If the remote URL is a direct file access,
        the returned link is the same, otherwise a link to a local copy of the url file is returned.
        Caching is enabled by default, cache limited by number of stored files per cache context.
        Oldest accessed files are deleted when cache is full.

        :param str remote_url: remote url link (string)
        :param str cache_context: Optional caching context identifier (string), default context 'global'
        :param bool extract_archive: if True returned path will be a cached folder containing the archive's content,
            currently only zip files are supported.
        :param str name: name of the target file
        :param bool force_download: download file from remote even if exists in local cache
        :return: Full path to local copy of the requested url. Return None on Error.
        """
        cache = CacheManager.get_cache_manager(cache_context=cache_context)
        cached_file = cache.get_local_copy(remote_url=remote_url, force_download=force_download)
        if extract_archive and cached_file:
            # this will get us the actual cache (even with direct access)
            cache_path_encoding = Path(cache.get_cache_folder()) / cache.get_hashed_url_file(remote_url)
            return cls._extract_to_cache(
                cached_file, name, cache_context, cache_path_encoding=cache_path_encoding.as_posix())

        return cached_file

    @classmethod
    def upload_file(
        cls, local_file, remote_url, wait_for_upload=True, retries=1
    ):  # type: (str, str, bool, int) -> str
        """
        Upload a local file to a remote location. remote url is the finale destination of the uploaded file.

        Examples:

        .. code-block:: py

            upload_file('/tmp/artifact.yaml', 'http://localhost:8081/manual_artifacts/my_artifact.yaml')
            upload_file('/tmp/artifact.yaml', 's3://a_bucket/artifacts/my_artifact.yaml')
            upload_file('/tmp/artifact.yaml', '/mnt/share/folder/artifacts/my_artifact.yaml')

        :param str local_file: Full path of a local file to be uploaded
        :param str remote_url: Full path or remote url to upload to (including file name)
        :param bool wait_for_upload: If False, return immediately and upload in the background. Default True.
        :param int retries: Number of retries before failing to upload file, default 1.
        :return: Newly uploaded remote URL.
        """
        return CacheManager.get_cache_manager().upload_file(
            local_file=local_file,
            remote_url=remote_url,
            wait_for_upload=wait_for_upload,
            retries=retries,
        )

    @classmethod
    def set_cache_file_limit(
        cls, cache_file_limit, cache_context=None
    ):  # type: (int, Optional[str]) -> int
        """
        Set the cache context file limit. File limit is the maximum number of files the specific cache context holds.
        Notice, there is no limit on the size of these files, only the total number of cached files.

        :param int cache_file_limit: New maximum number of cached files
        :param str cache_context: Optional cache context identifier, default global context
        :return: The new cache context file limit.
        """
        return CacheManager.get_cache_manager(
            cache_context=cache_context, cache_file_limit=cache_file_limit
        ).set_cache_limit(cache_file_limit)

    @classmethod
    def _extract_to_cache(cls, cached_file, name, cache_context=None, target_folder=None, cache_path_encoding=None):
        # type: (str, str, Optional[str], Optional[str], Optional[str]) -> str
        """
        Extract cached file to cache folder
        :param str cached_file: local copy of archive file
        :param str name: name of the target file
        :param str cache_context: cache context id
        :param str target_folder: specify target path to use for archive extraction
        :param str cache_path_encoding: specify representation of the local path of the cached files,
            this will always point to local cache folder, even if we have direct access file.
            Used for extracting the cached archived based on cache_path_encoding
        :return: cached folder containing the extracted archive content
        """
        if not cached_file:
            return cached_file

        cached_file = Path(cached_file)
        cache_path_encoding = Path(cache_path_encoding) if cache_path_encoding else None

        # we support zip and tar.gz files auto-extraction
        suffix = cached_file.suffix.lower()
        if suffix == '.gz':
            suffix = ''.join(a.lower() for a in cached_file.suffixes[-2:])

        if suffix not in (".zip", ".tgz", ".tar.gz"):
            return str(cached_file)

        cache_folder = Path(cache_path_encoding or cached_file).parent
        archive_suffix = (cache_path_encoding or cached_file).name[:-len(suffix)]
        name = encode_string_to_filename(name) if name else name
        if target_folder:
            target_folder = Path(target_folder)
        else:
            target_folder = cache_folder / CacheManager.get_context_folder_lookup(
                cache_context).format(archive_suffix, name)

        if target_folder.is_dir():
            # noinspection PyBroadException
            try:
                target_folder.touch(exist_ok=True)
                return target_folder.as_posix()
            except Exception:
                pass

        base_logger = LoggerRoot.get_base_logger()
        try:
            temp_target_folder = cache_folder / "{0}_{1}_{2}".format(
                target_folder.name, time() * 1000, str(random()).replace('.', ''))
            temp_target_folder.mkdir(parents=True, exist_ok=True)
            if suffix == ".zip":
                ZipFile(cached_file.as_posix()).extractall(path=temp_target_folder.as_posix())
            elif suffix == ".tar.gz":
                with tarfile.open(cached_file.as_posix()) as file:
                    file.extractall(temp_target_folder.as_posix())
            elif suffix == ".tgz":
                with tarfile.open(cached_file.as_posix(), mode='r:gz') as file:
                    file.extractall(temp_target_folder.as_posix())

            # we assume we will have such folder if we already extract the file
            # noinspection PyBroadException
            try:
                # if rename fails, it means that someone else already manged to extract the file, delete the current
                # folder and return the already existing cached zip folder
                shutil.move(temp_target_folder.as_posix(), target_folder.as_posix())
            except Exception:
                if target_folder.exists():
                    target_folder.touch(exist_ok=True)
                else:
                    base_logger.warning(
                        "Failed renaming {0} to {1}".format(temp_target_folder.as_posix(), target_folder.as_posix()))
                try:
                    shutil.rmtree(temp_target_folder.as_posix())
                except Exception as ex:
                    base_logger.warning(
                        "Exception {}\nFailed deleting folder {}".format(ex, temp_target_folder.as_posix()))
        except Exception as ex:
            # failed extracting the file:
            base_logger.warning(
                "Exception {}\nFailed extracting zip file {}".format(ex, cached_file.as_posix()))
            # noinspection PyBroadException
            try:
                target_folder.rmdir()
            except Exception:
                pass
            return cached_file.as_posix()
        return target_folder.as_posix()

    @classmethod
    def get_files_server(cls):
        from ..backend_api import Session
        return Session.get_files_server_host()

    @classmethod
    def upload_folder(cls, local_folder, remote_url, match_wildcard=None):
        # type: (str, str, Optional[str]) -> None
        """
        Upload local folder recursively to a remote storage, maintaining the sub folder structure
        in the remote storage.

        .. note::

            If we have a local file `~/folder/sub/file.ext` then
            `StorageManager.upload_folder('~/folder/', 's3://bucket/')`
            will create `s3://bucket/sub/file.ext`

        :param str local_folder: Local folder to recursively upload
        :param str remote_url: Target remote storage location, tree structure of `local_folder` will
            be created under the target remote_url. Supports Http/S3/GS/Azure and shared filesystem.
            Example: 's3://bucket/data/'
        :param str match_wildcard: If specified only upload files matching the `match_wildcard`
            Example: `*.json`
            Notice: target file size/date are not checked. Default True, always upload.
            Notice if uploading to http, we will always overwrite the target.
        """

        base_logger = LoggerRoot.get_base_logger()

        if not Path(local_folder).is_dir():
            base_logger.error("Local folder '{}' does not exist".format(local_folder))
            return
        results = []
        helper = StorageHelper.get(remote_url)
        with ThreadPool() as pool:
            for path in Path(local_folder).rglob(match_wildcard or "*"):
                if not path.is_file():
                    continue
                results.append(
                    pool.apply_async(
                        helper.upload,
                        args=(str(path), str(path).replace(local_folder, remote_url)),
                    )
                )
            for res in results:
                res.wait()

    @classmethod
    def download_folder(
        cls, remote_url, local_folder=None, match_wildcard=None, overwrite=False
    ):
        # type: (str, Optional[str], Optional[str], bool) -> Optional[str]
        """
        Download remote folder recursively to the local machine, maintaining the sub folder structure
        from the remote storage.

        .. note::

            If we have a local file `s3://bucket/sub/file.ext` then
            `StorageManager.download_folder('s3://bucket/', '~/folder/')`
            will create `~/folder/sub/file.ext`

        :param str remote_url: Source remote storage location, tree structure of `remote_url` will
            be created under the target local_folder. Supports S3/GS/Azure and shared filesystem.
            Example: 's3://bucket/data/'
        :param str local_folder: Local target folder to create the full tree from remote_url.
            If None, use the cache folder. (Default: use cache folder)
        :param match_wildcard: If specified only download files matching the `match_wildcard`
            Example: `*.json`
        :param bool overwrite: If False, and target files exist do not download.
            If True always download the remote files. Default False.

        :return: Target local folder
        """

        base_logger = LoggerRoot.get_base_logger()

        if local_folder:
            try:
                Path(local_folder).mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                base_logger.error("Failed creating local folder '{}': {}".format(local_folder, ex))
                return
        else:
            local_folder = CacheManager.get_cache_manager().get_cache_folder()

        helper = StorageHelper.get(remote_url)
        results = []

        with ThreadPool() as pool:
            for path in helper.list(prefix=remote_url):
                remote_path = os.path.join(helper.base_url, path)
                if match_wildcard and not fnmatch.fnmatch(remote_path, match_wildcard):
                    continue
                local_url = remote_path.replace(remote_url, local_folder)
                if not os.path.exists(local_url) or os.path.getsize(local_url) == 0:
                    results.append(
                        pool.apply_async(
                            helper.download_to_file,
                            args=(remote_path, local_url),
                            kwds={"overwrite_existing": overwrite},
                        )
                    )
            for res in results:
                res.wait()

        return local_folder

    @classmethod
    def list(cls, remote_url):
        # type: (str) -> Optional[List[str]]
        """
        Return a list of object names inside the base path

        :param str remote_url: The base path.
            For Google Storage, Azure and S3 it is the bucket of the path, for local files it is the root directory.
            For example: AWS S3: `s3://bucket/folder_` will list all the files you have in
            `s3://bucket-name/folder_*/*`. The same behaviour with Google Storage: `gs://bucket/folder_`,
            Azure blob storage: `azure://bucket/folder_` and also file system listing: `/mnt/share/folder_`

        :return: The paths of all the objects in the storage base path under prefix, relative to the base path.
            None in case of list operation is not supported (http and https protocols for example)
        """
        helper = StorageHelper.get(remote_url)
        try:
            names_list = helper.list(prefix=remote_url)
        except Exception as ex:
            LoggerRoot.get_base_logger().warning("Can not list files for '{}' - {}".format(remote_url, ex))
            names_list = None
        return names_list
