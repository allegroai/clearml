import shutil
import tarfile
from random import random
from time import time
from typing import Optional
from zipfile import ZipFile

from pathlib2 import Path

from .util import encode_string_to_filename
from ..debugging.log import LoggerRoot
from .cache import CacheManager


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
        :param name: name of artifact.
        :param force_download: download file from remote even if exists in local cache
        :return: Full path to local copy of the requested url. Return None on Error.
        """
        cached_file = CacheManager.get_cache_manager(
            cache_context=cache_context
        ).get_local_copy(remote_url=remote_url, force_download=force_download)
        if extract_archive and cached_file:
            return cls._extract_to_cache(cached_file, name, cache_context)

        return cached_file

    @classmethod
    def upload_file(
        cls, local_file, remote_url, wait_for_upload=True
    ):  # type: (str, str, bool) -> str
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
        :return: Newly uploaded remote URL.
        """
        return CacheManager.get_cache_manager().upload_file(
            local_file=local_file,
            remote_url=remote_url,
            wait_for_upload=wait_for_upload,
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
    def _extract_to_cache(cls, cached_file, name, cache_context=None, target_folder=None):
        # type: (str, str, Optional[str], Optional[str]) -> str
        """
        Extract cached file to cache folder
        :param str cached_file: local copy of archive file
        :param str name: name of the target file
        :param str cache_context: cache context id
        :param str target_folder: specify target path to use for archive extraction
        :return: cached folder containing the extracted archive content
        """
        if not cached_file:
            return cached_file

        cached_file = Path(cached_file)

        # we support zip and tar.gz files auto-extraction
        suffix = cached_file.suffix.lower()
        if suffix == '.gz':
            suffix = ''.join(a.lower() for a in cached_file.suffixes[-2:])

        if suffix not in (".zip", ".tgz", ".tar.gz"):
            return str(cached_file)

        cached_folder = Path(cached_file).parent
        archive_suffix = cached_file.name[:-len(suffix)]
        name = encode_string_to_filename(name)
        target_folder = Path(
            target_folder or CacheManager.get_context_folder_lookup(cache_context).format(archive_suffix, name))

        if target_folder.exists():
            # noinspection PyBroadException
            try:
                target_folder.touch(exist_ok=True)
                return target_folder.as_posix()
            except Exception:
                pass

        base_logger = LoggerRoot.get_base_logger()
        try:
            temp_target_folder = cached_folder / "{0}_{1}_{2}".format(
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
