from typing import Optional

from .cache import CacheManager


class StorageManager(object):
    """
    StorageManager is helper interface for downloading & uploading files to supported remote storage
    Support remote servers: http(s)/S3/GS/Azure/File-System-Folder
    Cache is enabled by default for all downloaded remote urls/files
    """

    @classmethod
    def get_local_copy(cls, remote_url, cache_context=None):  # type: (str, Optional[str]) -> str
        """
        Get a local copy of the remote file. If the remote URL is a direct file access,
        the returned link is the same, otherwise a link to a local copy of the url file is returned.
        Caching is enabled by default, cache limited by number of stored files per cache context.
        Oldest accessed files are deleted when cache is full.

        :param str remote_url: remote url link (string)
        :param str cache_context: Optional caching context identifier (string), default context 'global'
        :return str: full path to local copy of the requested url. Return None on Error.
        """
        return CacheManager.get_cache_manager(cache_context=cache_context).get_local_copy(remote_url=remote_url)

    @classmethod
    def upload_file(cls, local_file, remote_url, wait_for_upload=True):  # type: (str, str, bool) -> str
        """
        Upload a local file to a remote location.
        remote url is the finale destination of the uploaded file.
        Examples:
            upload_file('/tmp/artifact.yaml', 'http://localhost:8081/manual_artifacts/my_artifact.yaml')
            upload_file('/tmp/artifact.yaml', 's3://a_bucket/artifacts/my_artifact.yaml')
            upload_file('/tmp/artifact.yaml', '/mnt/share/folder/artifacts/my_artifact.yaml')

        :param str local_file: Full path of a local file to be uploaded
        :param str remote_url: Full path or remote url to upload to (including file name)
        :param bool wait_for_upload: If False, return immediately and upload in the background. Default True.
        :return str: Newly uploaded remote url
        """
        return CacheManager.get_cache_manager().upload_file(
            local_file=local_file, remote_url=remote_url, wait_for_upload=wait_for_upload)

    @classmethod
    def set_cache_file_limit(cls, cache_file_limit, cache_context=None):  # type: (int, Optional[str]) -> int
        """
        Set the cache context file limit. File limit is the maximum number of files the specific cache context holds.
        Notice, there is no limit on the size of these files, only the total number of cached files.

        :param int cache_file_limit: New maximum number of cached files
        :param str cache_context: Optional cache context identifier, default global context
        :return int: Return new cache context file limit
        """
        return CacheManager.get_cache_manager(
            cache_context=cache_context, cache_file_limit=cache_file_limit).set_cache_limit(cache_file_limit)
