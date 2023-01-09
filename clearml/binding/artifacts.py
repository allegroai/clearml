import json
import yaml
import mimetypes
import os
import pickle
from six.moves.urllib.parse import quote
from copy import deepcopy
from datetime import datetime
from multiprocessing.pool import ThreadPool
from tempfile import mkdtemp, mkstemp
from threading import Thread
from time import time
from zipfile import ZipFile, ZIP_DEFLATED

import six
from PIL import Image
from pathlib2 import Path
from six.moves.urllib.parse import urlparse
from typing import Dict, Union, Optional, Any, Sequence, Callable

from ..backend_api import Session
from ..backend_api.services import tasks
from ..backend_interface.metrics.events import UploadEvent
from ..debugging.log import LoggerRoot
from ..storage.helper import remote_driver_schemes
from ..storage.util import sha256sum, format_size, get_common_path
from ..utilities.process.mp import SafeEvent, ForkSafeRLock
from ..utilities.proxy_object import LazyEvalWrapper

try:
    import pandas as pd
    DataFrame = pd.DataFrame
except ImportError:
    pd = None
    DataFrame = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    from pathlib import Path as pathlib_Path
except ImportError:
    pathlib_Path = None


class Artifact(object):
    """
    Read-Only Artifact object
    """

    _not_set = object()

    @property
    def url(self):
        # type: () -> str
        """
        :return: The URL of uploaded artifact.
        """
        return self._url

    @property
    def name(self):
        # type: () -> str
        """
        :return: The name of artifact.
        """
        return self._name

    @property
    def size(self):
        # type: () -> int
        """
        :return: The size in bytes of artifact.
        """
        return self._size

    @property
    def type(self):
        # type: () -> str
        """
        :return: The type (str) of of artifact.
        """
        return self._type

    @property
    def mode(self):
        # type: () -> Union["input", "output"]  # noqa: F821
        """
        :return: The mode (str) of of artifact: "input" or "output".
        """
        return self._mode

    @property
    def hash(self):
        # type: () -> str
        """
        :return: SHA2 hash (str) of of artifact content.
        """
        return self._hash

    @property
    def timestamp(self):
        # type: () -> datetime
        """
        :return: Timestamp (datetime) of uploaded artifact.
        """
        return self._timestamp

    @property
    def metadata(self):
        # type: () -> Optional[Dict[str, str]]
        """
        :return: Key/Value dictionary attached to artifact.
        """
        return self._metadata

    @property
    def preview(self):
        # type: () -> str
        """
        :return: A string (str) representation of the artifact.
        """
        return self._preview

    def __init__(self, artifact_api_object):
        """
        construct read-only object from api artifact object

        :param tasks.Artifact artifact_api_object:
        """
        self._name = artifact_api_object.key
        self._size = artifact_api_object.content_size
        self._type = artifact_api_object.type
        self._mode = artifact_api_object.mode
        self._url = artifact_api_object.uri
        self._hash = artifact_api_object.hash
        self._timestamp = datetime.fromtimestamp(artifact_api_object.timestamp or 0)
        self._metadata = dict(artifact_api_object.display_data) if artifact_api_object.display_data else {}
        self._preview = artifact_api_object.type_data.preview if artifact_api_object.type_data else None
        self._content_type = artifact_api_object.type_data.content_type if artifact_api_object.type_data else None
        self._object = self._not_set

    def get(self, force_download=False, deserialization_function=None):
        # type: (bool, Optional[Callable[[bytes], Any]]) -> Any
        """
        Return an object constructed from the artifact file

        Currently supported types: Numpy.array, pandas.DataFrame, PIL.Image, dict.
        Supported content types are:
            - dict - ``.json``, ``.yaml``
            - pandas.DataFrame - ``.csv.gz``, ``.parquet``, ``.feather``, ``.pickle``
            - numpy.ndarray - ``.npz``, ``.csv.gz``
            - PIL.Image - whatever content types PIL supports
        All other types will return a pathlib2.Path object pointing to a local copy of the artifacts file (or directory).
        In case the content of a supported type could not be parsed, a pathlib2.Path object
        pointing to a local copy of the artifacts file (or directory) will be returned

        :param bool force_download: download file from remote even if exists in local cache
        :param Callable[bytes, Any] deserialization_function: A deserialization function that takes one parameter of type `bytes`,
            which represents the serialized object. This function should return the deserialized object.
            Useful when the artifact was uploaded using a custom serialization function when calling the
            `Task.upload_artifact` method with the `serialization_function` argument.
        :return: Usually, one of the following objects: Numpy.array, pandas.DataFrame, PIL.Image, dict (json), or pathlib2.Path.
            An object with an arbitrary type may also be returned if it was serialized (using pickle or a custom serialization function).
        """
        if self._object is not self._not_set:
            return self._object

        local_file = self.get_local_copy(raise_on_error=True, force_download=force_download)

        # noinspection PyBroadException
        try:
            if deserialization_function:
                with open(local_file, "rb") as f:
                    self._object = deserialization_function(f.read())
            elif self.type == "numpy" and np:
                if self._content_type == "text/csv":
                    self._object = np.genfromtxt(local_file, delimiter=",")
                else:
                    self._object = np.load(local_file)[self.name]
            elif self.type == Artifacts._pd_artifact_type or self.type == "pandas" and pd:
                if self._content_type == "application/parquet":
                    self._object = pd.read_parquet(local_file)
                elif self._content_type == "application/feather":
                    self._object = pd.read_feather(local_file)
                elif self._content_type == "application/pickle":
                    self._object = pd.read_pickle(local_file)
                elif self.type == Artifacts._pd_artifact_type:
                    self._object = pd.read_csv(local_file)
                else:
                    self._object = pd.read_csv(local_file, index_col=[0])
            elif self.type == "image":
                self._object = Image.open(local_file)
            elif self.type == "JSON" or self.type == "dict":
                with open(local_file, "rt") as f:
                    if self.type == "JSON" or self._content_type == "application/json":
                        self._object = json.load(f)
                    else:
                        self._object = yaml.safe_load(f)
            elif self.type == "string":
                with open(local_file, "rt") as f:
                    self._object = f.read()
            elif self.type == "pickle":
                with open(local_file, "rb") as f:
                    self._object = pickle.load(f)
        except Exception as e:
            LoggerRoot.get_base_logger().warning(
                "Exception '{}' encountered when getting artifact with type {} and content type {}".format(
                    e, self.type, self._content_type
                )
            )

        if self._object is self._not_set:
            local_file = Path(local_file)
            self._object = local_file

        return self._object

    def get_local_copy(self, extract_archive=True, raise_on_error=False, force_download=False):
        # type: (bool, bool, bool) -> str
        """
        :param bool extract_archive: If True and artifact is of type 'archive' (compressed folder)
            The returned path will be a temporary folder containing the archive content
        :param bool raise_on_error: If True and the artifact could not be downloaded,
            raise ValueError, otherwise return None on failure and output log warning.
        :param bool force_download: download file from remote even if exists in local cache
        :raise: Raises error if local copy not found.
        :return: A local path to a downloaded copy of the artifact.
        """
        from clearml.storage import StorageManager
        local_copy = StorageManager.get_local_copy(
            remote_url=self.url,
            extract_archive=extract_archive and self.type == 'archive',
            name=self.name,
            force_download=force_download
        )
        if raise_on_error and local_copy is None:
            raise ValueError(
                "Could not retrieve a local copy of artifact {}, failed downloading {}".format(self.name, self.url))

        return local_copy

    def __repr__(self):
        return str({'name': self.name, 'size': self.size, 'type': self.type, 'mode': self.mode, 'url': self.url,
                    'hash': self.hash, 'timestamp': self.timestamp,
                    'metadata': self.metadata, 'preview': self.preview, })


class Artifacts(object):
    max_preview_size_bytes = 65536

    _flush_frequency_sec = 300.
    # notice these two should match
    _save_format = '.csv.gz'
    _compression = 'gzip'
    # hashing constants
    _hash_block_size = 65536
    _pd_artifact_type = 'data-audit-table'

    class _ProxyDictWrite(dict):
        """ Dictionary wrapper that updates an arguments instance on any item set in the dictionary """

        def __init__(self, artifacts_manager, *args, **kwargs):
            super(Artifacts._ProxyDictWrite, self).__init__(*args, **kwargs)
            self._artifacts_manager = artifacts_manager
            # list of artifacts we should not upload (by name & weak-reference)
            self.artifact_metadata = {}
            # list of hash columns to calculate uniqueness for the artifacts
            self.artifact_hash_columns = {}

        def __setitem__(self, key, value):
            # check that value is of type pandas
            if pd and isinstance(value, pd.DataFrame):
                super(Artifacts._ProxyDictWrite, self).__setitem__(key, value)

                if self._artifacts_manager:
                    self._artifacts_manager.flush()
            else:
                raise ValueError('Artifacts currently support pandas.DataFrame objects only')

        def unregister_artifact(self, name):
            self.artifact_metadata.pop(name, None)
            self.pop(name, None)

        def add_metadata(self, name, metadata):
            self.artifact_metadata[name] = deepcopy(metadata)

        def get_metadata(self, name):
            return self.artifact_metadata.get(name)

        def add_hash_columns(self, artifact_name, hash_columns):
            self.artifact_hash_columns[artifact_name] = hash_columns

        def get_hash_columns(self, artifact_name):
            return self.artifact_hash_columns.get(artifact_name)

    @property
    def registered_artifacts(self):
        # type: () -> Dict[str, Artifact]
        return self._artifacts_container

    @property
    def summary(self):
        # type: () -> str
        return self._summary

    def __init__(self, task):
        self._task = task
        # notice the double link, this is important since the Artifact
        # dictionary needs to signal the Artifacts base on changes
        self._artifacts_container = self._ProxyDictWrite(self)
        self._last_artifacts_upload = {}
        self._unregister_request = set()
        self._thread = None
        self._flush_event = SafeEvent()
        self._exit_flag = False
        self._summary = ''
        self._temp_folder = []
        self._task_artifact_list = []
        self._task_edit_lock = ForkSafeRLock()
        self._storage_prefix = None
        self._task_name = None
        self._project_name = None

    def register_artifact(self, name, artifact, metadata=None, uniqueness_columns=True):
        # type: (str, DataFrame, Optional[dict], Union[bool, Sequence[str]]) -> ()
        """
        :param str name: name of the artifacts. Notice! it will override previous artifacts if name already exists.
        :param pandas.DataFrame artifact: artifact object, supported artifacts object types: pandas.DataFrame
        :param dict metadata: dictionary of key value to store with the artifact (visible in the UI)
        :param list uniqueness_columns: list of columns for artifact uniqueness comparison criteria. The default value
            is True, which equals to all the columns (same as artifact.columns).
        """
        # currently we support pandas.DataFrame (which we will upload as csv.gz)
        if name in self._artifacts_container:
            LoggerRoot.get_base_logger().info('Register artifact, overwriting existing artifact \"{}\"'.format(name))
        self._artifacts_container.add_hash_columns(
            name, list(artifact.columns if uniqueness_columns is True else uniqueness_columns)
        )
        self._artifacts_container[name] = artifact
        if metadata:
            self._artifacts_container.add_metadata(name, metadata)

    def unregister_artifact(self, name):
        # type: (str) -> ()
        # Remove artifact from the watch list
        self._unregister_request.add(name)
        self.flush()

    def upload_artifact(
        self,
        name,  # type: str
        artifact_object=None,  # type: Optional[object]
        metadata=None,  # type: Optional[dict]
        preview=None,  # type: Optional[str]
        delete_after_upload=False,  # type: bool
        auto_pickle=True,  # type: bool
        wait_on_upload=False,  # type: bool
        extension_name=None,  # type: Optional[str]
        serialization_function=None,  # type: Optional[Callable[[Any], Union[bytes, bytearray]]]
    ):
        # type: (...) -> bool
        if not Session.check_min_api_version("2.3"):
            LoggerRoot.get_base_logger().warning(
                "Artifacts not supported by your ClearML-server version,"
                " please upgrade to the latest server version"
            )
            return False

        if name in self._artifacts_container:
            raise ValueError("Artifact by the name of {} is already registered, use register_artifact".format(name))

        # cast preview to string
        if preview is not None and not (isinstance(preview, bool) and preview is False):
            preview = str(preview)

        # evaluate lazy proxy object
        if isinstance(artifact_object, LazyEvalWrapper):
            # noinspection PyProtectedMember
            artifact_object = LazyEvalWrapper._load_object(artifact_object)

        pathlib_types = (Path, pathlib_Path,) if pathlib_Path is not None else (Path,)
        local_filename = None

        # try to convert string Path object (it might reference a file/folder)
        # dont not try to serialize long texts.
        if isinstance(artifact_object, six.string_types) and artifact_object and len(artifact_object) < 2048:
            # noinspection PyBroadException
            try:
                artifact_path = Path(artifact_object)
                if artifact_path.exists():
                    artifact_object = artifact_path
                elif '*' in artifact_object or '?' in artifact_object:
                    # hackish, detect wildcard in tr files
                    folder = Path('').joinpath(*artifact_path.parts[:-1])
                    if folder.is_dir() and folder.parts:
                        wildcard = artifact_path.parts[-1]
                        if list(Path(folder).rglob(wildcard)):
                            artifact_object = artifact_path
            except Exception:
                pass

        store_as_pickle = False
        artifact_type_data = tasks.ArtifactTypeData()
        artifact_type_data.preview = ''
        override_filename_in_uri = None
        override_filename_ext_in_uri = None
        uri = None

        def get_extension(extension_name_, valid_extensions, default_extension, artifact_type_):
            if not extension_name_:
                return default_extension
            if extension_name_ in valid_extensions:
                return extension_name_
            LoggerRoot.get_base_logger().warning(
                "{} artifact can not be uploaded with extension {}. Valid extensions are: {}. Defaulting to {}.".format(
                    artifact_type_, extension_name_, ", ".join(valid_extensions), default_extension
                )
            )
            return default_extension

        if serialization_function:
            artifact_type = "custom"
            # noinspection PyBroadException
            try:
                artifact_type_data.preview = preview or str(artifact_object.__repr__())[:self.max_preview_size_bytes]
            except Exception:
                artifact_type_data.preview = ""
            override_filename_ext_in_uri = extension_name or ""
            override_filename_in_uri = name + override_filename_ext_in_uri
            fd, local_filename = mkstemp(prefix=quote(name, safe="") + ".", suffix=override_filename_ext_in_uri)
            os.close(fd)
            # noinspection PyBroadException
            try:
                with open(local_filename, "wb") as f:
                    f.write(serialization_function(artifact_object))
            except Exception:
                # cleanup and raise exception
                os.unlink(local_filename)
                raise
            artifact_type_data.content_type = mimetypes.guess_type(local_filename)[0]
        elif extension_name == ".pkl":
            store_as_pickle = True
        elif np and isinstance(artifact_object, np.ndarray):
            artifact_type = 'numpy'
            artifact_type_data.preview = preview or str(artifact_object.__repr__())
            override_filename_ext_in_uri = get_extension(
                extension_name, [".npz", ".csv.gz"], ".npz", artifact_type
            )
            override_filename_in_uri = name + override_filename_ext_in_uri
            fd, local_filename = mkstemp(prefix=quote(name, safe="") + '.', suffix=override_filename_ext_in_uri)
            os.close(fd)
            if override_filename_ext_in_uri == ".npz":
                artifact_type_data.content_type = "application/numpy"
                np.savez_compressed(local_filename, **{name: artifact_object})
            elif override_filename_ext_in_uri == ".csv.gz":
                artifact_type_data.content_type = "text/csv"
                np.savetxt(local_filename, artifact_object, delimiter=",")
            delete_after_upload = True
        elif pd and isinstance(artifact_object, pd.DataFrame):
            artifact_type = "pandas"
            artifact_type_data.preview = preview or str(artifact_object.__repr__())
            override_filename_ext_in_uri = get_extension(
                extension_name, [".csv.gz", ".parquet", ".feather", ".pickle"], ".csv.gz", artifact_type
            )
            override_filename_in_uri = name
            fd, local_filename = mkstemp(prefix=quote(name, safe="") + '.', suffix=override_filename_ext_in_uri)
            os.close(fd)
            if override_filename_ext_in_uri == ".csv.gz":
                artifact_type_data.content_type = "text/csv"
                artifact_object.to_csv(local_filename, compression=self._compression)
            elif override_filename_ext_in_uri == ".parquet":
                try:
                    artifact_type_data.content_type = "application/parquet"
                    artifact_object.to_parquet(local_filename)
                except Exception as e:
                    LoggerRoot.get_base_logger().warning(
                        "Exception '{}' encountered when uploading artifact as .parquet. Defaulting to .csv.gz".format(
                            e
                        )
                    )
                    artifact_type_data.content_type = "text/csv"
                    artifact_object.to_csv(local_filename, compression=self._compression)
            elif override_filename_ext_in_uri == ".feather":
                try:
                    artifact_type_data.content_type = "application/feather"
                    artifact_object.to_feather(local_filename)
                except Exception as e:
                    LoggerRoot.get_base_logger().warning(
                        "Exception '{}' encountered when uploading artifact as .feather. Defaulting to .csv.gz".format(
                            e
                        )
                    )
                    artifact_type_data.content_type = "text/csv"
                    artifact_object.to_csv(local_filename, compression=self._compression)
            elif override_filename_ext_in_uri == ".pickle":
                artifact_type_data.content_type = "application/pickle"
                artifact_object.to_pickle(local_filename)
            delete_after_upload = True
        elif isinstance(artifact_object, Image.Image):
            artifact_type = "image"
            artifact_type_data.content_type = "image/png"
            desc = str(artifact_object.__repr__())
            artifact_type_data.preview = preview or desc[1:desc.find(' at ')]

            # noinspection PyBroadException
            try:
                if not Image.EXTENSION:
                    Image.init()
                    if not Image.EXTENSION:
                        raise Exception()
                override_filename_ext_in_uri = get_extension(
                    extension_name, Image.EXTENSION.keys(), ".png", artifact_type
                )
            except Exception:
                override_filename_ext_in_uri = ".png"
                if extension_name and extension_name != ".png":
                    LoggerRoot.get_base_logger().warning(
                        "image artifact can not be uploaded with extension {}. Defaulting to .png.".format(
                            extension_name
                        )
                    )

            override_filename_in_uri = name + override_filename_ext_in_uri
            artifact_type_data.content_type = "image/unknown-type"
            guessed_type = mimetypes.guess_type(override_filename_in_uri)[0]
            if guessed_type:
                artifact_type_data.content_type = guessed_type

            fd, local_filename = mkstemp(prefix=quote(name, safe="") + '.', suffix=override_filename_ext_in_uri)
            os.close(fd)
            artifact_object.save(local_filename)
            delete_after_upload = True
        elif isinstance(artifact_object, dict):
            artifact_type = "dict"
            override_filename_ext_in_uri = get_extension(extension_name, [".json", ".yaml"], ".json", artifact_type)
            if override_filename_ext_in_uri == ".json":
                artifact_type_data.content_type = "application/json"
                # noinspection PyBroadException
                try:
                    serialized_text = json.dumps(artifact_object, sort_keys=True, indent=4)
                except Exception:
                    if not auto_pickle:
                        raise
                    LoggerRoot.get_base_logger().warning(
                        "JSON serialization of artifact \'{}\' failed, reverting to pickle".format(name))
                    store_as_pickle = True
                    serialized_text = None
            else:
                artifact_type_data.content_type = "application/yaml"
                # noinspection PyBroadException
                try:
                    serialized_text = yaml.dump(artifact_object, sort_keys=True, indent=4)
                except Exception:
                    if not auto_pickle:
                        raise
                    LoggerRoot.get_base_logger().warning(
                        "YAML serialization of artifact \'{}\' failed, reverting to pickle".format(name))
                    store_as_pickle = True
                    serialized_text = None

            if serialized_text is not None:
                override_filename_in_uri = name + override_filename_ext_in_uri
                fd, local_filename = mkstemp(prefix=quote(name, safe="") + ".", suffix=override_filename_ext_in_uri)
                with open(fd, "w") as f:
                    f.write(serialized_text)
                preview = preview or serialized_text
                if len(preview) < self.max_preview_size_bytes:
                    artifact_type_data.preview = preview
                else:
                    artifact_type_data.preview = (
                        "# full serialized dict too large to store, storing first {}kb\n{}".format(
                            self.max_preview_size_bytes // 1024, preview[: self.max_preview_size_bytes]
                        )
                    )
                delete_after_upload = True
        elif isinstance(artifact_object, pathlib_types):
            # check if single file
            artifact_object = Path(artifact_object)

            artifact_object.expanduser().absolute()
            # noinspection PyBroadException
            try:
                create_zip_file = not artifact_object.is_file()
            except Exception:  # Hack for windows pathlib2 bug, is_file isn't valid.
                create_zip_file = True
            else:  # We assume that this is not Windows os
                if artifact_object.is_dir():
                    # change to wildcard
                    artifact_object /= '*'

            if create_zip_file:
                folder = Path('').joinpath(*artifact_object.parts[:-1])
                if not folder.is_dir() or not folder.parts:
                    raise ValueError("Artifact file/folder '{}' could not be found".format(
                        artifact_object.as_posix()))

                wildcard = artifact_object.parts[-1]
                files = list(Path(folder).rglob(wildcard))
                override_filename_ext_in_uri = '.zip'
                override_filename_in_uri = folder.parts[-1] + override_filename_ext_in_uri
                fd, zip_file = mkstemp(
                    prefix=quote(folder.parts[-1], safe="") + '.', suffix=override_filename_ext_in_uri
                )
                try:
                    artifact_type_data.content_type = 'application/zip'
                    archive_preview = 'Archive content {}:\n'.format(artifact_object.as_posix())

                    with ZipFile(zip_file, 'w', allowZip64=True, compression=ZIP_DEFLATED) as zf:
                        for filename in sorted(files):
                            if filename.is_file():
                                relative_file_name = filename.relative_to(folder).as_posix()
                                archive_preview += '{} - {}\n'.format(
                                    relative_file_name, format_size(filename.stat().st_size))
                                zf.write(filename.as_posix(), arcname=relative_file_name)
                except Exception as e:
                    # failed uploading folder:
                    LoggerRoot.get_base_logger().warning('Exception {}\nFailed zipping artifact folder {}'.format(
                        folder, e))
                    return False
                finally:
                    os.close(fd)
                artifact_type_data.preview = preview or archive_preview
                artifact_object = zip_file
                artifact_type = 'archive'
                artifact_type_data.content_type = mimetypes.guess_type(artifact_object)[0]
                local_filename = artifact_object
                delete_after_upload = True
            else:
                if not artifact_object.is_file():
                    raise ValueError("Artifact file '{}' could not be found".format(artifact_object.as_posix()))

                override_filename_in_uri = artifact_object.parts[-1]
                artifact_type_data.preview = preview or '{} - {}\n'.format(
                    artifact_object.name, format_size(artifact_object.stat().st_size))
                artifact_object = artifact_object.as_posix()
                artifact_type = 'custom'
                artifact_type_data.content_type = mimetypes.guess_type(artifact_object)[0]
                local_filename = artifact_object
        elif (
            isinstance(artifact_object, (list, tuple))
            and artifact_object
            and all(isinstance(p, pathlib_types) for p in artifact_object)
        ):
            # find common path if exists
            list_files = [Path(p) for p in artifact_object]
            override_filename_ext_in_uri = '.zip'
            override_filename_in_uri = quote(name, safe="") + override_filename_ext_in_uri
            common_path = get_common_path(list_files)
            fd, zip_file = mkstemp(
                prefix='artifact_folder.', suffix=override_filename_ext_in_uri
            )
            try:
                artifact_type_data.content_type = 'application/zip'
                archive_preview = 'Archive content:\n'

                with ZipFile(zip_file, 'w', allowZip64=True, compression=ZIP_DEFLATED) as zf:
                    for filename in sorted(list_files):
                        if filename.is_file():
                            relative_file_name = filename.relative_to(Path(common_path)).as_posix() \
                                if common_path else filename.as_posix()
                            archive_preview += '{} - {}\n'.format(
                                relative_file_name, format_size(filename.stat().st_size))
                            zf.write(filename.as_posix(), arcname=relative_file_name)
                        else:
                            LoggerRoot.get_base_logger().warning(
                                "Failed zipping artifact file '{}', file not found!".format(filename.as_posix()))
            except Exception as e:
                # failed uploading folder:
                LoggerRoot.get_base_logger().warning('Exception {}\nFailed zipping artifact files {}'.format(
                    artifact_object, e))
                return False
            finally:
                os.close(fd)
            artifact_type_data.preview = preview or archive_preview
            artifact_object = zip_file
            artifact_type = 'archive'
            artifact_type_data.content_type = mimetypes.guess_type(artifact_object)[0]
            local_filename = artifact_object
            delete_after_upload = True
        elif (
                isinstance(artifact_object, six.string_types) and len(artifact_object) < 4096
                and urlparse(artifact_object).scheme in remote_driver_schemes
        ):
            # we should not upload this, just register
            local_filename = None
            uri = artifact_object
            artifact_type = 'custom'
            artifact_type_data.content_type = mimetypes.guess_type(artifact_object)[0]
            if preview:
                artifact_type_data.preview = preview
        elif isinstance(artifact_object, six.string_types) and artifact_object:
            # if we got here, we should store it as text file.
            artifact_type = 'string'
            artifact_type_data.content_type = 'text/plain'
            if preview:
                artifact_type_data.preview = preview
            elif len(artifact_object) < self.max_preview_size_bytes:
                artifact_type_data.preview = artifact_object
            else:
                artifact_type_data.preview = '# full text too large to store, storing first {}kb\n{}'.format(
                    self.max_preview_size_bytes//1024, artifact_object[:self.max_preview_size_bytes]
                )
            delete_after_upload = True
            override_filename_ext_in_uri = ".txt"
            override_filename_in_uri = name + override_filename_ext_in_uri
            fd, local_filename = mkstemp(prefix=quote(name, safe="") + ".", suffix=override_filename_ext_in_uri)
            os.close(fd)
            # noinspection PyBroadException
            try:
                with open(local_filename, "wt") as f:
                    f.write(artifact_object)
            except Exception:
                # cleanup and raise exception
                os.unlink(local_filename)
                raise
        elif artifact_object is None or (isinstance(artifact_object, str) and artifact_object == ""):
            artifact_type = ''
            store_as_pickle = True
        elif auto_pickle:
            # revert to pickling the object
            store_as_pickle = True
        else:
            raise ValueError("Artifact type {} not supported".format(type(artifact_object)))

        # revert to serializing the object with pickle
        if store_as_pickle:
            # if we are here it means we do not know what to do with the object, so we serialize it with pickle.
            artifact_type = 'pickle'
            artifact_type_data.content_type = 'application/pickle'
            # noinspection PyBroadException
            try:
                artifact_type_data.preview = preview or str(artifact_object.__repr__())[:self.max_preview_size_bytes]
            except Exception:
                artifact_type_data.preview = preview or ''
            delete_after_upload = True
            override_filename_ext_in_uri = '.pkl'
            override_filename_in_uri = name + override_filename_ext_in_uri
            fd, local_filename = mkstemp(prefix=quote(name, safe="") + '.', suffix=override_filename_ext_in_uri)
            os.close(fd)
            # noinspection PyBroadException
            try:
                with open(local_filename, 'wb') as f:
                    pickle.dump(artifact_object, f)
            except Exception:
                # cleanup and raise exception
                os.unlink(local_filename)
                raise

        # verify preview not out of scope:
        if artifact_type_data.preview and len(artifact_type_data.preview) > (self.max_preview_size_bytes+1024):
            artifact_type_data.preview = '# full preview too large to store, storing first {}kb\n{}'.format(
                self.max_preview_size_bytes // 1024, artifact_type_data.preview[:self.max_preview_size_bytes]
            )

        # remove from existing list, if exists
        for artifact in self._task_artifact_list:
            if artifact.key == name:
                if artifact.type == self._pd_artifact_type:
                    raise ValueError("Artifact of name {} already registered, "
                                     "use register_artifact instead".format(name))

                self._task_artifact_list.remove(artifact)
                break

        if not local_filename:
            file_size = None
            file_hash = None
        else:
            # check that the file to upload exists
            local_filename = Path(local_filename).absolute()
            if not local_filename.exists() or not local_filename.is_file():
                LoggerRoot.get_base_logger().warning('Artifact upload failed, cannot find file {}'.format(
                    local_filename.as_posix()))
                return False

            file_hash, _ = sha256sum(local_filename.as_posix(), block_size=Artifacts._hash_block_size)
            file_size = local_filename.stat().st_size

            uri = self._upload_local_file(local_filename, name,
                                          delete_after_upload=delete_after_upload,
                                          override_filename=override_filename_in_uri,
                                          override_filename_ext=override_filename_ext_in_uri,
                                          wait_on_upload=wait_on_upload)

        timestamp = int(time())

        artifact = tasks.Artifact(key=name, type=artifact_type,
                                  uri=uri,
                                  content_size=file_size,
                                  hash=file_hash,
                                  timestamp=timestamp,
                                  type_data=artifact_type_data,
                                  display_data=[(str(k), str(v)) for k, v in metadata.items()] if metadata else None)

        # update task artifacts
        self._add_artifact(artifact)

        return True

    def flush(self):
        # type: () -> ()
        # start the thread if it hasn't already:
        self._start()
        # flush the current state of all artifacts
        self._flush_event.set()

    def stop(self, wait=True):
        # type: (bool) -> ()
        # stop the daemon thread and quit
        # wait until thread exists
        self._exit_flag = True
        self._flush_event.set()
        if wait:
            if self._thread:
                self._thread.join()
            # remove all temp folders
            for f in self._temp_folder:
                # noinspection PyBroadException
                try:
                    Path(f).rmdir()
                except Exception:
                    pass

    def _start(self):
        # type: () -> ()
        """ Start daemon thread if any artifacts are registered and thread is not up yet """
        if not self._thread and self._artifacts_container:
            # start the daemon thread
            self._flush_event.clear()
            self._thread = Thread(target=self._daemon)
            self._thread.daemon = True
            self._thread.start()

    def _daemon(self):
        # type: () -> ()
        while not self._exit_flag:
            self._flush_event.wait(self._flush_frequency_sec)
            self._flush_event.clear()
            artifact_keys = list(self._artifacts_container.keys())
            for name in artifact_keys:
                try:
                    self._upload_data_audit_artifacts(name)
                except Exception as e:
                    LoggerRoot.get_base_logger().warning(str(e))

        # create summary
        self._summary = self._get_statistics()

    def _add_artifact(self, artifact):
        if not self._task:
            raise ValueError("Task object not set")
        with self._task_edit_lock:
            if artifact not in self._task_artifact_list:
                self._task_artifact_list.append(artifact)
            # noinspection PyProtectedMember
            self._task._add_artifacts(self._task_artifact_list)

    def _upload_data_audit_artifacts(self, name):
        # type: (str) -> ()
        logger = self._task.get_logger()
        pd_artifact = self._artifacts_container.get(name)
        pd_metadata = self._artifacts_container.get_metadata(name)

        # remove from artifacts watch list
        if name in self._unregister_request:
            try:
                self._unregister_request.remove(name)
            except KeyError:
                pass
            self._artifacts_container.unregister_artifact(name)

        if pd_artifact is None:
            return

        override_filename_ext_in_uri = self._save_format
        override_filename_in_uri = name
        fd, local_csv = mkstemp(prefix=quote(name, safe="") + '.', suffix=override_filename_ext_in_uri)
        os.close(fd)
        local_csv = Path(local_csv)
        pd_artifact.to_csv(local_csv.as_posix(), index=False, compression=self._compression)
        current_sha2, file_sha2 = sha256sum(
            local_csv.as_posix(), skip_header=32, block_size=Artifacts._hash_block_size)
        if name in self._last_artifacts_upload:
            previous_sha2 = self._last_artifacts_upload[name]
            if previous_sha2 == current_sha2:
                # nothing to do, we can skip the upload
                # noinspection PyBroadException
                try:
                    local_csv.unlink()
                except Exception:
                    pass
                return
        self._last_artifacts_upload[name] = current_sha2

        # If old clearml-server, upload as debug image
        if not Session.check_min_api_version('2.3'):
            logger.report_image(title='artifacts', series=name, local_path=local_csv.as_posix(),
                                delete_after_upload=True, iteration=self._task.get_last_iteration(),
                                max_image_history=2)
            return

        # Find our artifact
        artifact = None
        for an_artifact in self._task_artifact_list:
            if an_artifact.key == name:
                artifact = an_artifact
                break

        file_size = local_csv.stat().st_size

        # upload file
        uri = self._upload_local_file(local_csv, name, delete_after_upload=True,
                                      override_filename=override_filename_in_uri,
                                      override_filename_ext=override_filename_ext_in_uri)

        # update task artifacts
        with self._task_edit_lock:
            if not artifact:
                artifact = tasks.Artifact(key=name, type=self._pd_artifact_type)
            artifact_type_data = tasks.ArtifactTypeData()

            artifact_type_data.data_hash = current_sha2
            artifact_type_data.content_type = "text/csv"
            artifact_type_data.preview = str(pd_artifact.__repr__(
            )) + '\n\n' + self._get_statistics({name: pd_artifact})

            artifact.type_data = artifact_type_data
            artifact.uri = uri
            artifact.content_size = file_size
            artifact.hash = file_sha2
            artifact.timestamp = int(time())
            artifact.display_data = [(str(k), str(v)) for k, v in pd_metadata.items()] if pd_metadata else None

            self._add_artifact(artifact)

    def _upload_local_file(
            self, local_file, name, delete_after_upload=False, override_filename=None, override_filename_ext=None,
            wait_on_upload=False
    ):
        # type: (str, str, bool, Optional[str], Optional[str], bool) -> str
        """
        Upload local file and return uri of the uploaded file (uploading in the background)
        """
        from clearml.storage import StorageManager

        upload_uri = self._task.output_uri or self._task.get_logger().get_default_upload_destination()
        if not isinstance(local_file, Path):
            local_file = Path(local_file)
        ev = UploadEvent(metric='artifacts', variant=name,
                         image_data=None, upload_uri=upload_uri,
                         local_image_path=local_file.as_posix(),
                         delete_after_upload=delete_after_upload,
                         override_filename=override_filename,
                         override_filename_ext=override_filename_ext,
                         override_storage_key_prefix=self._get_storage_uri_prefix())
        _, uri = ev.get_target_full_upload_uri(upload_uri, quote_uri=False)

        # send for upload
        # noinspection PyProtectedMember
        if wait_on_upload:
            StorageManager.upload_file(local_file.as_posix(), uri, wait_for_upload=True, retries=ev.retries)
            if delete_after_upload:
                try:
                    os.unlink(local_file.as_posix())
                except OSError:
                    LoggerRoot.get_base_logger().warning('Failed removing temporary {}'.format(local_file))
        else:
            self._task._reporter._report(ev)

        _, quoted_uri = ev.get_target_full_upload_uri(upload_uri)

        return quoted_uri

    def _get_statistics(self, artifacts_dict=None):
        # type: (Optional[Dict[str, Artifact]]) -> str
        summary = ''
        artifacts_dict = artifacts_dict or self._artifacts_container
        thread_pool = ThreadPool()

        try:
            # build hash row sets
            artifacts_summary = []
            for a_name, a_df in artifacts_dict.items():
                hash_cols = self._artifacts_container.get_hash_columns(a_name)
                if not pd or not isinstance(a_df, pd.DataFrame):
                    continue

                if hash_cols is True:
                    hash_col_drop = []
                else:
                    hash_cols = set(hash_cols)
                    missing_cols = hash_cols.difference(a_df.columns)
                    if missing_cols == hash_cols:
                        LoggerRoot.get_base_logger().warning(
                            'Uniqueness columns {} not found in artifact {}. '
                            'Skipping uniqueness check for artifact.'.format(list(missing_cols), a_name)
                        )
                        continue
                    elif missing_cols:
                        # missing_cols must be a subset of hash_cols
                        hash_cols.difference_update(missing_cols)
                        LoggerRoot.get_base_logger().warning(
                            'Uniqueness columns {} not found in artifact {}. Using {}.'.format(
                                list(missing_cols), a_name, list(hash_cols)
                            )
                        )

                    hash_col_drop = [col for col in a_df.columns if col not in hash_cols]

                a_unique_hash = set()

                def hash_row(r):
                    a_unique_hash.add(hash(bytes(r)))

                a_shape = a_df.shape
                # parallelize
                a_hash_cols = a_df.drop(columns=hash_col_drop)
                thread_pool.map(hash_row, a_hash_cols.values)
                # add result
                artifacts_summary.append((a_name, a_shape, a_unique_hash,))

            # build intersection summary
            for i, (name, shape, unique_hash) in enumerate(artifacts_summary):
                summary += '[{name}]: shape={shape}, {unique} unique rows, {percentage:.1f}% uniqueness\n'.format(
                    name=name, shape=shape, unique=len(unique_hash),
                    percentage=100 * len(unique_hash) / float(shape[0]))
                for name2, shape2, unique_hash2 in artifacts_summary[i + 1:]:
                    intersection = len(unique_hash & unique_hash2)
                    summary += '\tIntersection with [{name2}] {intersection} rows: {percentage:.1f}%\n'.format(
                        name2=name2, intersection=intersection,
                        percentage=100 * intersection / float(len(unique_hash2)))
        except Exception as e:
            LoggerRoot.get_base_logger().warning(str(e))
        finally:
            thread_pool.close()
            thread_pool.terminate()
        return summary

    def _get_temp_folder(self, force_new=False):
        # type: (bool) -> str
        if force_new or not self._temp_folder:
            new_temp = mkdtemp(prefix='artifacts_')
            self._temp_folder.append(new_temp)
            return new_temp
        return self._temp_folder[0]

    def _get_storage_uri_prefix(self):
        # type: () -> str
        if not self._storage_prefix or self._task_name != self._task.name or self._project_name != self._task.get_project_name():
            # noinspection PyProtectedMember
            self._storage_prefix = self._task._get_output_destination_suffix()
            self._task_name = self._task.name
            self._project_name = self._task.get_project_name()
        return self._storage_prefix
