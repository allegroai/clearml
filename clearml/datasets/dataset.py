import json
import os
import shutil
from copy import deepcopy, copy
from fnmatch import fnmatch
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from tempfile import mkstemp, mkdtemp
from typing import Union, Optional, Sequence, List, Dict, Any, Mapping
from zipfile import ZipFile, ZIP_DEFLATED

from attr import attrs, attrib
from pathlib2 import Path

from .. import Task, StorageManager, Logger
from ..backend_api.session.client import APIClient
from ..backend_interface.task.development.worker import DevWorker
from ..backend_interface.util import mutually_exclusive, exact_match_regex, get_existing_project
from ..config import deferred_config
from ..debugging.log import LoggerRoot
from ..storage.helper import StorageHelper
from ..storage.cache import CacheManager
from ..storage.util import sha256sum, is_windows, md5text, format_size

try:
    from pathlib import Path as _Path  # noqa
except ImportError:
    _Path = None


@attrs
class FileEntry(object):
    relative_path = attrib(default=None, type=str)
    hash = attrib(default=None, type=str)
    parent_dataset_id = attrib(default=None, type=str)
    size = attrib(default=None, type=int)
    # support multi part artifact storage
    artifact_name = attrib(default=None, type=str)
    # cleared when file is uploaded.
    local_path = attrib(default=None, type=str)

    def as_dict(self):
        # type: () -> Dict
        state = dict(relative_path=self.relative_path, hash=self.hash,
                     parent_dataset_id=self.parent_dataset_id, size=self.size,
                     artifact_name=self.artifact_name,
                     **dict([('local_path', self.local_path)] if self.local_path else ()))
        return state


class Dataset(object):
    __private_magic = 42 * 1337
    __state_entry_name = 'state'
    __default_data_entry_name = 'data'
    __data_entry_name_prefix = 'data_'
    __cache_context = 'datasets'
    __tag = 'dataset'
    __cache_folder_prefix = 'ds_'
    __dataset_folder_template = CacheManager.set_context_folder_lookup(__cache_context, "{0}_archive_{1}")
    __preview_max_file_entries = 15000
    __preview_max_size = 5 * 1024 * 1024
    _dataset_chunk_size_mb = deferred_config("storage.dataset_chunk_size_mb", 512, transform=int)

    def __init__(self, _private, task=None, dataset_project=None, dataset_name=None, dataset_tags=None):
        # type: (int, Optional[Task], Optional[str], Optional[str], Optional[Sequence[str]]) -> ()
        """
        Do not use directly! Use Dataset.create(...) or Dataset.get(...) instead.
        """
        assert _private == self.__private_magic
        # key for the dataset file entries are the relative path within the data
        self._dataset_file_entries = {}  # type: Dict[str, FileEntry]
        # this will create a graph of all the dependencies we have, each entry lists it's own direct parents
        self._dependency_graph = {}  # type: Dict[str, List[str]]
        if task:
            self._task_pinger = None
            self._created_task = False
            task_status = task.data.status
            # if we are continuing aborted Task, force the state
            if str(task_status) == 'stopped':
                # print warning that we are opening a stopped dataset:
                LoggerRoot.get_base_logger().warning(
                    'Reopening aborted Dataset, any change will clear and overwrite current state')
                task.mark_started(force=True)
                task_status = 'in_progress'

            # If we are reusing the main current Task, make sure we set its type to data_processing
            if str(task_status) in ('created', 'in_progress'):
                if str(task.task_type) != str(Task.TaskTypes.data_processing):
                    task.set_task_type(task_type=Task.TaskTypes.data_processing)
                task_system_tags = task.get_system_tags() or []
                if self.__tag not in task_system_tags:
                    task.set_system_tags(task_system_tags + [self.__tag])
                if dataset_tags:
                    task.set_tags((task.get_tags() or []) + list(dataset_tags))
        else:
            self._created_task = True
            task = Task.create(
                project_name=dataset_project, task_name=dataset_name, task_type=Task.TaskTypes.data_processing)
            # set default output_uri
            task.output_uri = True
            task.set_system_tags((task.get_system_tags() or []) + [self.__tag])
            if dataset_tags:
                task.set_tags((task.get_tags() or []) + list(dataset_tags))
            task.mark_started()
            # generate the script section
            script = \
                'from clearml import Dataset\n\n' \
                'ds = Dataset.create(dataset_project=\'{dataset_project}\', dataset_name=\'{dataset_name}\')\n'.format(
                    dataset_project=dataset_project, dataset_name=dataset_name)
            task.data.script.diff = script
            task.data.script.working_dir = '.'
            task.data.script.entry_point = 'register_dataset.py'
            from clearml import __version__
            task.data.script.requirements = {'pip': 'clearml == {}\n'.format(__version__)}
            # noinspection PyProtectedMember
            task._edit(script=task.data.script)

            # if the task is running make sure we ping to the server so it will not be aborted by a watchdog
            self._task_pinger = DevWorker()
            self._task_pinger.register(task, stop_signal_support=False)
            # set the newly created Dataset parent ot the current Task, so we know who created it.
            if Task.current_task() and Task.current_task().id != task.id:
                task.set_parent(Task.current_task())

        # store current dataset Task
        self._task = task
        # store current dataset id
        self._id = task.id
        # store the folder where the dataset was downloaded to
        self._local_base_folder = None  # type: Optional[Path]
        # dirty flag, set True by any function call changing the dataset (regardless of weather it did anything)
        self._dirty = False
        self._using_current_task = False
        # set current artifact name to be used (support for multiple upload sessions)
        self._data_artifact_name = self._get_next_data_artifact_name()
        # store a cached lookup of the number of chunks each parent dataset has.
        # this will help with verifying we have n up-to-date partial local copy
        self._dependency_chunk_lookup = None  # type: Optional[Dict[str, int]]

    @property
    def id(self):
        # type: () -> str
        return self._id

    @property
    def file_entries(self):
        # type: () -> List[FileEntry]
        return list(self._dataset_file_entries.values())

    @property
    def file_entries_dict(self):
        # type: () -> Mapping[str, FileEntry]
        """
        Notice this call returns an internal representation, do not modify!
        :return: dict with relative file path as key, and FileEntry as value
        """
        return self._dataset_file_entries

    @property
    def project(self):
        # type: () -> str
        return self._task.get_project_name()

    @property
    def name(self):
        # type: () -> str
        return self._task.name

    @property
    def tags(self):
        # type: () -> List[str]
        return self._task.get_tags() or []

    @tags.setter
    def tags(self, values):
        # type: (List[str]) -> ()
        self._task.set_tags(values or [])

    def add_files(
            self,
            path,  # type: Union[str, Path, _Path]
            wildcard=None,  # type: Optional[Union[str, Sequence[str]]]
            local_base_folder=None,  # type: Optional[str]
            dataset_path=None,  # type: Optional[str]
            recursive=True,  # type: bool
            verbose=False  # type: bool
    ):
        # type: (...) -> ()
        """
        Add a folder into the current dataset. calculate file hash,
        and compare against parent, mark files to be uploaded

        :param path: Add a folder/file to the dataset
        :param wildcard: add only specific set of files.
            Wildcard matching, can be a single string or a list of wildcards)
        :param local_base_folder: files will be located based on their relative path from local_base_folder
        :param dataset_path: where in the dataset the folder/files should be located
        :param recursive: If True match all wildcard files recursively
        :param verbose: If True print to console files added/modified
        :return: number of files added
        """
        self._dirty = True
        self._task.get_logger().report_text(
            'Adding files to dataset: {}'.format(
                dict(path=path, wildcard=wildcard, local_base_folder=local_base_folder,
                     dataset_path=dataset_path, recursive=recursive, verbose=verbose)),
            print_console=False)

        num_added = self._add_files(
            path=path, wildcard=wildcard, local_base_folder=local_base_folder,
            dataset_path=dataset_path, recursive=recursive, verbose=verbose)

        # update the task script
        self._add_script_call(
            'add_files', path=path, wildcard=wildcard, local_base_folder=local_base_folder,
            dataset_path=dataset_path, recursive=recursive)

        self._serialize()

        return num_added

    def remove_files(self, dataset_path=None, recursive=True, verbose=False):
        # type: (Optional[str], bool, bool) -> int
        """
        Remove files from the current dataset

        :param dataset_path: Remove files from the dataset.
            The path is always relative to the dataset (e.g 'folder/file.bin')
        :param recursive: If True match all wildcard files recursively
        :param verbose: If True print to console files removed
        :return: Number of files removed
        """
        self._task.get_logger().report_text(
            'Removing files from dataset: {}'.format(
                dict(dataset_path=dataset_path, recursive=recursive, verbose=verbose)),
            print_console=False)

        if dataset_path and dataset_path.startswith('/'):
            dataset_path = dataset_path[1:]

        num_files = len(self._dataset_file_entries)
        org_files = list(self._dataset_file_entries.keys()) if verbose else None

        if not recursive:
            self._dataset_file_entries = {
                k: v for k, v in self._dataset_file_entries.items()
                if not fnmatch(k + '/', dataset_path + '/')}
        else:
            wildcard = dataset_path.split('/')[-1]
            path = dataset_path[:-len(dataset_path)] + '*'

            self._dataset_file_entries = {
                k: v for k, v in self._dataset_file_entries.items()
                if not (fnmatch(k, path) and fnmatch(k if '/' in k else '/{}'.format(k), '*/' + wildcard))}

        if verbose and org_files:
            for f in org_files:
                if f not in self._dataset_file_entries:
                    self._task.get_logger().report_text('Remove {}'.format(f))

        # update the task script
        self._add_script_call(
            'remove_files', dataset_path=dataset_path, recursive=recursive)

        self._serialize()

        return num_files - len(self._dataset_file_entries)

    def sync_folder(self, local_path, dataset_path=None, verbose=False):
        # type: (Union[Path, _Path, str], Union[Path, _Path, str], bool) -> (int, int)
        """
        Synchronize the dataset with a local folder. The dataset is synchronized from the
            relative_base_folder (default: dataset root)  and deeper with the specified local path.

        :param local_path: Local folder to sync (assumes all files and recursive)
        :param dataset_path: Target dataset path to sync with (default the root of the dataset)
        :param verbose: If true print to console files added/modified/removed
        :return: number of files removed, number of files modified/added
        """
        def filter_f(f):
            keep = (not f.relative_path.startswith(relative_prefix) or
                    (local_path / f.relative_path[len(relative_prefix):]).is_file())
            if not keep and verbose:
                self._task.get_logger().report_text('Remove {}'.format(f.relative_path))
            return keep

        self._task.get_logger().report_text(
            'Syncing local copy with dataset: {}'.format(
                dict(local_path=local_path, dataset_path=dataset_path, verbose=verbose)),
            print_console=False)

        self._dirty = True
        local_path = Path(local_path)

        # Path().as_posix() will never end with /
        relative_prefix = (Path(dataset_path).as_posix() + '/') if dataset_path else ''

        # remove files
        num_files = len(self._dataset_file_entries)
        self._dataset_file_entries = {
            k: f for k, f in self._dataset_file_entries.items() if filter_f(f)}
        removed_files = num_files - len(self._dataset_file_entries)

        # add remaining files
        added_files = self._add_files(path=local_path, dataset_path=dataset_path, recursive=True, verbose=verbose)

        if verbose:
            self._task.get_logger().report_text(
                'Syncing folder {} : {} files removed, {} added / modified'.format(
                    local_path.as_posix(), removed_files, added_files))

        # update the task script
        self._add_script_call(
            'sync_folder', local_path=local_path, dataset_path=dataset_path)

        self._serialize()
        return removed_files, added_files

    def upload(self, show_progress=True, verbose=False, output_url=None, compression=None, chunk_size=None):
        # type: (bool, bool, Optional[str], Optional[str], int) -> ()
        """
        Start file uploading, the function returns when all files are uploaded.

        :param show_progress: If True show upload progress bar
        :param verbose: If True print verbose progress report
        :param output_url: Target storage for the compressed dataset (default: file server)
            Examples: `s3://bucket/data`, `gs://bucket/data` , `azure://bucket/data` , `/mnt/share/data`
        :param compression: Compression algorithm for the Zipped dataset file (default: ZIP_DEFLATED)
        :param chunk_size: Artifact chunk size (MB) for the compressed dataset,
            if not provided (None) use the default chunk size (512mb).
            If -1 is provided, use a single zip artifact for the entire dataset change-set (old behaviour)
        """
        # set output_url
        if output_url:
            self._task.output_uri = output_url

        self._task.get_logger().report_text(
            'Uploading dataset files: {}'.format(
                dict(show_progress=show_progress, verbose=verbose, output_url=output_url, compression=compression)),
            print_console=False)

        list_zipped_artifacts = []  # List[Tuple[Path, int, str, str]]
        list_file_entries = list(self._dataset_file_entries.values())
        total_size = 0
        chunk_size = int(self._dataset_chunk_size_mb if not chunk_size else chunk_size)
        try:
            from tqdm import tqdm  # noqa
            a_tqdm = tqdm(total=len(list_file_entries))
        except ImportError:
            a_tqdm = None

        while list_file_entries:
            fd, zip_file = mkstemp(
                prefix='dataset.{}.'.format(self._id), suffix='.zip'
            )
            archive_preview = ''
            count = 0
            processed = 0
            zip_file = Path(zip_file)
            print('{}Compressing local files, chunk {} [remaining {} files]'.format(
                '\n' if a_tqdm else '', 1+len(list_zipped_artifacts), len(list_file_entries)))
            try:
                with ZipFile(zip_file.as_posix(), 'w', allowZip64=True, compression=compression or ZIP_DEFLATED) as zf:
                    for file_entry in list_file_entries:
                        processed += 1
                        if a_tqdm:
                            a_tqdm.update()
                        if not file_entry.local_path:
                            # file is already in an uploaded artifact
                            continue
                        filename = Path(file_entry.local_path)
                        if not filename.is_file():
                            LoggerRoot.get_base_logger().warning(
                                "Could not store dataset file {}. File skipped".format(file_entry.local_path))
                            # mark for removal
                            file_entry.relative_path = None
                            continue
                        if verbose:
                            self._task.get_logger().report_text('Compressing {}'.format(filename.as_posix()))

                        relative_file_name = file_entry.relative_path
                        zf.write(filename.as_posix(), arcname=relative_file_name)
                        archive_preview += '{} - {}\n'.format(
                            relative_file_name, format_size(filename.stat().st_size))
                        file_entry.artifact_name = self._data_artifact_name
                        count += 1

                        # limit the size of a single artifact
                        if chunk_size > 0 and zip_file.stat().st_size >= chunk_size * (1024**2):
                            break
            except Exception as e:
                # failed uploading folder:
                LoggerRoot.get_base_logger().warning(
                    'Exception {}\nFailed zipping dataset.'.format(e))
                return False
            finally:
                os.close(fd)

            if not count:
                zip_file.unlink()

            total_size += zip_file.stat().st_size
            # let's see what's left
            list_file_entries = list_file_entries[processed:]
            # update the artifact preview
            archive_preview = 'Dataset archive content [{} files]:\n'.format(count) + archive_preview
            # add into the list
            list_zipped_artifacts += [(zip_file, count, archive_preview, self._data_artifact_name)]
            # next artifact name to use
            self._data_artifact_name = self._get_next_data_artifact_name(self._data_artifact_name)

        if a_tqdm:
            a_tqdm.close()

        self._task.get_logger().report_text(
            'File compression completed: total size {}, {} chunked stored (average size {})'.format(
                format_size(total_size),
                len(list_zipped_artifacts),
                format_size(total_size / len(list_zipped_artifacts))))

        if not list_zipped_artifacts:
            LoggerRoot.get_base_logger().info('No pending files, skipping upload.')
            self._dirty = False
            self._serialize()
            return True

        for i, (zip_file, count, archive_preview, artifact_name) in enumerate(list_zipped_artifacts):
            # noinspection PyBroadException
            try:
                # let's try to rename it
                new_zip_file = zip_file.parent / 'dataset.{}.zip'.format(self._id)
                zip_file.rename(new_zip_file)
                zip_file = new_zip_file
            except Exception:
                pass

            # start upload
            zip_file_size = format_size(Path(zip_file).stat().st_size)
            self._task.get_logger().report_text(
                'Uploading compressed dataset changes {}/{} ({} files {}) to {}'.format(
                    i+1, len(list_zipped_artifacts), count, zip_file_size, self.get_default_storage()))

            self._task.upload_artifact(
                name=artifact_name, artifact_object=Path(zip_file), preview=archive_preview,
                delete_after_upload=True, wait_on_upload=True)

            # mark as upload completed and serialize
            for file_entry in self._dataset_file_entries.values():
                if file_entry.parent_dataset_id == self._id and file_entry.artifact_name == artifact_name:
                    file_entry.local_path = None
            # serialize current state
            self._serialize()

        # remove files that could not be zipped, containing Null relative Path
        self._dataset_file_entries = {
            k: v for k, v in self._dataset_file_entries.items() if v.relative_path is not None}
        # report upload completed
        self._task.get_logger().report_text('Upload completed ({})'.format(format_size(total_size)))

        self._add_script_call(
            'upload', show_progress=show_progress, verbose=verbose, output_url=output_url, compression=compression)

        self._dirty = False
        self._serialize()

    def finalize(self, verbose=False, raise_on_error=True, auto_upload=False):
        # type: (bool, bool, bool) -> bool
        """
        Finalize the dataset publish dataset Task. upload must first called to verify there are not pending uploads.
        If files do need to be uploaded, it throws an exception (or return False)

        :param verbose: If True print verbose progress report
        :param raise_on_error: If True raise exception if dataset finalizing failed
        :param auto_upload: Automatically upload dataset if not called yet, will upload to default location.
        """
        # check we do not have files waiting for upload.
        if self._dirty:
            if auto_upload:
                self._task.get_logger().report_text("Pending uploads, starting dataset upload to {}"
                                                    .format(self.get_default_storage()))
                self.upload()
            elif raise_on_error:
                raise ValueError("Cannot finalize dataset, pending uploads. Call Dataset.upload(...)")
            else:
                return False

        status = self._task.get_status()
        if status not in ('in_progress', 'created'):
            raise ValueError("Cannot finalize dataset, status '{}' is not valid".format(status))

        self._task.get_logger().report_text('Finalizing dataset', print_console=False)

        # make sure we have no redundant parent versions
        self._serialize(update_dependency_chunk_lookup=True)
        self._add_script_call('finalize')
        if verbose:
            print('Updating statistics and genealogy')
        self._report_dataset_genealogy()
        hashed_nodes = [self._get_dataset_id_hash(k) for k in self._dependency_graph.keys()]
        self._task.comment = 'Dependencies: {}\n'.format(hashed_nodes)
        if self._using_current_task:
            self._task.flush(wait_for_uploads=True)
        else:
            self._task.close()
            self._task.mark_completed()

        if self._task_pinger:
            self._task_pinger.unregister()
            self._task_pinger = None

        return True

    def publish(self, raise_on_error=True):
        # type: (bool) -> bool
        """
        Publish the dataset
        If dataset is not finalize, throw exception

        :param raise_on_error: If True raise exception if dataset publishing failed
        """
        # check we can publish this dataset
        if not self.is_final():
            raise ValueError("Cannot publish dataset, dataset in status {}.".format(self._task.get_status()))

        self._task.publish(ignore_errors=raise_on_error)
        return True

    def is_final(self):
        # type: () -> bool
        """
        Return True if the dataset was finalized and cannot be changed any more.

        :return: True if dataset if final
        """
        return self._task.get_status() not in (
            Task.TaskStatusEnum.in_progress, Task.TaskStatusEnum.created, Task.TaskStatusEnum.failed)

    def get_local_copy(self, use_soft_links=None, part=None, num_parts=None, raise_on_error=True):
        # type: (bool, Optional[int], Optional[int], bool) -> str
        """
        return a base folder with a read-only (immutable) local copy of the entire dataset
            download and copy / soft-link, files from all the parent dataset versions

        :param use_soft_links: If True use soft links, default False on windows True on Posix systems
        :param part: Optional, if provided only download the selected part (index) of the Dataset.
            First part number is `0` and last part is `num_parts-1`
            Notice, if `num_parts` is not provided, number of parts will be equal to the total number of chunks
            (i.e. sum over all chunks from the specified Dataset including all parent Datasets).
            This argument is passed to parent datasets, as well as the implicit `num_parts`,
            allowing users to get a partial copy of the entire dataset, for multi node/step processing.
        :param num_parts: Optional, If specified normalize the number of chunks stored to the
            requested number of parts. Notice that the actual chunks used per part are rounded down.
            Example: Assuming total 8 chunks for this dataset (including parent datasets),
            and `num_parts=5`, the chunk index used per parts would be:
            part=0 -> chunks[0,5], part=1 -> chunks[1,6], part=2 -> chunks[2,7], part=3 -> chunks[3, ]
        :param raise_on_error: If True raise exception if dataset merging failed on any file

        :return: A base folder for the entire dataset
        """
        assert self._id
        if not self._task:
            self._task = Task.get_task(task_id=self._id)

        # now let's merge the parents
        target_folder = self._merge_datasets(
            use_soft_links=use_soft_links, raise_on_error=raise_on_error, part=part, num_parts=num_parts)
        return target_folder

    def get_mutable_local_copy(self, target_folder, overwrite=False, part=None, num_parts=None, raise_on_error=True):
        # type: (Union[Path, _Path, str], bool, Optional[int], Optional[int], bool) -> Optional[str]
        """
        return a base folder with a writable (mutable) local copy of the entire dataset
            download and copy / soft-link, files from all the parent dataset versions

        :param target_folder: Target folder for the writable copy
        :param overwrite: If True, recursively delete the target folder before creating a copy.
            If False (default) and target folder contains files, raise exception or return None
        :param part: Optional, if provided only download the selected part (index) of the Dataset.
            First part number is `0` and last part is `num_parts-1`
            Notice, if `num_parts` is not provided, number of parts will be equal to the total number of chunks
            (i.e. sum over all chunks from the specified Dataset including all parent Datasets).
            This argument is passed to parent datasets, as well as the implicit `num_parts`,
            allowing users to get a partial copy of the entire dataset, for multi node/step processing.
        :param num_parts: Optional, If specified normalize the number of chunks stored to the
            requested number of parts. Notice that the actual chunks used per part are rounded down.
            Example: Assuming total 8 chunks for this dataset (including parent datasets),
            and `num_parts=5`, the chunk index used per parts would be:
            part=0 -> chunks[0,5], part=1 -> chunks[1,6], part=2 -> chunks[2,7], part=3 -> chunks[3, ]
        :param raise_on_error: If True raise exception if dataset merging failed on any file

        :return: A the target folder containing the entire dataset
        """
        assert self._id
        target_folder = Path(target_folder).absolute()
        target_folder.mkdir(parents=True, exist_ok=True)
        # noinspection PyBroadException
        try:
            target_folder.rmdir()
        except Exception:
            if not overwrite:
                if raise_on_error:
                    raise ValueError("Target folder {} already contains files".format(target_folder.as_posix()))
                else:
                    return None
            shutil.rmtree(target_folder.as_posix())

        ro_folder = self.get_local_copy(part=part, num_parts=num_parts, raise_on_error=raise_on_error)
        shutil.copytree(ro_folder, target_folder.as_posix(), symlinks=False)
        return target_folder.as_posix()

    def list_files(self, dataset_path=None, recursive=True, dataset_id=None):
        # type: (Optional[str], bool, Optional[str]) -> List[str]
        """
        returns a list of files in the current dataset
        If dataset_id is provided, return a list of files that remained unchanged since the specified dataset_version

        :param dataset_path: Only match files matching the dataset_path (including wildcards).
            Example: 'folder/sub/*.json'
        :param recursive: If True (default) matching dataset_path recursively
        :param dataset_id: Filter list based on the dataset id containing the latest version of the file.
            Default: None, do not filter files based on parent dataset.

        :return: List of files with relative path
            (files might not be available locally until get_local_copy() is called)
        """
        files = self._dataset_file_entries.keys() if not dataset_id else \
            [k for k, v in self._dataset_file_entries.items() if v.parent_dataset_id == dataset_id]

        if not dataset_path:
            return sorted(files)

        if dataset_path.startswith('/'):
            dataset_path = dataset_path[1:]

        if not recursive:
            return sorted([k for k in files if fnmatch(k + '/', dataset_path + '/')])

        wildcard = dataset_path.split('/')[-1]
        path = dataset_path[:-len(wildcard)] + '*'
        return sorted([k for k in files if fnmatch(k, path) and fnmatch(k, '*/' + wildcard)])

    def list_removed_files(self, dataset_id=None):
        # type: (str) -> List[str]
        """
        return a list of files removed when comparing to a specific dataset_version

        :param dataset_id: dataset id (str) to compare against, if None is given compare against the parents datasets
        :return: List of files with relative path
            (files might not be available locally until get_local_copy() is called)
        """
        datasets = self._dependency_graph[self._id] if not dataset_id or dataset_id == self._id else [dataset_id]
        unified_list = set()
        for ds_id in datasets:
            dataset = self.get(dataset_id=ds_id)
            unified_list |= set(dataset._dataset_file_entries.keys())

        removed_list = [f for f in unified_list if f not in self._dataset_file_entries]
        return sorted(removed_list)

    def list_modified_files(self, dataset_id=None):
        # type: (str) -> List[str]
        """
        return a list of files modified when comparing to a specific dataset_version

        :param dataset_id: dataset id (str) to compare against, if None is given compare against the parents datasets
        :return: List of files with relative path
            (files might not be available locally until get_local_copy() is called)
        """
        datasets = self._dependency_graph[self._id] if not dataset_id or dataset_id == self._id else [dataset_id]
        unified_list = dict()
        for ds_id in datasets:
            dataset = self.get(dataset_id=ds_id)
            unified_list.update(dict((k, v.hash) for k, v in dataset._dataset_file_entries.items()))

        modified_list = [k for k, v in self._dataset_file_entries.items()
                         if k in unified_list and v.hash != unified_list[k]]
        return sorted(modified_list)

    def list_added_files(self, dataset_id=None):
        # type: (str) -> List[str]
        """
        return a list of files added when comparing to a specific dataset_version

        :param dataset_id: dataset id (str) to compare against, if None is given compare against the parents datasets
        :return: List of files with relative path
            (files might not be available locally until get_local_copy() is called)
        """
        datasets = self._dependency_graph[self._id] if not dataset_id or dataset_id == self._id else [dataset_id]
        unified_list = set()
        for ds_id in datasets:
            dataset = self.get(dataset_id=ds_id)
            unified_list |= set(dataset._dataset_file_entries.keys())

        added_list = [f for f in self._dataset_file_entries.keys() if f not in unified_list]
        return sorted(added_list)

    def get_dependency_graph(self):
        """
        return the DAG of the dataset dependencies (all previous dataset version and their parents)

        Example:

        .. code-block:: py

            {
                'current_dataset_id': ['parent_1_id', 'parent_2_id'],
                'parent_2_id': ['parent_1_id'],
                'parent_1_id': [],
            }

        :return: dict representing the genealogy dag graph of the current dataset
        """
        return deepcopy(self._dependency_graph)

    def verify_dataset_hash(self, local_copy_path=None, skip_hash=False, verbose=False):
        # type: (Optional[str], bool, bool) -> List[str]
        """
        Verify the current copy of the dataset against the stored hash

        :param local_copy_path: Specify local path containing a copy of the dataset,
            If not provide use the cached folder
        :param skip_hash: If True, skip hash checks and verify file size only
        :param verbose: If True print errors while testing dataset files hash
        :return: List of files with unmatched hashes
        """
        local_path = local_copy_path or self.get_local_copy()

        def compare(file_entry):
            file_entry_copy = copy(file_entry)
            file_entry_copy.local_path = (Path(local_path) / file_entry.relative_path).as_posix()
            if skip_hash:
                file_entry_copy.size = Path(file_entry_copy.local_path).stat().st_size
                if file_entry_copy.size != file_entry.size:
                    if verbose:
                        print('Error: file size mismatch {} expected size {} current {}'.format(
                            file_entry.relative_path, file_entry.size, file_entry_copy.size))
                    return file_entry
            else:
                self._calc_file_hash(file_entry_copy)
                if file_entry_copy.hash != file_entry.hash:
                    if verbose:
                        print('Error: hash mismatch {} expected size/hash {}/{} recalculated {}/{}'.format(
                            file_entry.relative_path,
                            file_entry.size, file_entry.hash,
                            file_entry_copy.size, file_entry_copy.hash))
                    return file_entry

            return None

        pool = ThreadPool(cpu_count() * 2)
        matching_errors = pool.map(compare, self._dataset_file_entries.values())
        pool.close()
        return [f.relative_path for f in matching_errors if f is not None]

    def get_default_storage(self):
        # type: () -> Optional[str]
        """
        Return the default storage location of the dataset

        :return: URL for the default storage location
        """
        if not self._task:
            return None
        return self._task.output_uri or self._task.get_logger().get_default_upload_destination()

    @classmethod
    def create(
            cls,
            dataset_name=None,  # type: Optional[str]
            dataset_project=None,  # type: Optional[str]
            dataset_tags=None,  # type: Optional[Sequence[str]]
            parent_datasets=None,  # type: Optional[Sequence[Union[str, Dataset]]]
            use_current_task=False  # type: bool
    ):
        # type: (...) -> "Dataset"
        """
        Create a new dataset. Multiple dataset parents are supported.
        Merging of parent datasets is done based on the order,
        where each one can override overlapping files in the previous parent

        :param dataset_name: Naming the new dataset
        :param dataset_project: Project containing the dataset.
            If not specified, infer project name form parent datasets
        :param dataset_tags: Optional, list of tags (strings) to attach to the newly created Dataset
        :param parent_datasets: Expand a parent dataset by adding/removing files
        :param use_current_task: False (default), a new Dataset task is created.
            If True, the dataset is created on the current Task.
        :return: Newly created Dataset object
        """
        parent_datasets = [cls.get(dataset_id=p) if not isinstance(p, Dataset) else p for p in (parent_datasets or [])]
        if any(not p.is_final() for p in parent_datasets):
            raise ValueError("Cannot inherit from a parent that was not finalized/closed")

        # if dataset name + project are None, default to use current_task
        if dataset_project is None and dataset_name is None and not use_current_task:
            LoggerRoot.get_base_logger().info('New dataset project/name not provided, storing on Current Task')
            use_current_task = True

        # get project name
        if not dataset_project and not use_current_task:
            if not parent_datasets:
                raise ValueError("Missing dataset project name. Could not infer project name from parent dataset.")
            # get project name from parent dataset
            dataset_project = parent_datasets[-1]._task.get_project_name()

        # merge datasets according to order
        dataset_file_entries = {}
        dependency_graph = {}
        for p in parent_datasets:
            dataset_file_entries.update(deepcopy(p._dataset_file_entries))
            dependency_graph.update(deepcopy(p._dependency_graph))
        instance = cls(_private=cls.__private_magic,
                       dataset_project=dataset_project,
                       dataset_name=dataset_name,
                       dataset_tags=dataset_tags,
                       task=Task.current_task() if use_current_task else None)
        instance._using_current_task = use_current_task
        instance._task.get_logger().report_text('Dataset created', print_console=False)
        instance._dataset_file_entries = dataset_file_entries
        instance._dependency_graph = dependency_graph
        instance._dependency_graph[instance._id] = [p._id for p in parent_datasets]
        instance._serialize()
        instance._task.flush(wait_for_uploads=True)
        cls._set_project_system_tags(instance._task)
        return instance

    @classmethod
    def delete(cls, dataset_id=None, dataset_project=None, dataset_name=None, force=False):
        # type: (Optional[str], Optional[str], Optional[str], bool) -> ()
        """
        Delete a dataset, raise exception if dataset is used by other dataset versions.
        Use force=True to forcefully delete the dataset

        :param dataset_id: Dataset id to delete
        :param dataset_project: Project containing the dataset
        :param dataset_name: Naming the new dataset
        :param force: If True delete even if other datasets depend on the specified dataset version
        """
        mutually_exclusive(dataset_id=dataset_id, dataset_project=dataset_project)
        mutually_exclusive(dataset_id=dataset_id, dataset_name=dataset_name)
        if not dataset_id:
            tasks = Task.get_tasks(
                project_name=dataset_project,
                task_name=exact_match_regex(dataset_name) if dataset_name else None,
                task_filter=dict(
                    system_tags=[cls.__tag],
                    type=[str(Task.TaskTypes.data_processing)],
                    page_size=2, page=0,)
            )
            if not tasks:
                raise ValueError("Dataset project={} name={} could not be found".format(dataset_project, dataset_name))
            if len(tasks) > 1:
                raise ValueError("Too many datasets matching project={} name={}".format(dataset_project, dataset_name))
            dataset_id = tasks[0].id

        # check if someone is using the datasets
        if not force:
            # todo: use Task runtime_properties
            # noinspection PyProtectedMember
            dependencies = Task._query_tasks(
                system_tags=[cls.__tag],
                type=[str(Task.TaskTypes.data_processing)],
                only_fields=['created', 'id', 'name'],
                search_text='{}'.format(cls._get_dataset_id_hash(dataset_id))
            )
            # filter us out
            if dependencies:
                dependencies = [d for d in dependencies if d.id != dataset_id]
            if dependencies:
                raise ValueError("Dataset id={} is used by datasets: {}".format(
                    dataset_id, [d.id for d in dependencies]))

        client = APIClient()
        # notice the force here is a must, since the state is never draft
        # noinspection PyBroadException
        try:
            t = client.tasks.get_by_id(dataset_id)
        except Exception:
            t = None
        if not t:
            raise ValueError("Dataset id={} could not be found".format(dataset_id))
        if str(t.type) != str(Task.TaskTypes.data_processing) or cls.__tag not in t.system_tags:
            raise ValueError("Dataset id={} is not of type Dataset".format(dataset_id))

        task = Task.get_task(task_id=dataset_id)
        # first delete all the artifacts from the dataset
        for artifact in task.artifacts.values():
            h = StorageHelper.get(artifact.url)
            # noinspection PyBroadException
            try:
                h.delete(artifact.url)
            except Exception as ex:
                LoggerRoot.get_base_logger().warning('Failed deleting remote file \'{}\': {}'.format(
                    artifact.url, ex))

        # now delete the actual task
        client.tasks.delete(task=dataset_id, force=True)

    @classmethod
    def get(
            cls,
            dataset_id=None,  # type: Optional[str]
            dataset_project=None,  # type: Optional[str]
            dataset_name=None,  # type: Optional[str]
            dataset_tags=None,  # type: Optional[Sequence[str]]
            only_completed=False,  # type: bool
            only_published=False,  # type: bool
            auto_create=False,   # type: bool
            writable_copy=False  # type: bool
    ):
        # type: (...) -> "Dataset"
        """
        Get a specific Dataset. If only dataset_project is given, return the last Dataset in the Dataset project

        :param dataset_id: Requested Dataset ID
        :param dataset_project: Requested Dataset project name
        :param dataset_name: Requested Dataset name
        :param dataset_tags: Requested Dataset tags (list of tag strings)
        :param only_completed: Return only if the requested dataset is completed or published
        :param only_published: Return only if the requested dataset is published
        :param auto_create: Create new dataset if it does not exist yet
        :param writable_copy: Get a newly created mutable dataset with the current one as its parent,
            so new files can added to the instance.
        :return: Dataset object
        """
        mutually_exclusive(dataset_id=dataset_id, dataset_project=dataset_project, _require_at_least_one=False)
        mutually_exclusive(dataset_id=dataset_id, dataset_name=dataset_name, _require_at_least_one=False)
        if not any([dataset_id, dataset_project, dataset_name, dataset_tags]):
            raise ValueError("Dataset selection criteria not met. Didn't provide id/name/project/tags correctly.")

        if auto_create and not get_existing_project(
            session=Task._get_default_session(), project_name=dataset_project
        ):
            tasks = []
        else:
            tasks = Task.get_tasks(
                task_ids=[dataset_id] if dataset_id else None,
                project_name=dataset_project,
                task_name=exact_match_regex(dataset_name) if dataset_name else None,
                tags=dataset_tags,
                task_filter=dict(
                    system_tags=[cls.__tag, '-archived'], order_by=['-created'],
                    type=[str(Task.TaskTypes.data_processing)],
                    page_size=1, page=0,
                    status=['published'] if only_published else
                    ['published', 'completed', 'closed'] if only_completed else None)
            )

        if not tasks:
            if auto_create:
                instance = Dataset.create(dataset_name=dataset_name, dataset_project=dataset_project,
                                          dataset_tags=dataset_tags)
                return instance
            raise ValueError('Could not find Dataset {} {}'.format(
                'id' if dataset_id else 'project/name',
                dataset_id if dataset_id else (dataset_project, dataset_name)))
        task = tasks[0]
        if task.status == 'created':
            raise ValueError('Dataset id={} is in draft mode, delete and recreate it'.format(task.id))
        force_download = False if task.status in ('stopped', 'published', 'closed', 'completed') else True
        if cls.__state_entry_name in task.artifacts:
            local_state_file = StorageManager.get_local_copy(
                remote_url=task.artifacts[cls.__state_entry_name].url, cache_context=cls.__cache_context,
                extract_archive=False, name=task.id, force_download=force_download)
            if not local_state_file:
                raise ValueError('Could not load Dataset id={} state'.format(task.id))
        else:
            # we could not find the serialized state, start empty
            local_state_file = {}

        instance = cls._deserialize(local_state_file, task)
        # remove the artifact, just in case
        if force_download and local_state_file:
            os.unlink(local_state_file)

        # Now we have the requested dataset, but if we want a mutable copy instead, we create a new dataset with the
        # current one as its parent. So one can add files to it and finalize as a new version.
        if writable_copy:
            writeable_instance = Dataset.create(
                dataset_name=instance.name,
                dataset_project=instance.project,
                dataset_tags=instance.tags,
                parent_datasets=[instance.id],
            )
            return writeable_instance

        return instance

    def get_logger(self):
        # type: () -> Logger
        """
        Return a Logger object for the Dataset, allowing users to report statistics metrics
        and debug samples on the Dataset itself
        :return: Logger object
        """
        return self._task.get_logger()

    def get_num_chunks(self, include_parents=True):
        # type: (bool) -> int
        """
        Return the number of chunks stored on this dataset
        (it does not imply on the number of chunks parent versions store)

        :param include_parents: If True (default),
        return the total number of chunks from this version and all parent versions.
        If False, only return the number of chunks we stored on this specific version.

        :return: Number of chunks stored on the dataset.
        """
        if not include_parents:
            return len(self._get_data_artifact_names())

        return sum(self._get_dependency_chunk_lookup().values())

    @classmethod
    def squash(
            cls,
            dataset_name,  # type: str
            dataset_ids=None,  # type: Optional[Sequence[Union[str, Dataset]]]
            dataset_project_name_pairs=None,  # type: Optional[Sequence[(str, str)]]
            output_url=None,  # type: Optional[str]
    ):
        # type: (...) -> "Dataset"
        """
        Generate a new dataset from the squashed set of dataset versions.
        If a single version is given it will squash to the root (i.e. create single standalone version)
        If a set of versions are given it will squash the versions diff into a single version

        :param dataset_name: Target name for the newly generated squashed dataset
        :param dataset_ids: List of dataset Ids (or objects) to squash. Notice order does matter.
            The versions are merged from first to last.
        :param dataset_project_name_pairs: List of pairs (project_name, dataset_name) to squash.
            Notice order does matter. The versions are merged from first to last.
        :param output_url: Target storage for the compressed dataset (default: file server)
            Examples: `s3://bucket/data`, `gs://bucket/data` , `azure://bucket/data` , `/mnt/share/data`
        :return: Newly created dataset object.
        """
        mutually_exclusive(dataset_ids=dataset_ids, dataset_project_name_pairs=dataset_project_name_pairs)
        datasets = [cls.get(dataset_id=d) for d in dataset_ids] if dataset_ids else \
            [cls.get(dataset_project=pair[0], dataset_name=pair[1]) for pair in dataset_project_name_pairs]
        # single dataset to squash, squash it all.
        if len(datasets) == 1:
            temp_folder = datasets[0].get_local_copy()
            parents = set()
        else:
            parents = None
            temp_folder = Path(mkdtemp(prefix='squash-datasets.'))
            pool = ThreadPool()
            for ds in datasets:
                base_folder = Path(ds._extract_dataset_archive())
                files = [f.relative_path for f in ds.file_entries if f.parent_dataset_id == ds.id]
                pool.map(
                    lambda x:
                        (temp_folder / x).parent.mkdir(parents=True, exist_ok=True) or
                        shutil.copy((base_folder / x).as_posix(), (temp_folder / x).as_posix(), follow_symlinks=True),
                    files)
                parents = set(ds._get_parents()) if parents is None else (parents & set(ds._get_parents()))
            pool.close()

        squashed_ds = cls.create(
            dataset_project=datasets[0].project, dataset_name=dataset_name, parent_datasets=list(parents))
        squashed_ds._task.get_logger().report_text('Squashing dataset', print_console=False)
        squashed_ds.add_files(temp_folder)
        squashed_ds.upload(output_url=output_url)
        squashed_ds.finalize()
        return squashed_ds

    @classmethod
    def list_datasets(cls, dataset_project=None, partial_name=None, tags=None, ids=None, only_completed=True):
        # type: (Optional[str], Optional[str], Optional[Sequence[str]], Optional[Sequence[str]], bool) -> List[dict]
        """
        Query list of dataset in the system

        :param dataset_project: Specify dataset project name
        :param partial_name: Specify partial match to a dataset name
        :param tags: Specify user tags
        :param ids: List specific dataset based on IDs list
        :param only_completed: If False return dataset that are still in progress (uploading/edited etc.)
        :return: List of dictionaries with dataset information
            Example: [{'name': name, 'project': project name, 'id': dataset_id, 'created': date_created},]
        """
        # noinspection PyProtectedMember
        datasets = Task._query_tasks(
            task_ids=ids or None, project_name=dataset_project or None,
            task_name=partial_name,
            system_tags=[cls.__tag],
            type=[str(Task.TaskTypes.data_processing)],
            tags=tags or None,
            status=['stopped', 'published', 'completed', 'closed'] if only_completed else None,
            only_fields=['created', 'id', 'name', 'project', 'tags']
        )
        project_ids = {d.project for d in datasets}
        # noinspection PyProtectedMember
        project_id_lookup = {d: Task._get_project_name(d) for d in project_ids}
        return [
            {'name': d.name,
             'created': d.created,
             'project': project_id_lookup[d.project],
             'id': d.id,
             'tags': d.tags}
            for d in datasets
        ]

    def _add_files(self,
                   path,  # type: Union[str, Path, _Path]
                   wildcard=None,  # type: Optional[Union[str, Sequence[str]]]
                   local_base_folder=None,  # type: Optional[str]
                   dataset_path=None,  # type: Optional[str]
                   recursive=True,  # type: bool
                   verbose=False  # type: bool
                   ):
        # type: (...) -> int
        """
        Add a folder into the current dataset. calculate file hash,
        and compare against parent, mark files to be uploaded

        :param path: Add a folder/file to the dataset
        :param wildcard: add only specific set of files.
            Wildcard matching, can be a single string or a list of wildcards)
        :param local_base_folder: files will be located based on their relative path from local_base_folder
        :param dataset_path: where in the dataset the folder/files should be located
        :param recursive: If True match all wildcard files recursively
        :param verbose: If True print to console added files
        """
        if dataset_path and dataset_path.startswith('/'):
            dataset_path = dataset_path[1:]
        path = Path(path)
        local_base_folder = Path(local_base_folder or path)
        wildcard = wildcard or '*'
        # single file, no need for threading
        if path.is_file():
            if not local_base_folder.is_dir():
                local_base_folder = local_base_folder.parent
            file_entry = self._calc_file_hash(
                FileEntry(local_path=path.absolute().as_posix(),
                          relative_path=(Path(dataset_path or '.') / path.relative_to(local_base_folder)).as_posix(),
                          parent_dataset_id=self._id))
            file_entries = [file_entry]
        else:
            # if not a folder raise exception
            if not path.is_dir():
                raise ValueError("Could not find file/folder \'{}\'", path.as_posix())

            # prepare a list of files
            files = list(path.rglob(wildcard)) if recursive else list(path.glob(wildcard))
            file_entries = [
                FileEntry(
                    parent_dataset_id=self._id,
                    local_path=f.absolute().as_posix(),
                    relative_path=(Path(dataset_path or '.') / f.relative_to(local_base_folder)).as_posix())
                for f in files if f.is_file()]
            self._task.get_logger().report_text('Generating SHA2 hash for {} files'.format(len(file_entries)))
            pool = ThreadPool(cpu_count() * 2)
            try:
                import tqdm  # noqa
                for _ in tqdm.tqdm(pool.imap_unordered(self._calc_file_hash, file_entries), total=len(file_entries)):
                    pass
            except ImportError:
                pool.map(self._calc_file_hash, file_entries)
            pool.close()
            self._task.get_logger().report_text('Hash generation completed')

        # merge back into the dataset
        count = 0
        for f in file_entries:
            ds_cur_f = self._dataset_file_entries.get(f.relative_path)
            if not ds_cur_f:
                if verbose:
                    self._task.get_logger().report_text('Add {}'.format(f.relative_path))
                self._dataset_file_entries[f.relative_path] = f
                count += 1
            elif ds_cur_f.hash != f.hash:
                if verbose:
                    self._task.get_logger().report_text('Modified {}'.format(f.relative_path))
                self._dataset_file_entries[f.relative_path] = f
                count += 1
            elif f.parent_dataset_id == self._id and ds_cur_f.parent_dataset_id == self._id:
                # check if we have the file in an already uploaded chunk
                if ds_cur_f.local_path is None:
                    # skipping, already uploaded.
                    if verbose:
                        self._task.get_logger().report_text('Skipping {}'.format(f.relative_path))
                else:
                    # if we never uploaded it, mark for upload
                    if verbose:
                        self._task.get_logger().report_text('Re-Added {}'.format(f.relative_path))
                    self._dataset_file_entries[f.relative_path] = f
                    count += 1
            else:
                if verbose:
                    self._task.get_logger().report_text('Unchanged {}'.format(f.relative_path))

        return count

    def _update_dependency_graph(self):
        """
        Update the dependency graph based on the current self._dataset_file_entries state
        :return:
        """
        # collect all dataset versions
        used_dataset_versions = set(f.parent_dataset_id for f in self._dataset_file_entries.values())
        used_dataset_versions.add(self._id)
        current_parents = self._dependency_graph.get(self._id) or []
        # remove parent versions we no longer need from the main version list
        # per version, remove unnecessary parent versions, if we do not need them
        self._dependency_graph = {
            k: [p for p in parents or [] if p in used_dataset_versions]
            for k, parents in self._dependency_graph.items() if k in used_dataset_versions}
        # make sure we do not remove our parents, for geology sake
        self._dependency_graph[self._id] = current_parents

    def _serialize(self, update_dependency_chunk_lookup=False):
        # type: (bool) -> ()
        """
        store current state of the Dataset for later use

        :param update_dependency_chunk_lookup: If True, update the parent versions number of chunks

        :return: object to be used for later deserialization
        """
        self._update_dependency_graph()

        state = dict(
            dataset_file_entries=[f.as_dict() for f in self._dataset_file_entries.values()],
            dependency_graph=self._dependency_graph,
            id=self._id,
            dirty=self._dirty,
        )
        if update_dependency_chunk_lookup:
            state['dependency_chunk_lookup'] = self._build_dependency_chunk_lookup()

        modified_files = [f['size'] for f in state['dataset_file_entries'] if f.get('parent_dataset_id') == self._id]
        preview = \
            'Dataset state\n' \
            'Files added/modified: {0} - total size {1}\n' \
            'Current dependency graph: {2}\n'.format(
                len(modified_files), format_size(sum(modified_files)),
                json.dumps(self._dependency_graph, indent=2, sort_keys=True))
        # store as artifact of the Task.
        self._task.upload_artifact(
            name=self.__state_entry_name, artifact_object=state, preview=preview, wait_on_upload=True)

    def _download_dataset_archives(self):
        """
        Download the dataset archive, return a link to locally stored zip file
        :return: List of paths to locally stored zip files
        """
        pass  # TODO: implement

    def _extract_dataset_archive(
            self,
            force=False,
            selected_chunks=None,
            lock_target_folder=False,
            cleanup_target_folder=True,
            target_folder=None,
    ):
        # type: (bool, Optional[List[int]], bool, bool, Optional[Path]) -> str
        """
        Download the dataset archive, and extract the zip content to a cached folder.
        Notice no merging is done.

        :param force: If True extract dataset content even if target folder exists and is not empty
        :param selected_chunks: Optional, if provided only download the selected chunks (index) of the Dataset.
            Example: Assuming 8 chunks on this version
            selected_chunks=[0,1,2]
        :param lock_target_folder: If True, local the target folder so the next cleanup will not delete
            Notice you should unlock it manually, or wait fro the process to fnish for auto unlocking.
        :param cleanup_target_folder: If True remove target folder recursively
        :param target_folder: If provided use the specified target folder, default, auto generate from Dataset ID.

        :return: Path to a local storage extracted archive
        """
        assert selected_chunks is None or isinstance(selected_chunks, (list, tuple))

        if not self._task:
            self._task = Task.get_task(task_id=self._id)

        data_artifact_entries = self._get_data_artifact_names()

        if selected_chunks is not None and data_artifact_entries:
            data_artifact_entries = [
                d for d in data_artifact_entries
                if self._get_chunk_idx_from_artifact_name(d) in selected_chunks]

        # get cache manager
        local_folder = Path(target_folder) if target_folder else \
            self._create_ds_target_folder(lock_target_folder=lock_target_folder)

        # check if we have a dataset with empty change set
        if not data_artifact_entries:
            return local_folder.as_posix()

        # check if target folder is not empty
        if not force and next(local_folder.glob('*'), None):
            return local_folder.as_posix()

        # if we got here, we need to clear the target folder
        local_folder = local_folder.as_posix()
        if cleanup_target_folder:
            shutil.rmtree(local_folder, ignore_errors=True)
        # verify target folder exists
        Path(local_folder).mkdir(parents=True, exist_ok=True)

        def _download_part(data_artifact_name):
            # download the dataset zip
            local_zip = StorageManager.get_local_copy(
                remote_url=self._task.artifacts[data_artifact_name].url, cache_context=self.__cache_context,
                extract_archive=False, name=self._id)
            if not local_zip:
                raise ValueError("Could not download dataset id={} entry={}".format(self._id, data_artifact_name))
            # noinspection PyProtectedMember
            StorageManager._extract_to_cache(
                cached_file=local_zip, name=self._id,
                cache_context=self.__cache_context, target_folder=local_folder, force=True)

        # download al parts in parallel
        # if len(data_artifact_entries) > 1:
        #     pool = ThreadPool()
        #     pool.map(_download_part, data_artifact_entries)
        #     pool.close()
        # else:
        #     _download_part(data_artifact_entries[0])
        for d in data_artifact_entries:
            _download_part(d)

        return local_folder

    def _create_ds_target_folder(self, part=None, num_parts=None, lock_target_folder=True):
        # type: (Optional[int], Optional[int], bool) -> Path
        cache = CacheManager.get_cache_manager(cache_context=self.__cache_context)
        local_folder = Path(cache.get_cache_folder()) / self._get_cache_folder_name(part=part, num_parts=num_parts)
        if lock_target_folder:
            cache.lock_cache_folder(local_folder)
        local_folder.mkdir(parents=True, exist_ok=True)
        return local_folder

    def _get_data_artifact_names(self):
        # type: () -> List[str]
        data_artifact_entries = [
            a for a in self._task.artifacts
            if a and (a == self.__default_data_entry_name or str(a).startswith(self.__data_entry_name_prefix))]
        return data_artifact_entries

    def _get_next_data_artifact_name(self, last_artifact_name=None):
        # type: (Optional[str]) -> str
        if not last_artifact_name:
            data_artifact_entries = self._get_data_artifact_names()
            if len(data_artifact_entries) < 1:
                return self.__default_data_entry_name
        else:
            data_artifact_entries = [last_artifact_name]
        prefix = self.__data_entry_name_prefix
        prefix_len = len(prefix)
        numbers = sorted([int(a[prefix_len:]) for a in data_artifact_entries if a.startswith(prefix)])
        return '{}{:03d}'.format(prefix, numbers[-1]+1 if numbers else 1)

    def _merge_datasets(self, use_soft_links=None, raise_on_error=True, part=None, num_parts=None):
        # type: (bool, bool, Optional[int], Optional[int]) -> str
        """
        download and copy / soft-link, files from all the parent dataset versions
        :param use_soft_links: If True use soft links, default False on windows True on Posix systems
        :param raise_on_error: If True raise exception if dataset merging failed on any file
        :param part: Optional, if provided only download the selected part (index) of the Dataset.
            Notice, if `num_parts` is not provided, number of parts will be equal to the number of chunks.
            This argument is passed to parent versions, as well as the implicit `num_parts`,
            allowing users to get a partial copy of the entire dataset, for multi node/step processing.
        :param num_parts: Optional, If specified normalize the number of chunks stored to the
            requested number of parts. Notice that the actual chunks used per part are rounded down.
            Example: Assuming 8 chunks on this version, and `num_parts=5`, the chunk index used per parts would be:
            part=0 -> chunks[0,5], part=1 -> chunks[1,6], part=2 -> chunks[2,7], part=3 -> chunks[3, ]

        :return: the target folder
        """
        assert part is None or (isinstance(part, int) and part >= 0)
        assert num_parts is None or (isinstance(num_parts, int) and num_parts >= 1)

        if use_soft_links is None:
            use_soft_links = False if is_windows() else True

        if part is not None and not num_parts:
            num_parts = self.get_num_chunks()

        # just create the dataset target folder
        target_base_folder = self._create_ds_target_folder(
            part=part, num_parts=num_parts, lock_target_folder=True)

        # selected specific chunks if `part` was passed
        chunk_selection = None if part is None else self._build_chunk_selection(part=part, num_parts=num_parts)

        # check if target folder is not empty, see if it contains everything we need
        if target_base_folder and next(target_base_folder.iterdir(), None):
            if self._verify_dataset_folder(target_base_folder, part, chunk_selection):
                target_base_folder.touch()
                return target_base_folder.as_posix()
            else:
                LoggerRoot.get_base_logger().info('Dataset needs refreshing, fetching all parent datasets')
                # we should delete the entire cache folder
                shutil.rmtree(target_base_folder.as_posix())
                # make sure we recreate the dataset target folder
                target_base_folder.mkdir(parents=True, exist_ok=True)

        # get the dataset dependencies (if `part` was passed, only selected the ones in the selected part)
        dependencies_by_order = self._get_dependencies_by_order(include_unused=False, include_current=True) \
            if chunk_selection is None else list(chunk_selection.keys())

        # first get our dataset
        if self._id in dependencies_by_order:
            self._extract_dataset_archive(
                force=True,
                selected_chunks=chunk_selection.get(self._id) if chunk_selection else None,
                cleanup_target_folder=True,
                target_folder=target_base_folder,
            )
            dependencies_by_order.remove(self._id)

        # update target folder timestamp
        target_base_folder.touch()

        # if we have no dependencies, we can just return now
        if not dependencies_by_order:
            return target_base_folder.absolute().as_posix()

        # extract parent datasets
        self._extract_parent_datasets(
            target_base_folder=target_base_folder, dependencies_by_order=dependencies_by_order,
            chunk_selection=chunk_selection, use_soft_links=use_soft_links,
            raise_on_error=False, force=False)

        # verify entire dataset (if failed, force downloading parent datasets)
        if not self._verify_dataset_folder(target_base_folder, part, chunk_selection):
            LoggerRoot.get_base_logger().info('Dataset parents need refreshing, re-fetching all parent datasets')
            # we should delete the entire cache folder
            self._extract_parent_datasets(
                target_base_folder=target_base_folder, dependencies_by_order=dependencies_by_order,
                chunk_selection=chunk_selection, use_soft_links=use_soft_links,
                raise_on_error=raise_on_error, force=True)

        return target_base_folder.absolute().as_posix()

    def _get_dependencies_by_order(self, include_unused=False, include_current=True):
        # type: (bool, bool) -> List[str]
        """
        Return the dataset dependencies by order of application (from the last to the current)
        :param include_unused: If True include unused datasets in the dependencies
        :param include_current: If True include the current dataset ID as the last ID in the list
        :return: list of str representing the datasets id
        """
        roots = [self._id]
        dependencies = []
        # noinspection DuplicatedCode
        while roots:
            r = roots.pop(0)
            if r not in dependencies:
                dependencies.append(r)
            # add the parents of the current node, only if the parents are in the general graph node list
            if include_unused and r not in self._dependency_graph:
                roots.extend(list(reversed(
                    [p for p in (self.get(dataset_id=r)._get_parents() or []) if p not in roots])))
            else:
                roots.extend(list(reversed(
                    [p for p in (self._dependency_graph.get(r) or [])
                     if p not in roots and (include_unused or (p in self._dependency_graph))])))

        # make sure we cover leftovers
        leftovers = set(self._dependency_graph.keys()) - set(dependencies)
        if leftovers:
            roots = list(leftovers)
            # noinspection DuplicatedCode
            while roots:
                r = roots.pop(0)
                if r not in dependencies:
                    dependencies.append(r)
                # add the parents of the current node, only if the parents are in the general graph node list
                if include_unused and r not in self._dependency_graph:
                    roots.extend(list(reversed(
                        [p for p in (self.get(dataset_id=r)._get_parents() or []) if p not in roots])))
                else:
                    roots.extend(list(reversed(
                        [p for p in (self._dependency_graph.get(r) or [])
                         if p not in roots and (include_unused or (p in self._dependency_graph))])))

        # skip our id
        dependencies = list(reversed(dependencies[1:]))
        return (dependencies + [self._id]) if include_current else dependencies

    def _get_parents(self):
        # type: () -> Sequence[str]
        """
        Return a list of direct parent datasets (str)
        :return: list of dataset ids
        """
        return self._dependency_graph[self.id]

    @classmethod
    def _deserialize(cls, stored_state, task):
        # type: (Union[dict, str, Path, _Path], Task) -> "Dataset"
        """
        reload a dataset state from the stored_state object
        :param task: Task object associated with the dataset
        :return: A Dataset object
        """
        assert isinstance(stored_state, (dict, str, Path, _Path))

        if isinstance(stored_state, (str, Path, _Path)):
            stored_state_file = Path(stored_state).as_posix()
            with open(stored_state_file, 'rt') as f:
                stored_state = json.load(f)

        instance = cls(_private=cls.__private_magic, task=task)
        # assert instance._id == stored_state['id']  # They should match
        instance._dependency_graph = stored_state.get('dependency_graph', {})
        instance._dirty = stored_state.get('dirty', False)
        instance._dataset_file_entries = {
            s['relative_path']: FileEntry(**s) for s in stored_state.get('dataset_file_entries', [])}
        if stored_state.get('dependency_chunk_lookup') is not None:
            instance._dependency_chunk_lookup = stored_state.get('dependency_chunk_lookup')

        # update the last used artifact (remove the one we never serialized, they rae considered broken)
        if task.status in ('in_progress', 'created', 'stopped'):
            artifact_names = set([
                a.artifact_name for a in instance._dataset_file_entries.values()
                if a.artifact_name and a.parent_dataset_id == instance._id])
            missing_artifact_name = set(instance._get_data_artifact_names()) - artifact_names
            if missing_artifact_name:
                instance._task._delete_artifacts(list(missing_artifact_name))
                # if we removed any data artifact, update the next data artifact name
                instance._data_artifact_name = instance._get_next_data_artifact_name()

        return instance

    @staticmethod
    def _calc_file_hash(file_entry):
        # calculate hash
        file_entry.hash, _ = sha256sum(file_entry.local_path)
        file_entry.size = Path(file_entry.local_path).stat().st_size
        return file_entry

    @classmethod
    def _get_dataset_id_hash(cls, dataset_id):
        # type: (str) -> str
        """
        Return hash used to search for the dataset id in text fields.
        This is not a strong hash and used for defining dependencies.
        :param dataset_id:
        :return:
        """
        return 'dsh{}'.format(md5text(dataset_id))

    def _build_dependency_chunk_lookup(self):
        # type: () -> Dict[str, int]
        """
        Build the dependency dataset id to number-of-chunks, lookup table
        :return: lookup dictionary from dataset-id to number of chunks
        """
        # with ThreadPool() as pool:
        #     chunks_lookup = pool.map(
        #         lambda d: (d, Dataset.get(dataset_id=d).get_num_chunks()),
        #         self._dependency_graph.keys())
        #     return dict(chunks_lookup)
        chunks_lookup = map(
            lambda d: (d, (self if d == self.id else Dataset.get(dataset_id=d)).get_num_chunks(include_parents=False)),
            self._dependency_graph.keys())
        return dict(chunks_lookup)

    def _get_cache_folder_name(self, part=None, num_parts=None):
        # type: (Optional[int], Optional[int]) -> str
        if part is None:
            return '{}{}'.format(self.__cache_folder_prefix, self._id)
        return '{}{}_{}_{}'.format(self.__cache_folder_prefix, self._id, part, num_parts)

    def _add_script_call(self, func_name, **kwargs):
        # type: (str, **Any) -> ()

        # if we never created the Task, we should not add the script calls
        if not self._created_task:
            return

        args = ', '.join('\n    {}={}'.format(k, '\''+str(v)+'\'' if isinstance(v, (str, Path, _Path)) else v)
                         for k, v in kwargs.items())
        if args:
            args += '\n'
        line = 'ds.{}({})\n'.format(func_name, args)
        self._task.data.script.diff += line
        # noinspection PyProtectedMember
        self._task._edit(script=self._task.data.script)

    def _report_dataset_genealogy(self):
        sankey_node = dict(
            label=[],
            color=[],
            customdata=[],
            hovertemplate='%{customdata}<extra></extra>',
            hoverlabel={"align": "left"},
        )
        sankey_link = dict(
            source=[],
            target=[],
            value=[],
            hovertemplate='<extra></extra>',
        )
        # get DAG nodes
        nodes = self._get_dependencies_by_order(include_unused=True, include_current=True)
        # dataset name lookup
        # noinspection PyProtectedMember
        node_names = {t.id: t.name for t in Task._query_tasks(task_ids=nodes, only_fields=['id', 'name'])}
        node_details = {}
        # Generate table and details
        table_values = [["Dataset id", "name", "removed", "modified", "added", "size"]]
        for node in nodes:
            count = 0
            size = 0
            for f in self._dataset_file_entries.values():
                if f.parent_dataset_id == node:
                    count += 1
                    size += f.size
            removed = len(self.list_removed_files(node))
            modified = len(self.list_modified_files(node))
            table_values += [[node, node_names.get(node, ''),
                              removed, modified, max(0, count-modified), format_size(size)]]
            node_details[node] = [removed, modified, max(0, count-modified), format_size(size)]

        # create DAG
        visited = []
        # add nodes
        for idx, node in enumerate(nodes):
            visited.append(node)
            sankey_node['color'].append("mediumpurple" if node == self.id else "lightblue")
            sankey_node['label'].append('{}'.format(node))
            sankey_node['customdata'].append(
                "name {}<br />removed {}<br />modified {}<br />added {}<br />size {}".format(
                    node_names.get(node, ''), *node_details[node]))

        # add edges
        for idx, node in enumerate(nodes):
            if node in self._dependency_graph:
                parents = [visited.index(p) for p in self._dependency_graph[node] or [] if p in visited]
            else:
                parents = [visited.index(p) for p in self.get(dataset_id=node)._get_parents() or [] if p in visited]

            for p in parents:
                sankey_link['source'].append(p)
                sankey_link['target'].append(idx)
                sankey_link['value'].append(max(1, node_details[visited[p]][-2]))

        if len(nodes) > 1:
            # create the sankey graph
            dag_flow = dict(
                link=sankey_link,
                node=sankey_node,
                textfont=dict(color='rgba(0,0,0,255)', size=10),
                type='sankey',
                orientation='h'
            )
            fig = dict(data=[dag_flow], layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
        elif len(nodes) == 1:
            # hack, show single node sankey
            singles_flow = dict(
                x=list(range(len(nodes))), y=[1] * len(nodes),
                text=sankey_node['label'],
                customdata=sankey_node['customdata'],
                mode='markers',
                hovertemplate='%{customdata}<extra></extra>',
                marker=dict(
                    color=[v for i, v in enumerate(sankey_node['color']) if i in nodes],
                    size=[40] * len(nodes),
                ),
                showlegend=False,
                type='scatter',
            )
            # only single nodes
            fig = dict(data=[singles_flow], layout={
                'hovermode': 'closest', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
        else:
            fig = None

        # report genealogy
        if fig:
            self._task.get_logger().report_plotly(
                title='Dataset Genealogy', series='', iteration=0, figure=fig)

        # report detailed table
        self._task.get_logger().report_table(
            title='Dataset Summary', series='Details', iteration=0, table_plot=table_values,
            extra_layout={"title": "Files by parent dataset id"})

        # report the detailed content of the dataset as configuration,
        # this allows for easy version comparison in the UI
        dataset_details = None
        dataset_details_header = None
        dataset_details_header_template = 'File Name ({} files) - File Size (total {}) - Hash (SHA2)\n'
        if len(self._dataset_file_entries) < self.__preview_max_file_entries:
            file_entries = sorted(self._dataset_file_entries.values(), key=lambda x: x.relative_path)
            dataset_details_header = dataset_details_header_template.format(
                    len(file_entries), format_size(sum(f.size for f in file_entries))
                )
            dataset_details = dataset_details_header + \
                '\n'.join('{} - {} - {}'.format(f.relative_path, f.size, f.hash) for f in file_entries)

        # too large to store
        if not dataset_details or len(dataset_details) > self.__preview_max_size:
            if not dataset_details_header:
                dataset_details_header = dataset_details_header_template.format(
                    len(self._dataset_file_entries),
                    format_size(sum(f.size for f in self._dataset_file_entries.values()))
                )
            dataset_details = dataset_details_header + 'Dataset content is too large to preview'

        # noinspection PyProtectedMember
        self._task._set_configuration(
            name='Dataset Content', description='Dataset content preview',
            config_type='read-only',
            config_text=dataset_details
        )

    @classmethod
    def _set_project_system_tags(cls, task):
        from ..backend_api.services import projects
        res = task.send(projects.GetByIdRequest(project=task.project), raise_on_errors=False)
        if not res or not res.response or not res.response.project:
            return
        system_tags = res.response.project.system_tags or []
        if cls.__tag not in system_tags:
            system_tags += [cls.__tag]
            task.send(projects.UpdateRequest(project=task.project, system_tags=system_tags), raise_on_errors=False)

    def is_dirty(self):
        # type: () -> bool
        """
        Return True if the dataset has pending uploads (i.e. we cannot finalize it)

        :return: Return True means dataset has pending uploads, call 'upload' to start an upload process.
        """
        return self._dirty

    def _extract_parent_datasets(
            self,
            target_base_folder,
            dependencies_by_order,
            chunk_selection,
            use_soft_links,
            raise_on_error,
            force
    ):
        # type: (Path, List[str], Optional[dict], bool, bool, bool) -> ()
        # create thread pool, for creating soft-links / copying
        # todo: parallelize by parent datasets
        pool = ThreadPool(cpu_count() * 2)
        for dataset_version_id in dependencies_by_order:
            # make sure we skip over empty dependencies
            if dataset_version_id not in self._dependency_graph:
                continue
            selected_chunks = chunk_selection.get(dataset_version_id) if chunk_selection else None

            ds = Dataset.get(dataset_id=dataset_version_id)
            ds_base_folder = Path(ds._extract_dataset_archive(
                selected_chunks=selected_chunks,
                force=force,
                lock_target_folder=True,
                cleanup_target_folder=False,
            ))
            ds_base_folder.touch()

            def copy_file(file_entry):
                if file_entry.parent_dataset_id != dataset_version_id or \
                        (selected_chunks is not None and
                         self._get_chunk_idx_from_artifact_name(file_entry.artifact_name) not in selected_chunks):
                    return
                source = (ds_base_folder / file_entry.relative_path).as_posix()
                target = (target_base_folder / file_entry.relative_path).as_posix()
                try:
                    # make sure we have can overwrite the target file
                    # noinspection PyBroadException
                    try:
                        os.unlink(target)
                    except Exception:
                        Path(target).parent.mkdir(parents=True, exist_ok=True)

                    # copy / link
                    if use_soft_links:
                        if not os.path.isfile(source):
                            raise ValueError("Extracted file missing {}".format(source))
                        os.symlink(source, target)
                    else:
                        shutil.copy2(source, target, follow_symlinks=True)
                except Exception as ex:
                    LoggerRoot.get_base_logger().warning('{}\nFailed {} file {} to {}'.format(
                        ex, 'linking' if use_soft_links else 'copying', source, target))
                    return ex

                return None

            errors = pool.map(copy_file, self._dataset_file_entries.values())

            CacheManager.get_cache_manager(cache_context=self.__cache_context).unlock_cache_folder(
                ds_base_folder.as_posix())

            if raise_on_error and any(errors):
                raise ValueError("Dataset merging failed: {}".format([e for e in errors if e is not None]))
        pool.close()

    def _verify_dataset_folder(self, target_base_folder, part, chunk_selection):
        # type: (Path, Optional[int], Optional[dict]) -> bool
        target_base_folder = Path(target_base_folder)
        # check dataset file size, if we have a full match no need for parent dataset download / merge
        verified = True
        # noinspection PyBroadException
        try:
            for f in self._dataset_file_entries.values():
                # check if we need it for the current part
                if part is not None:
                    f_parts = chunk_selection.get(f.parent_dataset_id, [])
                    # this is not in our current part, no need to check it.
                    if self._get_chunk_idx_from_artifact_name(f.artifact_name) not in f_parts:
                        continue

                # check if the local size and the stored size match (faster than comparing hash)
                if (target_base_folder / f.relative_path).stat().st_size != f.size:
                    verified = False
                    break

        except Exception:
            verified = False

        return verified

    def _get_dependency_chunk_lookup(self):
        # type: () -> Dict[str, int]
        """
        Return The parent dataset ID to number of chunks lookup table
        :return: Dict key is dataset ID, value is total number of chunks for the specific dataset version.
        """
        if self._dependency_chunk_lookup is None:
            self._dependency_chunk_lookup = self._build_dependency_chunk_lookup()
        return self._dependency_chunk_lookup

    def _build_chunk_selection(self, part, num_parts):
        # type: (int, int) -> Dict[str, int]
        """
        Build the selected chunks from each parent version, based on the current selection.
        Notice that for a specific part, one can only get the chunks from parent versions (not including this one)
        :param part: Current part index (between 0 and num_parts-1)
        :param num_parts: Total number of parts to divide the dataset into
        :return: Dict of Dataset ID and their respected chunks used for this part number
        """
        # get the chunk dependencies
        dependency_chunk_lookup = self._get_dependency_chunk_lookup()

        # first collect the total number of chunks
        total_chunks = sum(dependency_chunk_lookup.values())

        avg_chunk_per_part = total_chunks // num_parts
        leftover_chunks = total_chunks % num_parts

        dependencies = self._get_dependencies_by_order(include_unused=False, include_current=True)
        # create the part look up
        ds_id_chunk_list = [(d, i) for d in dependencies for i in range(dependency_chunk_lookup.get(d, 1))]

        # select the chunks for this part
        if part < leftover_chunks:
            indexes = ds_id_chunk_list[part*(avg_chunk_per_part+1):(part+1)*(avg_chunk_per_part+1)]
        else:
            ds_id_chunk_list = ds_id_chunk_list[leftover_chunks*(avg_chunk_per_part+1):]
            indexes = ds_id_chunk_list[(part-leftover_chunks)*avg_chunk_per_part:
                                       (part-leftover_chunks+1)*avg_chunk_per_part]

        # convert to lookup
        chunk_selection = {}
        for d, i in indexes:
            chunk_selection[d] = chunk_selection.get(d, []) + [i]

        return chunk_selection

    @classmethod
    def _get_chunk_idx_from_artifact_name(cls, artifact_name):
        # type: (str) -> int
        if not artifact_name:
            return -1
        artifact_name = str(artifact_name)

        if artifact_name == cls.__default_data_entry_name:
            return 0
        if artifact_name.startswith(cls.__data_entry_name_prefix):
            return int(artifact_name[len(cls.__data_entry_name_prefix):])
        return -1
