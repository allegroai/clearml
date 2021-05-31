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
from ..backend_interface.util import mutually_exclusive, exact_match_regex
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
    # cleared when file is uploaded.
    local_path = attrib(default=None, type=str)

    def as_dict(self):
        # type: () -> Dict
        state = dict(relative_path=self.relative_path, hash=self.hash,
                     parent_dataset_id=self.parent_dataset_id, size=self.size,
                     **dict([('local_path', self.local_path)] if self.local_path else ()))
        return state


class Dataset(object):
    __private_magic = 42 * 1337
    __state_entry_name = 'state'
    __data_entry_name = 'data'
    __cache_context = 'datasets'
    __tag = 'dataset'
    __cache_folder_prefix = 'ds_'
    __dataset_folder_template = CacheManager.set_context_folder_lookup(__cache_context, "{0}_archive_{1}")
    __preview_max_file_entries = 15000
    __preview_max_size = 5 * 1024 * 1024

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
            # If we are reusing the main current Task, make sure we set its type to data_processing
            if str(task.task_type) != str(Task.TaskTypes.data_processing) and \
                    str(task.data.status) in ('created', 'in_progress'):
                task.set_task_type(task_type=Task.TaskTypes.data_processing)
                task.set_system_tags((task.get_system_tags() or []) + [self.__tag])
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

        # store current dataset Task
        self._task = task
        # store current dataset id
        self._id = task.id
        # store the folder where the dataset was downloaded to
        self._local_base_folder = None  # type: Optional[Path]
        # dirty flag, set True by any function call changing the dataset (regardless of weather it did anything)
        self._dirty = False
        self._using_current_task = False

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

    def upload(self, show_progress=True, verbose=False, output_url=None, compression=None):
        # type: (bool, bool, Optional[str], Optional[str]) -> ()
        """
        Start file uploading, the function returns when all files are uploaded.

        :param show_progress: If True show upload progress bar
        :param verbose: If True print verbose progress report
        :param output_url: Target storage for the compressed dataset (default: file server)
            Examples: `s3://bucket/data`, `gs://bucket/data` , `azure://bucket/data` , `/mnt/share/data`
        :param compression: Compression algorithm for the Zipped dataset file (default: ZIP_DEFLATED)
        """
        # set output_url
        if output_url:
            self._task.output_uri = output_url

        self._task.get_logger().report_text(
            'Uploading dataset files: {}'.format(
                dict(show_progress=show_progress, verbose=verbose, output_url=output_url, compression=compression)),
            print_console=False)

        fd, zip_file = mkstemp(
            prefix='dataset.{}.'.format(self._id), suffix='.zip'
        )
        archive_preview = ''
        count = 0
        try:
            with ZipFile(zip_file, 'w', allowZip64=True, compression=compression or ZIP_DEFLATED) as zf:
                for file_entry in self._dataset_file_entries.values():
                    if not file_entry.local_path:
                        # file is located in a different version
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
                    file_entry.local_path = None
                    count += 1
        except Exception as e:
            # failed uploading folder:
            LoggerRoot.get_base_logger().warning(
                'Exception {}\nFailed zipping dataset.'.format(e))
            return False
        finally:
            os.close(fd)

        zip_file = Path(zip_file)

        if not count:
            zip_file.unlink()
            LoggerRoot.get_base_logger().info('No pending files, skipping upload.')
            self._dirty = False
            self._serialize()
            return True

        archive_preview = 'Dataset archive content [{} files]:\n'.format(count) + archive_preview

        # noinspection PyBroadException
        try:
            # let's try to rename it
            new_zip_file = zip_file.parent / 'dataset.{}.zip'.format(self._id)
            zip_file.rename(new_zip_file)
            zip_file = new_zip_file
        except Exception:
            pass
        # remove files that could not be zipped, containing Null relative Path
        self._dataset_file_entries = {k: v for k, v in self._dataset_file_entries.items()
                                      if v.relative_path is not None}
        # start upload
        zip_file_size = format_size(Path(zip_file).stat().st_size)
        self._task.get_logger().report_text(
            'Uploading compressed dataset changes ({} files, total {}) to {}'.format(
                count, zip_file_size, self.get_default_storage()))
        self._task.upload_artifact(
            name=self.__data_entry_name, artifact_object=Path(zip_file), preview=archive_preview,
            delete_after_upload=True, wait_on_upload=True)
        self._task.get_logger().report_text('Upload completed ({})'.format(zip_file_size))

        self._add_script_call(
            'upload', show_progress=show_progress, verbose=verbose, output_url=output_url, compression=compression)

        self._dirty = False
        self._serialize()

    def finalize(self, verbose=False, raise_on_error=True):
        # type: (bool, bool) -> bool
        """
        Finalize the dataset publish dataset Task. upload must first called to verify there are not pending uploads.
        If files do need to be uploaded, it throws an exception (or return False)

        :param verbose: If True print verbose progress report
        :param raise_on_error: If True raise exception if dataset finalizing failed
        """
        # check we do not have files waiting for upload.
        if self._dirty:
            if raise_on_error:
                raise ValueError("Cannot finalize dataset, pending uploads. Call Dataset.upload(...)")
            return False

        status = self._task.get_status()
        if status not in ('in_progress', 'created'):
            raise ValueError("Cannot finalize dataset, status '{}' is not valid".format(status))

        self._task.get_logger().report_text('Finalizing dataset', print_console=False)

        # make sure we have no redundant parent versions
        self._serialize()
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
            self._task.completed()

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

    def get_local_copy(self, use_soft_links=None, raise_on_error=True):
        # type: (bool, bool) -> str
        """
        return a base folder with a read-only (immutable) local copy of the entire dataset
            download and copy / soft-link, files from all the parent dataset versions

        :param use_soft_links: If True use soft links, default False on windows True on Posix systems
        :param raise_on_error: If True raise exception if dataset merging failed on any file
        :return: A base folder for the entire dataset
        """
        assert self._id
        if not self._task:
            self._task = Task.get_task(task_id=self._id)

        # now let's merge the parents
        target_folder = self._merge_datasets(use_soft_links=use_soft_links, raise_on_error=raise_on_error)
        return target_folder

    def get_mutable_local_copy(self, target_folder, overwrite=False, raise_on_error=True):
        # type: (Union[Path, _Path, str], bool, bool) -> Optional[str]
        """
        return a base folder with a writable (mutable) local copy of the entire dataset
            download and copy / soft-link, files from all the parent dataset versions

        :param target_folder: Target folder for the writable copy
        :param overwrite: If True, recursively delete the target folder before creating a copy.
            If False (default) and target folder contains files, raise exception or return None
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

        ro_folder = self.get_local_copy(raise_on_error=raise_on_error)
        shutil.copytree(ro_folder, target_folder.as_posix())
        return target_folder.as_posix()

    def list_files(self, dataset_path=None, recursive=True, dataset_id=None):
        # type: (Optional[str], bool, Optional[str]) -> List[str]
        """
        returns a list of files in the current dataset
        If dataset_id is give, return a list of files that remained unchanged since the specified dataset_version

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
        return a list of files removed when comparing to a specific dataset_version

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
        # type: (...) -> Dataset
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
        parent_datasets = [cls.get(dataset_id=p) if isinstance(p, str) else p for p in (parent_datasets or [])]
        if any(not p.is_final() for p in parent_datasets):
            raise ValueError("Cannot inherit from a parent that was not finalized/closed")

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
            only_published=False  # type: bool
    ):
        # type: (...) -> Dataset
        """
        Get a specific Dataset. If only dataset_project is given, return the last Dataset in the Dataset project

        :param dataset_id: Requested Dataset ID
        :param dataset_project: Requested Dataset project name
        :param dataset_name: Requested Dataset name
        :param dataset_tags: Requested Dataset tags (list of tag strings)
        :param only_completed: Return only if the requested dataset is completed or published
        :param only_published: Return only if the requested dataset is published
        :return: Dataset object
        """
        mutually_exclusive(dataset_id=dataset_id, dataset_project=dataset_project, _require_at_least_one=False)
        mutually_exclusive(dataset_id=dataset_id, dataset_name=dataset_name, _require_at_least_one=False)
        if not any([dataset_id, dataset_project, dataset_name, dataset_tags]):
            raise ValueError('Dataset selection provided not provided (id/name/project/tags')

        tasks = Task.get_tasks(
            task_ids=[dataset_id] if dataset_id else None,
            project_name=dataset_project,
            task_name=exact_match_regex(dataset_name) if dataset_name else None,
            task_filter=dict(
                system_tags=[cls.__tag, '-archived'], order_by=['-created'],
                tags=dataset_tags,
                type=[str(Task.TaskTypes.data_processing)],
                page_size=1, page=0,
                status=['published'] if only_published else
                ['published', 'completed', 'closed'] if only_completed else None)
        )
        if not tasks:
            raise ValueError('Could not find Dataset {} {}'.format(
                'id' if dataset_id else 'project/name',
                dataset_id if dataset_id else (dataset_project, dataset_name)))
        task = tasks[0]
        if task.status == 'created':
            raise ValueError('Dataset id={} is in draft mode, delete and recreate it'.format(task.id))
        force_download = False if task.status in ('stopped', 'published', 'closed', 'completed') else True
        local_state_file = StorageManager.get_local_copy(
            remote_url=task.artifacts[cls.__state_entry_name].url, cache_context=cls.__cache_context,
            extract_archive=False, name=task.id, force_download=force_download)
        if not local_state_file:
            raise ValueError('Could not load Dataset id={} state'.format(task.id))

        instance = cls._deserialize(local_state_file, task)
        # remove the artifact, just in case
        if force_download:
            os.unlink(local_state_file)

        return instance

    def get_logger(self):
        # type: () -> Logger
        """
        Return a Logger object for the Dataset, allowing users to report statistics metrics
        and debug samples on the Dataset itself
        :return: Logger object
        """
        return self._task.get_logger()

    @classmethod
    def squash(cls, dataset_name, dataset_ids=None, dataset_project_name_pairs=None, output_url=None):
        # type: (str, Optional[Sequence[Union[str, Dataset]]],Optional[Sequence[(str, str)]], Optional[str]) -> Dataset
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
        current_parents = self._dependency_graph.get(self._id)
        # remove parent versions we no longer need from the main version list
        # per version, remove unnecessary parent versions, if we do not need them
        self._dependency_graph = {k: [p for p in parents if p in used_dataset_versions]
                                  for k, parents in self._dependency_graph.items() if k in used_dataset_versions}
        # make sure we do not remove our parents, for geology sake
        self._dependency_graph[self._id] = current_parents

    def _serialize(self):
        """
        store current state of the Dataset for later use
        :return: object to be used for later deserialization
        """
        self._update_dependency_graph()

        state = dict(
            dataset_file_entries=[f.as_dict() for f in self._dataset_file_entries.values()],
            dependency_graph=self._dependency_graph,
            id=self._id,
            dirty=self._dirty,
        )
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

    def _download_dataset_archive(self):
        """
        Download the dataset archive, return a link to locally stored zip file
        :return: Path to locally stored zip file
        """
        pass  # TODO: implement

    def _extract_dataset_archive(self):
        """
        Download the dataset archive, and extract the zip content to a cached folder.
        Notice no merging is done.

        :return: Path to a local storage extracted archive
        """
        if not self._task:
            self._task = Task.get_task(task_id=self._id)
        # check if we have a dataset with empty change set
        if not self._task.artifacts.get(self.__data_entry_name):
            cache = CacheManager.get_cache_manager(cache_context=self.__cache_context)
            local_folder = Path(cache.get_cache_folder()) / self._get_cache_folder_name()
            local_folder.mkdir(parents=True, exist_ok=True)
            return local_folder.as_posix()

        # download the dataset zip
        local_zip = StorageManager.get_local_copy(
            remote_url=self._task.artifacts[self.__data_entry_name].url, cache_context=self.__cache_context,
            extract_archive=False, name=self._id)
        if not local_zip:
            raise ValueError("Could not download dataset id={}".format(self._id))
        local_folder = (Path(local_zip).parent / self._get_cache_folder_name()).as_posix()
        # if we got here, we need to clear the target folder
        shutil.rmtree(local_folder, ignore_errors=True)
        # noinspection PyProtectedMember
        local_folder = StorageManager._extract_to_cache(
            cached_file=local_zip, name=self._id,
            cache_context=self.__cache_context, target_folder=local_folder)
        return local_folder

    def _merge_datasets(self, use_soft_links=None, raise_on_error=True):
        # type: (bool, bool) -> str
        """
        download and copy / soft-link, files from all the parent dataset versions
        :param use_soft_links: If True use soft links, default False on windows True on Posix systems
        :param raise_on_error: If True raise exception if dataset merging failed on any file
        :return: the target folder
        """
        if use_soft_links is None:
            use_soft_links = False if is_windows() else True

        # check if we already have everything
        target_base_folder, target_base_size = CacheManager.get_cache_manager(
            cache_context=self.__cache_context).get_cache_file(local_filename=self._get_cache_folder_name())
        if target_base_folder and target_base_size is not None:
            target_base_folder = Path(target_base_folder)
            # check dataset file size, if we have a full match no need for parent dataset download / merge
            verified = True
            # noinspection PyBroadException
            try:
                for f in self._dataset_file_entries.values():
                    if (target_base_folder / f.relative_path).stat().st_size != f.size:
                        verified = False
                        break
            except Exception:
                verified = False

            if verified:
                return target_base_folder.as_posix()
            else:
                LoggerRoot.get_base_logger().info('Dataset needs refreshing, fetching all parent datasets')

        # first get our dataset
        target_base_folder = Path(self._extract_dataset_archive())
        target_base_folder.touch()

        # create thread pool
        pool = ThreadPool(cpu_count() * 2)
        for dataset_version_id in self._get_dependencies_by_order():
            # make sure we skip over empty dependencies
            if dataset_version_id not in self._dependency_graph:
                continue

            ds = Dataset.get(dataset_id=dataset_version_id)
            ds_base_folder = Path(ds._extract_dataset_archive())
            ds_base_folder.touch()

            def copy_file(file_entry):
                if file_entry.parent_dataset_id != dataset_version_id:
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
            if raise_on_error and any(errors):
                raise ValueError("Dataset merging failed: {}".format([e for e in errors if e is not None]))

        pool.close()
        return target_base_folder.absolute().as_posix()

    def _get_dependencies_by_order(self, include_unused=False):
        # type: (bool) -> List[str]
        """
        Return the dataset dependencies by order of application (from the last to the current)
        :param bool include_unused: If True include unused datasets in the dependencies
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
                    [p for p in self.get(dataset_id=r)._get_parents() if p not in roots])))
            else:
                roots.extend(list(reversed(
                    [p for p in self._dependency_graph.get(r, [])
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
                        [p for p in self.get(dataset_id=r)._get_parents() if p not in roots])))
                else:
                    roots.extend(list(reversed(
                        [p for p in self._dependency_graph.get(r, [])
                         if p not in roots and (include_unused or (p in self._dependency_graph))])))

        # skip our id
        return list(reversed(dependencies[1:]))

    def _get_parents(self):
        # type: () -> Sequence[str]
        """
        Return a list of direct parent datasets (str)
        :return: list of dataset ids
        """
        return self._dependency_graph[self.id]

    @classmethod
    def _deserialize(cls, stored_state, task):
        # type: (Union[dict, str, Path, _Path], Task) -> Dataset
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
        instance._dependency_graph = stored_state['dependency_graph']
        instance._dirty = stored_state.get('dirty', False)
        instance._dataset_file_entries = {
            s['relative_path']: FileEntry(**s) for s in stored_state['dataset_file_entries']}
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

    def _get_cache_folder_name(self):
        return '{}{}'.format(self.__cache_folder_prefix, self._id)

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
        nodes = self._get_dependencies_by_order(include_unused=True) + [self.id]
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
        if len(self._dataset_file_entries) < self.__preview_max_file_entries:
            file_entries = sorted(self._dataset_file_entries.values(), key=lambda x: x.relative_path)
            dataset_details = \
                'File Name - File Size - Hash (SHA2)\n' +\
                '\n'.join('{} - {} - {}'.format(f.relative_path, f.size, f.hash) for f in file_entries)
        # too large to store
        if not dataset_details or len(dataset_details) > self.__preview_max_size:
            dataset_details = 'Dataset content is too large to preview'

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
