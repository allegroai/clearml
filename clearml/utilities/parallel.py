import os

from copy import deepcopy
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Union, Generator, List, Type
from zipfile import ZipFile, ZIP_DEFLATED
from six.moves.queue import PriorityQueue, Queue, Empty
from pathlib2 import Path
from tempfile import mkstemp
from collections import deque
from threading import Thread

from ..debugging.log import LoggerRoot
from ..storage.util import format_size


class _DeferredClass(object):
    __slots__ = ('__queue', '__future_caller', '__future_func')

    def __init__(self, a_future_caller, future_func):
        self.__queue = Queue()
        self.__future_caller = a_future_caller
        self.__future_func = future_func

    def __nested_caller(self, item, args, kwargs):
        # wait until object is constructed
        getattr(self.__future_caller, "id")  # noqa

        future_func = getattr(self.__future_caller, self.__future_func)
        the_object = future_func()
        the_object_func = getattr(the_object, item)
        return the_object_func(*args, **kwargs)

    def _flush_into_logger(self, a_future_object=None, a_future_func=None):
        self.__close_queue(a_future_object=a_future_object, a_future_func=a_future_func)

    def __close_queue(self, a_future_object=None, a_future_func=None):
        # call this function when we Know the object is initialization is completed
        if self.__queue is None:
            return

        _queue = self.__queue
        self.__queue = None
        while True:
            # noinspection PyBroadException
            try:
                item, args, kwargs = _queue.get(block=False)
                if a_future_object:
                    future_func = getattr(a_future_object, self.__future_func)
                    the_object = future_func()
                    the_object_func = getattr(the_object, item)
                    the_object_func(*args, **kwargs)
                elif a_future_func:
                    the_object_func = getattr(a_future_func, item)
                    the_object_func(*args, **kwargs)
                else:
                    self.__nested_caller(item, args, kwargs)
            except Empty:
                break
            except Exception:
                # import traceback
                # stdout_print(''.join(traceback.format_exc()))
                pass

    def __getattr__(self, item):
        def _caller(*args, **kwargs):
            # if we already completed the background initialization, call functions immediately
            # noinspection PyProtectedMember
            if not self.__queue or self.__future_caller._FutureTaskCaller__executor is None:
                return self.__nested_caller(item, args, kwargs)

            # noinspection PyBroadException
            try:
                # if pool is still active call async
                self.__queue.put((item, deepcopy(args) if args else args, deepcopy(kwargs) if kwargs else kwargs))
            except Exception:
                # assume we wait only if self.__pool was nulled between the if and now, so just call directly
                return self.__nested_caller(item, args, kwargs)

            # let's hope it is the right one
            return True

        return _caller


class FutureTaskCaller(object):
    """
    FutureTaskCaller is used to create a class via a functions async, in another thread.

    For example:

    .. code-block:: py

        future = FutureTaskCaller().call(func=max, func_cb=None, override_cls=None, 1, 2)
        print('Running other code')
        print(future.result())  # will print '2'
    """
    __slots__ = ('__object', '__object_cls', '__executor', '__deferred_bkg_class')

    @property
    def __class__(self):
        return self.__object_cls

    def __init__(self, func, func_cb, override_cls, *args, **kwargs):
        # type: (Callable, Optional[Callable], Type, *Any, **Any) -> None
        """
        __init__(*args, **kwargs) in another thread

        :return: This FutureTaskCaller instance
        """
        self.__object = None
        self.__object_cls = override_cls
        self.__deferred_bkg_class = _DeferredClass(self, "get_logger")

        self.__executor = Thread(target=self.__submit__, args=(func, func_cb, args, kwargs))
        self.__executor.daemon = True
        self.__executor.start()

    def __submit__(self, fn, fn_cb, args, kwargs):
        # background initialization call
        _object = fn(*args, **kwargs)

        # push all background calls (now that the initialization is complete)
        if self.__deferred_bkg_class:
            _deferred_bkg_class = self.__deferred_bkg_class
            self.__deferred_bkg_class = None
            # noinspection PyProtectedMember
            _deferred_bkg_class._flush_into_logger(a_future_object=_object)

        # store the initialized object
        self.__object = _object
        # callback function
        if fn_cb is not None:
            fn_cb(self.__object)

    def __getattr__(self, item):
        # if we get here, by definition this is not a __slot__ entry, pass to the object
        return getattr(self.__result__(), item)

    def __setattr__(self, item, value):
        # make sure we can set the slots
        if item in ["_FutureTaskCaller__executor", "_FutureTaskCaller__object",
                    "_FutureTaskCaller__object_cls", "_FutureTaskCaller__deferred_bkg_class"]:
            return super(FutureTaskCaller, self).__setattr__(item, value)

        setattr(self.__result__(), item, value)

    def __result__(self, timeout=None):
        # type: (Optional[float]) -> Any
        """
        Wait and get the result of the function called with self.call()

        :param timeout: The maximum number of seconds to wait for the result. If None,
            there is no limit for the wait time.

        :return: The result of the called function
        """
        if self.__executor:
            # since the test is not atomic, we assume that if we failed joining
            # it is because someone else joined before us
            # noinspection PyBroadException
            try:
                self.__executor.join(timeout=timeout)
            except RuntimeError:
                # this is probably calling ourselves from the same thread
                raise
            except Exception:
                # wait until that someone else updated the __object
                while self.__object is None:
                    sleep(1)
            self.__executor = None
        return self.__object

    # This is the part where we are no longer generic, but __slots__
    # inheritance is too cumbersome to actually inherit and make sure it works optimally
    def get_logger(self):
        if self.__object is not None:
            return self.__object.get_logger()

        if self.__deferred_bkg_class is None:
            # we are shutting down, wait until object is available
            return self.__result__().get_logger()

        return self.__deferred_bkg_class


class ParallelZipper(object):
    """
    Used to zip multiple files in zip chunks of a particular size, all in parallel
    """
    class ZipperObject(object):
        def __init__(
            self,
            chunk_size,  # int
            zipper_queue,  # PriorityQueue[ParallelZipper.ZipperObject]
            zipper_results,  # Queue[ParallelZipper.ZipperObject]
            allow_zip_64,  # bool
            compression,  # Any
            zip_prefix,  # str
            zip_suffix,  # str
            verbose,  # bool
            task,  # Any
        ):
            # (...) -> ParallelZipper.ZipperObject
            """
            Initialize a ParallelZipper.ZipperObject instance that holds its corresponding zip
            file, as well as other relevant data

            :param chunk_size: Chunk size, in MB. The ParallelZipper will try its best not to exceed this size
                when bulding up this zipper object, but that is not guaranteed
            :param zipper_queue: PriorityQueue that holds ParallelZipper.ZipperObject instances.
                When this ParallelZipper.ZipperObject can hold more data (i.e. chunk_size was not exceeded),
                this object will reinsert itself in this queue to be reused by the ParallelZipper.
                Else, a fresh ParallelZipper.ZipperObject will be inserted
            :param zipper_results: Queue that holds ParallelZipper.ZipperObject instances. These instances
                are added to this queue when chunk_size is exceeded
            :param allow_zip_64: if True ZipFile will create files with ZIP64 extensions when
                needed, otherwise it will raise an exception when this would be necessary
            :param compression: ZipFile.ZIP_STORED (no compression), ZipFile.ZIP_DEFLATED (requires zlib),
                ZipFile.ZIP_BZIP2 (requires bz2) or ZipFile.ZIP_LZMA (requires lzma).
            :param zip_prefix: The zip file created by this object will have its name prefixed by this
            :param zip_suffix: The zip file created by this object will have its name suffixed by this
            :param verbose: If True, print data relevant to the file compression
            :param task: ClearML Task, used for logging

            :return: ParallelZipper.ZipperObject instance
            """
            self._chunk_size = chunk_size
            self._zipper_queue = zipper_queue
            self._zipper_results = zipper_results
            self._allow_zip_64 = allow_zip_64
            self._compression = compression
            self._zip_prefix = zip_prefix
            self._zip_suffix = zip_suffix
            self._verbose = verbose
            self._task = task
            self.fd, self.zip_path = mkstemp(prefix=zip_prefix, suffix=zip_suffix)
            self.zip_path = Path(self.zip_path)
            self.zip_file = ZipFile(self.zip_path.as_posix(), "w", allowZip64=allow_zip_64, compression=compression)
            self.archive_preview = ""
            self.count = 0
            self.files_zipped = set()

        def zip(self, file_path, arcname=None):
            # type: (Union[str, Path], str) -> ()
            """
            Zips a file into the ZipFile created by this instance. This instance will either add
            itself back to the PriorityQueue used to select the best zipping candidate or add itself
            to the result Queue after exceeding self.chunk_size.

            :param file_path: Path to the file to be zipped
            :param arcname: Name of the file in the archive
            """
            if self._verbose and self._task:
                self._task.get_logger().report_text("Compressing {}".format(Path(file_path).as_posix()))
            self.zip_file.write(file_path, arcname=arcname)
            self.count += 1
            preview_path = arcname
            if not preview_path:
                preview_path = file_path
            self.archive_preview += "{} - {}\n".format(preview_path, format_size(self.size))
            self.files_zipped.add(Path(file_path).as_posix())
            if self._chunk_size <= 0 or self.size < self._chunk_size:
                self._zipper_queue.put(self)
            else:
                self._zipper_queue.put(
                    ParallelZipper.ZipperObject(
                        self._chunk_size,
                        self._zipper_queue,
                        self._zipper_results,
                        self._allow_zip_64,
                        self._compression,
                        self._zip_prefix,
                        self._zip_suffix,
                        self._verbose,
                        self._task,
                    )
                )
                self._zipper_results.put(self)

        def merge(self, other):
            # type: (ParallelZipper.ZipperObject) -> ()
            """
            Merges one ParallelZipper.ZipperObject instance into the current one.
            All the files zipped by the other instance will be added to this instance,
            as well as any other useful additional data

            :param other: ParallelZipper.ZipperObject instance to merge into this one
            """
            with ZipFile(self.zip_path.as_posix(), "a") as parent_zip:
                with ZipFile(other.zip_path.as_posix(), "r") as child_zip:
                    for child_name in child_zip.namelist():
                        parent_zip.writestr(child_name, child_zip.open(child_name).read())
            self.files_zipped |= other.files_zipped
            self.count += other.count
            self.archive_preview += other.archive_preview

        def close(self):
            # type: () -> ()
            """
            Attempts to close file descriptors associated to the ZipFile
            """
            # noinspection PyBroadException
            try:
                self.zip_file.close()
                os.close(self.fd)
            except Exception:
                pass

        def delete(self):
            # type: () -> ()
            """
            Attempts to delete the ZipFile from the disk
            """
            # noinspection PyBroadException
            try:
                self.close()
                self.zip_path.unlink()
            except Exception:
                pass

        @property
        def size(self):
            # type: () -> ()
            """
            :return: Size of the ZipFile, in bytes
            """
            return self.zip_path.stat().st_size

        def __lt__(self, other):
            # we want to completely "fill" as many zip files as possible, hence the ">" comparison
            return self.size > other.size

    def __init__(
        self,
        chunk_size,  # type: int
        max_workers,  # type: int
        allow_zip_64=True,  # type: Optional[bool]
        compression=ZIP_DEFLATED,  # type: Optional[Any]
        zip_prefix="",  # type: Optional[str]
        zip_suffix="",  # type: Optional[str]
        verbose=False,  # type: Optional[bool]
        task=None,  # type: Optional[Any]
        pool=None,  # type: Optional[ThreadPoolExecutor]
    ):
        # type: (...) -> ParallelZipper
        """
        Initialize the ParallelZipper. Each zip created by this object will have the following naming
        format: [zip_prefix]<random_string>[zip_suffix]

        :param chunk_size: Chunk size, in MB. The ParallelZipper will try its best not to exceed this size,
            but that is not guaranteed
        :param max_workers: The maximum number of workers spawned when zipping the files
        :param allow_zip_64: if True ZipFile will create files with ZIP64 extensions when
            needed, otherwise it will raise an exception when this would be necessary
        :param compression: ZipFile.ZIP_STORED (no compression), ZipFile.ZIP_DEFLATED (requires zlib),
            ZipFile.ZIP_BZIP2 (requires bz2) or ZipFile.ZIP_LZMA (requires lzma).
        :param zip_prefix: Zip file names will be prefixed by this
        :param zip_suffix: Zip file names will pe suffixed by this
        :param verbose: If True, print data relevant to the file compression
        :param task: ClearML Task, used for logging
        :param pool: Use this ThreadPoolExecutor instead of creating one. Note that this pool will not be
            closed after zipping is finished.

        :return: ParallelZipper instance
        """
        self._chunk_size = chunk_size * (1024 ** 2)
        self._max_workers = max_workers
        self._allow_zip_64 = allow_zip_64
        self._compression = compression
        self._zip_prefix = zip_prefix
        self._zip_suffix = zip_suffix
        self._verbose = verbose
        self._task = task
        self._pool = pool
        self._zipper_queue = PriorityQueue()
        self._zipper_results = Queue()

    def zip_iter(self, file_paths, arcnames={}):
        # type: (List[Union(str, Path)], Optional[dict[Union(str, Path), str]]) -> Generator[ParallelZipper.ZipperObject]
        """
        Generator function that returns zip files as soon as they are available.
        The zipping is done in parallel

        :param file_paths: List of paths to the files to zip
        :param arcnames: Dictionary that maps the file path to what should be its name in the archive.

        :return: Generator of ParallelZipper.ZipperObjects
        """
        while not self._zipper_queue.empty():
            self._zipper_queue.get_nowait()
        for _ in range(self._max_workers):
            self._zipper_queue.put(
                ParallelZipper.ZipperObject(
                    self._chunk_size,
                    self._zipper_queue,
                    self._zipper_results,
                    self._allow_zip_64,
                    self._compression,
                    self._zip_prefix,
                    self._zip_suffix,
                    self._verbose,
                    self._task,
                )
            )
        filtered_file_paths = []
        for file_path in file_paths:
            if not Path(file_path).is_file():
                LoggerRoot.get_base_logger().warning("Could not store dataset file {}. File skipped".format(file_path))
            else:
                filtered_file_paths.append(file_path)
        file_paths = filtered_file_paths

        file_paths = sorted(file_paths, key=lambda k: Path(k).stat().st_size, reverse=True)
        # zip in parallel
        pooled = []
        if not self._pool:
            pool = ThreadPoolExecutor(max_workers=self._max_workers)
        else:
            pool = self._pool
        for f in file_paths:
            zipper = self._zipper_queue.get()
            pooled.append(pool.submit(zipper.zip, Path(f).as_posix(), arcname=arcnames.get(f)))
            for result in self._yield_zipper_results():
                yield result
        for task in pooled:
            task.result()
        if not self._pool:
            pool.close()

        for result in self._yield_zipper_results():
            yield result

        zipper_results_leftover = []

        # extract remaining results
        while not self._zipper_queue.empty():
            result = self._zipper_queue.get()
            if result.count != 0:
                zipper_results_leftover.append(result)
            else:
                result.delete()
        zipper_results_leftover = deque(sorted(zipper_results_leftover, reverse=True))

        # merge zip files greedily if possible and get the paths as results
        while len(zipper_results_leftover) > 0:
            zip_ = zipper_results_leftover.pop()
            zip_.close()
            if zip_.size >= self._chunk_size > 0:
                yield zip_
                continue
            while len(zipper_results_leftover) > 0 and (
                self._chunk_size <= 0 or zipper_results_leftover[0].size + zip_.size < self._chunk_size
            ):
                child_zip = zipper_results_leftover.popleft()
                child_zip.close()
                zip_.merge(child_zip)
                child_zip.delete()
            yield zip_

    def _yield_zipper_results(self):
        while True:
            try:
                result = self._zipper_results.get_nowait()
                result.close()
                yield result
            except Empty:
                break
