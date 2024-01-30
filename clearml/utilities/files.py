import os
from sys import platform

import pathlib2
import psutil


def get_filename_max_length(dir_path):
    # type: (str) -> int
    try:
        dir_path = pathlib2.Path(os.path.abspath(dir_path))
        if platform == "win32":
            dir_drive = dir_path.drive
            for drv in psutil.disk_partitions():
                if drv.device.startswith(dir_drive):
                    return drv.maxfile
        elif platform in ("linux", "darwin"):
            return os.statvfs(dir_path).f_namemax
    except Exception as err:
        print(err)

    return 255  # Common filesystems like NTFS, EXT4 and HFS+ limited with 255


def is_path_traversal(target_folder, relative_path):
    try:
        target_folder = pathlib2.Path(target_folder)
        relative_path = pathlib2.Path(relative_path)
        # returns the relative path starting from the target_folder,
        # or raise an ValueError if a directory traversal attack is tried
        target_folder.joinpath(relative_path).resolve().relative_to(target_folder.resolve())
        return False
    except ValueError:
        return True
