"""
How to register data with masks from a json file.
Create a list of masks for each image and add to a DatasetVersion.
Define DatasetVersion-level mask-label mapping, which maps RGB values from the mask to class labels.

Notice: This is a custom parser for a specific dataset. Each dataset requires a different parser.

You can run this example from this dir with:

python register_dataset_masks.py
--ext jpg --ds_name my_uploaded_dataset --version_name my_version
"""

import glob
import json
import os
from argparse import ArgumentParser

from allegroai import DatasetVersion, FrameGroup, SingleFrame, Task
from clearml import StorageManager

def get_frames_with_masks(data_path, ext="png", mask_ext="_mask.png"):
    frame_groups = {}

    # Go over each jpg file in base path
    for file in glob.glob(os.path.join(data_path, "*.{}".format(ext))):
        full_path = os.path.abspath(file)

        # if this is a mask file skip it, we will manually add it later to the images it belongs to
        if full_path.endswith(mask_ext):
            continue

        # let's check if we have a mask file
        full_path_mask = full_path.replace(f".{ext}", mask_ext)
        if not os.path.exists(full_path_mask):
            # we do not have a mask file, so let's skip this one
            continue

        # now we need to add the actual
        print("Getting files from: " + full_path)

        # let's split the file name based on '_' and use the first part as ID
        file_parts_key = os.path.split(full_path)[-1].split("_")

        # this is used just so we can easily collect (group) the frames together
        frame_group_id = file_parts_key[0]
        # find the correct FrameGroup based on the filename
        if frame_group_id not in frame_groups:
            # this is acts like a Dict and the keys are string and the values are SingleFrames
            frame_group = FrameGroup()
            frame_groups[frame_group_id] = frame_group
        else:
            frame_group = frame_groups[frame_group_id]

        # add the frame and the mask to the frame group,
        # we have to give it a name (inside the FrameGroup) so we use
        source_id = file_parts_key[1]
        frame_group[source_id] = SingleFrame(source=full_path, mask_source=full_path_mask)

    # return a list of FrameGroups
    return list(frame_groups.values())


def read_mask_class_values(local_dataset_path):
    json_file_path = os.path.join(local_dataset_path, "_mask_legend.json")

    json_file = open(json_file_path, "r")
    data = json.load(json_file)
    json_file.close()

    # now we need to convert it to pixel RGB value mapping to classes
    label_mapping = {tuple(value): [key] for key, value in data.items()}

    return label_mapping


def create_version_with_frames(new_frames, masks_lookup, ds_name, ver_name, local_dataset_path):

    # Get the dataset (it will create a new one if we don't have it)
    ds = DatasetVersion.create_new_dataset(dataset_name=ds_name)

    # create a specific dataset version, or just use the latest version
    dv = ds.create_version(version_name=ver_name) if ver_name else \
        DatasetVersion.get_current(dataset_name=ds_name)

    dv.set_masks_labels(masks_lookup)

    # Add and upload frames to created version
    dv.add_frames(
        new_frames,
        # where to upload the files, we will use for example the default one, you can also s3://bucket/ etc
        auto_upload_destination=Task.current_task().get_output_destination(),
        # The local root, this will make sure we keep the same
        # files structure in the upload destination as we have on the local machine
        local_dataset_root_path=local_dataset_path
    )
    dv.commit_version()


if __name__ == '__main__':
    parser = ArgumentParser(description='Register allegro dataset with frame group and masks')

    parser.add_argument(
        '--ext', type=str, help='Files extension to upload from the dir. Default "png"',
        default="png")
    parser.add_argument(
        '--mask-ext', type=str, help='Files extension to upload from the dir. Default "_mask.png"',
        default="_mask.png")

    parser.add_argument(
        '--ds_name', type=str, help='Dataset name for the data',
        default="sample-dataset-masks")
    parser.add_argument(
        '--version_name', type=str, help='Version name for the data (default is current version)',
        default="initial")

    args = parser.parse_args()

    example_dataset_path = 's3://clearml-public/datasets/hyperdataset_example/ds_with_masks'
    local_img_path = StorageManager.download_folder(example_dataset_path)
    # this folder contains the images and json files for the data
    base_path = os.path.abspath('{}/datasets/hyperdataset_example/ds_with_masks'.format(local_img_path))
    dataset_name = args.ds_name
    version_name = args.version_name

    task = Task.init(
        project_name="uploading_datasets", task_name="upload_sample_dataset_with_masks",
        task_type=Task.TaskTypes.data_processing,
        # This will make sure we have a valid output destination for our local files to be uploaded to
        output_uri=True
    )

    frames = get_frames_with_masks(data_path=base_path, ext=args.ext, mask_ext=args.mask_ext)
    mask_class_lookup = read_mask_class_values(base_path)
    create_version_with_frames(frames, mask_class_lookup, dataset_name, version_name, base_path)

    print("We are done :)")
