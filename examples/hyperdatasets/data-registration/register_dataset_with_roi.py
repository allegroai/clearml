"""
How to register data with ROIs and metadata from a json file.
Create a list of ROI's for each image in the metadata format required by a frame.

Notice: This is a custom parser for a specific dataset. Each dataset requires a different parser.

You can run this example from this dir with:

python register_dataset_with_roi.py
--ext jpg --ds_name my_uploaded_dataset --version_name my_version
"""

import glob
import json
import os
from allegroai import DatasetVersion, SingleFrame, Task
from argparse import ArgumentParser
from clearml import StorageManager

def get_json_file(filename):
    """
    Get the data from the json file

    :param filename: Full file path
    :type filename: str
    :return: json data parse as python dictionary
    """
    json_file_path = filename.replace('.jpg', '.json')

    json_file = open(json_file_path, "r")
    data = json.load(json_file)
    json_file.close()

    return data


def get_frames_with_roi_meta(data_path, ext):
    """
    Create a ready to register list of SingleFrame(s)

    :param data_path: Path to the folder you like to register
    :type data_path: str
    :param ext: Files extension to upload from the dir
    :type ext: str
    :return: List[SingleFrame] list contains all the SingleFrame that should be register
    """
    frames_to_reg = []
    # Go over each jpg file in base path
    for file in glob.glob(os.path.join(data_path, "*.{}".format(ext))):
        full_path = os.path.abspath(file)
        print("Getting files from: " + full_path)
        # read the json file next to the image
        data = get_json_file(full_path)

        # Create the SingleFrame object
        a_frame = SingleFrame(source=full_path)

        # Iterating over rois in the json, and add them
        for roi in data['rois']:
            a_frame.add_annotation(
                poly2d_xy=roi["poly"],
                labels=roi['labels'],
                metadata={'alive': roi['meta']['alive']},
                confidence=roi['confidence']
            )
        # add generic meta-data to the frame
        a_frame.width = data['size']['x']
        a_frame.height = data['size']['y']
        a_frame.metadata['dangerous'] = data['meta']['dangerous']

        # add to our SingleFrame Collection
        frames_to_reg.append(a_frame)
    return frames_to_reg


def create_version_with_frames(new_frames, ds_name, ver_name, local_dataset_path):
    """
    Create a DatasetVersion with new_frames as frames

    :param new_frames: list with all the frames to be registered
    :type new_frames: List[SingleFrame]
    :param ds_name: The dataset name
    :type ds_name: str
    :param ver_name: The version name
    :type ver_name: str
    :param local_dataset_path: Path to the folder you register
    :param local_dataset_path: str
    """
    # Get the dataset (it will create a new one if we don't have it)
    ds = DatasetVersion.create_new_dataset(dataset_name=ds_name)

    # create a specific dataset version, or just use the latest version
    dv = ds.create_version(version_name=ver_name) if ver_name else \
        DatasetVersion.get_current(dataset_name=ds_name)

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
    parser = ArgumentParser(description='Register allegro dataset with rois and meta')

    parser.add_argument(
        '--ext', type=str, help='Files extension to upload from the dir. Default: "jpg"',
        default="jpg")
    parser.add_argument(
        '--ds_name', type=str, help='Dataset name for the data. Default: "sample-dataset"',
        default="sample-dataset")
    parser.add_argument(
        '--version_name', type=str, help='Version name for the data (default is current version)',
        default="initial")

    args = parser.parse_args()

    # this folder contains the images and json files for the data
    example_dataset_path = 's3://clearml-public/datasets/hyperdataset_example/ds_with_rois'
    local_img_path = StorageManager.download_folder(example_dataset_path)

    # this folder contains the images and json files for the data
    base_path = os.path.abspath('{}/datasets/hyperdataset_example/ds_with_rois'.format(local_img_path))
    dataset_name = args.ds_name
    version_name = args.version_name

    task = Task.init(
        project_name="uploading_datasets", task_name="upload_sample",
        task_type=Task.TaskTypes.data_processing,
        # This will make sure we have a valid output destination for our local files to be uploaded to. This support
        # also other storage types:
        #             - A shared folder: ``/mnt/share/folder``
        #             - S3: ``s3://bucket/folder``
        #             - Google Cloud Storage: ``gs://bucket-name/folder``
        #             - Azure Storage: ``azure://company.blob.core.windows.net/folder/``
        output_uri=True
    )

    frames = get_frames_with_roi_meta(base_path, args.ext)

    create_version_with_frames(frames, dataset_name, version_name, base_path)

    print("We are done :)")
