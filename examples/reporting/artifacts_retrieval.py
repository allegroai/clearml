# ClearML - example code, retrieve other task artifacts and print the artifacts
# Please run examples/reporting/artifacts.py example before running this example
#
from pprint import pprint

from clearml import Task


def main():
    # Getting the task we want to get the artifacts from
    artifacts_task = Task.get_task(project_name='ClearML Examples', task_name='artifacts example')

    # getting the numpy object back
    numpy_artifact = artifacts_task.artifacts['Numpy Eye'].get()
    print("numpy_artifact is:\n{}\n".format(numpy_artifact))

    # download the numpy object as a npz file
    download_numpy_artifact = artifacts_task.artifacts['Numpy Eye'].get_local_copy()
    print("download_numpy_artifact path is:\n{}\n".format(download_numpy_artifact))

    # getting the PIL Image object
    pil_artifact = artifacts_task.artifacts['pillow_image'].get()
    print("pil_artifact is:\n{}\n".format(pil_artifact))

    # getting the pandas object
    pandas_artifact = artifacts_task.artifacts['Pandas'].get()
    print("pandas_artifact is:\n{}\n".format(pandas_artifact))

    # getting the dictionary object
    dictionary_artifact = artifacts_task.artifacts['dictionary'].get()
    print("dictionary_artifact is:\n")
    pprint(dictionary_artifact)

    # getting the train DataFrame
    df_artifact = artifacts_task.artifacts['train'].get()
    print("df_artifact is:\n{}\n".format(df_artifact))

    # download the train DataFrame csv in the same format as in the UI (gz file)
    df_artifact_as_gz = artifacts_task.artifacts['train'].get_local_copy()
    print("df_artifact_as_gz path is:\n{}\n".format(df_artifact_as_gz))

    # download the wildcard jpegs images (getting the zip file already extracted into a cached folder),
    # the path containing those will be returned
    jpegs_artifact = artifacts_task.artifacts['wildcard jpegs'].get()
    print("jpegs_artifact path is:\n{}\n".format(jpegs_artifact))

    # download the local folder that was uploaded (getting the zip file already extracted into a cached folder),
    # the path containing those will be returned
    local_folder_artifact = artifacts_task.artifacts['local folder'].get()
    print("local_folder_artifact path is:\n{}\n".format(local_folder_artifact))

    # download the local folder that was uploaded (getting the zip file without extracting it),
    # the path containing the zip file will be returned
    local_folder_artifact_as_zip = artifacts_task.artifacts['local folder'].get_local_copy(extract_archive=False)
    print("local_folder_artifact_as_zip path is:\n{}\n".format(local_folder_artifact_as_zip))

    # download the local file that was uploaded (getting the zip file already extracted into a cached folder),
    # the path containing this file will be returned
    local_file_artifact = artifacts_task.artifacts['local file'].get()
    print("local_file_artifact path is:\n{}\n".format(local_file_artifact))


if __name__ == '__main__':
    main()
