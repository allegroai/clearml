from clearml import Task, StorageManager

# create an dataset experiment
task = Task.init(project_name="examples", task_name="pipeline step 1 dataset artifact")

# only create the task, we will actually execute it later
task.execute_remotely()

# simulate local dataset, download one, so we have something local
local_iris_pkl = StorageManager.get_local_copy(
    remote_url='https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl')

# add and upload local file containing our toy dataset
task.upload_artifact('dataset', artifact_object=local_iris_pkl)

print('uploading artifacts in the background')

# we are done
print('Done')
