# Using artifacts example
"""
Upload artifacts from a Task, and then a different Task can access and utilize the data from that artifact.
"""
from clearml import Task
from time import sleep


task1 = Task.init(project_name='examples', task_name='create artifact')
# upload data file to the initialized task, inputting a name and file location
task1.upload_artifact(name='data file', artifact_object='data_samples/sample.json')
# close the task, to be able to initialize a new task
task1.close()

# initialize another task to use some other task's artifacts
task2 = Task.init(project_name='examples', task_name='use artifact from other task')
# get instance of Task that created artifact (task1), using Task's project and name. You could also use its ID number.
preprocess_task = Task.get_task(project_name='examples', task_name='create artifact')
# access artifact from task1, using the artifact's name
# get_local_copy() caches the files for later use and returns a path to the cached file
local_json = preprocess_task.artifacts['data file'].get_local_copy()

# Doing some stuff with file from other Task in current Task
with open(local_json) as data_file:
    file_text = data_file.read()

print(file_text)
# Simulate the work of a Task
sleep(1.0)
print('Finished doing stuff with some data :)')
