# Handling the Google Cloud Storage breaking change

## Rationale

Due to an issue with ClearML SDK versions 1.11.x, URLs of objects uploaded to the Google Cloud Storage were stored in the ClearML backend as a quoted string. This behavior causes issues accessing registered objects using the ClearML SDK. The issue affects the URLs of models, datasets, artifacts, and media files/debug samples. In case you have such objects uploaded with the affected ClearML SDK versions and wish to be able to access them programmatically using the ClearML SDK using version 1.12 and above (note that access from the ClearML UI is still possible), you should perform the actions listed in the section below.

## Recommended Steps

The code snippets below should serve as an example rather than an actual conversion script.

The general flow is that you will first need to download these files by a custom access method, then upload them with the fixed SDK version. Depending on what object you're trying to fix, you should pick the respective lines of code from steps 1 and 2.



1. You need to be able to download objects (models, datasets, media, artifacts) registered by affected versions. See the code snippet below and adjust it according to your use case to be able to get a local copy of the object
```python
from clearml import Task, ImportModel
from urllib.parse import unquote # <- you will need this


ds_task = Task.get_task(dataset_id) # For Datasets
# OR
task = Task.get_task(task_id) # For Artifacts, Media, and Models


url = unquote(ds_task.artifacts['data'].url) # For Datasets
# OR
url = unquote(task.artifacts[artifact_name].url) # For Artifacts
# OR
model = InputModel(task.output_models_id['test_file']) # For Models associated to tasks
url = unquote(model.url)
# OR 
model = InputModel(model_id) # For any Models
url = unquote(model.url)
# OR
samples = task.get_debug_samples(title, series) # For Media/Debug samples
sample_urls = [unquote(sample['url']) for sample in samples]

local_path = StorageManager.get_local_copy(url)

# NOTE: For Datasets you will need to unzip the `local_path`
```

2. Once the object is downloaded locally, you can re-register it with the new version. See the snipped below and adjust according to your use case
```python
from clearml import Task, Dataset, OutputModel
import os


ds = Dataset.create(dataset_name=task.name, dataset_projecte=task.get_project_name(), parents=[Dataset.get(dataset_id)]) # For Datasets
# OR
task = Task.get_task(task_name=task.name, project_name=task.get_project_name()) # For Artifacts, Media, and Models


ds.add_files(unzipped_local_path) # For Datasets
ds.finalize(auto_upload=True)
# OR
task.upload_artifact(name=artifact_name, artifact_object=local_path) # For Artifacts
# OR
model = OutputModel(task=task) # For any Models
model.update_weights(local_path) # note: if the original model was created with update_weights_package,
                                 # preserve this behavior by saving the new one with update_weights_package too
# OR
for sample in samples:
   task.get_logger().report_media(sample['metric'], sample['variant'], local_path=unquote(sample['url'])) # For Media/Debug samples
```

## Alternative methods

The methods described next are more advanced (read "more likely to mess up"). If you're unsure whether to use them or not, better don't. Both methods described below will alter (i.e., modify **in-place**) the existing objects. Note that you still need to run the code from step 1 to have access to all required metadata.

**Method 1**: You can try to alter the existing unpublished experiments/models using the lower-level `APIClient`
```python
from clearml.backend_api.session.client import APIClient


client = APIClient()

client.tasks.add_or_update_artifacts(task=ds_task.id, force=True, artifacts=[{"uri": unquote(ds_task.artifacts['state'].url), "key": "state", "type": "dict"}])
client.tasks.add_or_update_artifacts(task=ds_task.id, force=True, artifacts=[{"uri": unquote(ds_task.artifacts['data'].url), "key": "data", "type": "custom"}]) # For datasets on completed dataset uploads
# OR
client.tasks.add_or_update_artifacts(task=task.id, force=True, artifacts=[{"uri": unquote(url), "key": artifact_name, "type": "custom"}]) # For artifacts on completed tasks
# OR
client.models.edit(model=model.id, force=True, uri=url) # For any unpublished Model
```

**Method 2**: There's an option available only to those who self-host their ClearML server. It is possible to manually update the values registered in MongoDB, but beware - this advanced procedure should be performed with extreme care, as it can lead to an inconsistent state if mishandled.
