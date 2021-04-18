# ClearML introducing Dataset management!

## Decoupling Data from Code - The Dataset Paradigm

<a href="https://app.community.clear.ml"><img src="https://github.com/allegroai/clearml/blob/master/docs/dataset_screenshots.gif?raw=true" width="80%"></a>

### The ultimate goal of `clearml-data` is to transform datasets into configuration parameters
Just like any other argument, the dataset argument should retrieve a full local copy of the
dataset to be used by the experiment. 
This means datasets can be efficiently retrieved by any machine in a reproducible way.
Together it creates a full version control solution for all your data,  
that is both machine and environment agnostic.


### Design Goals : Simple / Agnostic / File-based / Efficient

## Key Concepts:
1) **Dataset** is a **collection of files** : e.g. folder with all subdirectories and files included in the dataset
2) **Differential storage** : Efficient storage / network
3) **Flexible**: support addition / removal / merge of files and datasets
4) **Descriptive, transparent & searchable**: support projects, names, descriptions, tags and searchable fields
5) **Simple interface**  (CLI and programmatic)
6) **Accessible**: get a copy of the dataset files from anywhere on any machine

### Workflow:

#### Simple dataset creation with CLI:

- Create a dataset
``` bash
clearml-data create --project <my_project> --name <my_dataset_name>
```
- Add local files to the dataset
``` bash
clearml-data add --files ~/datasets/best_dataset/
```
- Close dataset and upload files (Optional: specify storage `--storage` `s3://bucket`, `gs://`, `azure://` or `/mnt/shared/`)
``` bash
clearml-data close --id <dataset_id>
```


#### Integrating datasets into your code:
```python
from argparse import ArgumentParser
from clearml import Dataset

# adding command line interface, so it is easy to use
parser = ArgumentParser()
parser.add_argument('--dataset', default='aayyzz', type=str, help='Dataset ID to train on')
args = parser.parse_args()

# creating a task, so that later we could override the argparse from UI
task = Task.init(project_name='examples', task_name='dataset demo')

# getting a local copy of the dataset
dataset_folder = Dataset.get(dataset_id=args.dataset).get_local_copy()

# go over the files in `dataset_folder` and train your model
```

#### Create dataset from code
Creating datasets from code is especially helpful when some preprocessing is done on raw data and we want to save
preprocessing code as well as dataset in a single Task.

```python
from clearml import Dataset

# Preprocessing code here

dataset = Dataset.create(dataset_name='dataset name',dataset_project='dataset project')
dataset.add_files('/path_to_data')
dataset.upload()
dataset.close()

```

#### Modifying a dataset with CLI:

- Create a new dataset (specify the parent dataset id)
```bash
clearml-data create --name <improved_dataset> --parents <existing_dataset_id>
```
- Get a mutable copy of the current dataset
```bash
clearml-data get --id <created_dataset_id> --copy ~/datasets/working_dataset
```
- Change / add / remove files from the dataset folder
```bash
vim ~/datasets/working_dataset/everything.csv
```

#### Folder sync mode

Folder sync mode updates dataset according to folder content changes.<br/>
This is useful in case there's a single point of truth, either a local or network folder that gets updated periodically.
When using `clearml-data sync` and specifying parent dataset, the folder changes will be reflected in a new dataset version.
This saves time manually updating (adding \ removing) files.

- Sync local changes
``` bash
clearml-data sync --id <created_dataset_id> --folder ~/datasets/working_dataset
```
- Upload files (Optional: specify storage `--storage` `s3://bucket`, `gs://`, `azure://`, `/mnt/shared/`)
``` bash
clearml-data upload --id <created_dataset_id>
```
- Close dataset
``` bash
clearml-data close --id <created_dataset_id>
```


#### Command Line Interface Summary:

- **`search`**  Search a dataset based on project / name / description / tag etc.
- **`list`**  List the file directory content of a dataset (no need to download a copy pf the dataset)
- **`verify`**  Verify a local copy of a dataset (verify the dataset files SHA2 hash)
- **`create`**  Create a new dataset (support extending/inheriting multiple parents)
- **`delete`**  Delete a dataset
- **`add`**  Add local files to a dataset
- **`sync`**  Sync dataset with a local folder (source-of-truth being the local folder)
- **`remove`**  Remove files from dataset (no need to download a copy of the dataset)
- **`get`**  Get a local copy of the dataset (either readonly --link, or writable --copy)
- **`upload`**  Upload the dataset (use --storage to specify storage target such as S3/GS/Azure/Folder, default: file server)


#### Under the hood (how it all works):

Each dataset instance stores the collection of files added/modified from the previous version (parent).

When requesting a copy of the dataset all parent datasets on the graph are downloaded and a new folder
is merged with all changes introduced in the dataset DAG.

Implementation details:

Dataset differential snapshot is stored in a single zip file for efficiency in storage and network
bandwidth. Local cache is built into the process making sure datasets are downloaded only once.
Dataset contains SHA2 hash of all the files in the dataset.
In order to increase dataset fetching speed, only file size is verified automatically,
the SHA2 hash is verified only on user's request.

The design supports multiple parents per dataset, essentially merging all parents based on order.
To improve deep dataset DAG storage and speed, dataset squashing was introduced. A user can squash
a dataset, merging down all changes introduced in the DAG, creating a new flat version without parent datasets.


### Datasets UI:

A dataset is represented as a special `Task` in the system. <br>
It is of type `data-processing` with a special tag `dataset`.

- Full log (calls / CLI) of the dataset creation process can be found in the "Execution" section.
- Listing of the dataset differential snapshot, summary of files added / modified / removed and details of files
in the differential snapshot (location / size / hash), is available in the Artifacts section you can find a
- The full dataset listing (all files included) is available in the Configuration section under `Dataset Content`.
This allows you to quickly compare two dataset contents and visually see the difference.
- The dataset genealogy DAG and change-set summary table is visualized in Results / Plots

