# Download CIFAR dataset and create a dataset with ClearML's Dataset class
from clearml import StorageManager, Dataset

manager = StorageManager()

dataset_path = manager.get_local_copy(
    remote_url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
)

dataset = Dataset.create(
    dataset_name="cifar_dataset", dataset_project="dataset_examples"
)

# Prepare and clean data here before it is added to the dataset

dataset.add_files(path=dataset_path)

# Dataset is uploaded to the ClearML Server by default
dataset.upload()

dataset.finalize()
