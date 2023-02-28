import shutil
from uuid import uuid4

from pathlib2 import Path

from clearml import Dataset, StorageManager


def download_mnist_dataset():
    manager = StorageManager()
    mnist_dataset = Path(manager.get_local_copy(
        remote_url="https://allegro-datasets.s3.amazonaws.com/datasets/MNIST.zip", name="MNIST"))
    mnist_dataset_train = mnist_dataset / "TRAIN"
    mnist_dataset_test = mnist_dataset / "TEST"

    return mnist_dataset_train, mnist_dataset_test


def main():
    print("STEP1 : Downloading mnist dataset")
    mnist_dataset_train, mnist_dataset_test = download_mnist_dataset()

    print("STEP2 : Preparing mnist dataset folder")
    mnist_path = Path(f"MNIST_{uuid4().hex}")
    mnist_train_path = mnist_path / "TRAIN"
    mnist_test_path = mnist_path / "TEST"
    mnist_path.mkdir()

    print("STEP3 : Creating the dataset")
    mnist_dataset = Dataset.create(
        dataset_project="dataset_examples", dataset_name="MNIST Complete Dataset (Syncing Example)")

    print("STEP4 : Syncing train dataset")
    shutil.copytree(mnist_dataset_train, mnist_train_path)  # Populating dataset folder with TRAIN images
    mnist_dataset.sync_folder(mnist_path)
    mnist_dataset.upload()

    print("STEP5 : Syncing test dataset")
    shutil.copytree(mnist_dataset_train, mnist_test_path)  # Populating dataset folder with TEST images
    mnist_dataset.sync_folder(mnist_path)
    mnist_dataset.upload()

    print("STEP6 : Finalizing dataset")
    mnist_dataset.finalize()

    print("We are done, have a great day :)")


if __name__ == '__main__':
    main()
