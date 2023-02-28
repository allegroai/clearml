from pathlib2 import Path

from clearml import Dataset, StorageManager


def main():
    manager = StorageManager()

    print("STEP1 : Downloading mnist dataset")
    mnist_dataset = Path(manager.get_local_copy(
        remote_url="https://allegro-datasets.s3.amazonaws.com/datasets/MNIST.zip", name="MNIST"))
    mnist_dataset_train = mnist_dataset / "TRAIN"
    mnist_dataset_test = mnist_dataset / "TEST"

    print("STEP2 : Creating the training dataset")
    mnist_dataset = Dataset.create(
        dataset_project="dataset_examples", dataset_name="MNIST Training Dataset")
    mnist_dataset.add_files(path=mnist_dataset_train, dataset_path="TRAIN")
    mnist_dataset.upload()
    mnist_dataset.finalize()

    print("STEP3 : Create a child dataset of mnist dataset using TEST Dataset")
    child_dataset = Dataset.create(
        dataset_project="dataset_examples", dataset_name="MNIST Complete Dataset", parent_datasets=[mnist_dataset.id])
    child_dataset.add_files(path=mnist_dataset_test, dataset_path="TEST")
    child_dataset.upload()
    child_dataset.finalize()

    print("We are done, have a great day :)")


if __name__ == '__main__':
    main()
