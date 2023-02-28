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
    train_dataset = Dataset.create(
        dataset_project="dataset_examples/MNIST", dataset_name="MNIST Training Dataset")
    train_dataset.add_files(path=mnist_dataset_train, dataset_path="TRAIN")
    train_dataset.upload()
    train_dataset.finalize()

    print("STEP3 : Creating the testing dataset")
    test_dataset = Dataset.create(
        dataset_project="dataset_examples/MNIST", dataset_name="MNIST Testing Dataset")
    test_dataset.add_files(path=mnist_dataset_test, dataset_path="TEST")
    test_dataset.upload()
    test_dataset.finalize()

    print("STEP4 : Create a child dataset with both mnist train and test data")
    child_dataset = Dataset.create(
        dataset_project="dataset_examples/MNIST", dataset_name="MNIST Complete Dataset",
        parent_datasets=[train_dataset.id, test_dataset.id])
    child_dataset.upload()
    child_dataset.finalize()

    print("We are done, have a great day :)")


if __name__ == "__main__":
    main()
