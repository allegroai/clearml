from clearml import StorageManager, Dataset


def main():
    manager = StorageManager()

    print("STEP1 : Downloading CSV dataset")
    csv_file_path = manager.get_local_copy(
        remote_url="https://allegro-datasets.s3.amazonaws.com/datasets/Iris_Species.csv")

    print("STEP2 : Creating a dataset")
    # By default, clearml data uploads to the clearml fileserver. Adding output_uri argument to the create() method
    # allows you to specify custom storage like s3 \ gcs \ azure \ local storage
    simple_dataset = Dataset.create(dataset_project="dataset_examples", dataset_name="CSV_Dataset")

    print("STEP3 : Adding CSV file to the Dataset")
    simple_dataset.add_files(path=csv_file_path)

    print("STEP4 : Upload and finalize")
    simple_dataset.upload()
    simple_dataset.finalize()

    print("We are done, have a great day :)")


if __name__ == '__main__':
    main()
