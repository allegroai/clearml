# CLI Examples

---
Install `clearml` in your environment

    $ pip3 install clearml

---

## 1. Create a simple dataset from a file

- Creation  
  `clearml-data create --project Datasets_Examples_CLI --name Simple_CSV_dataset_CLI`
- Adding the file  
  `clearml-data add --files YOUR_CSV_DATASET.csv`
- Upload and finalize  
  `clearml-data close --verbose`

### 2. Creating a dataset from a folder

- Creation  
  `clearml-data create --project Datasets_Examples_CLI --name Datset_From_Folder_CLI`
- Adding the folder
  `clearml-data add --files ./YOUR_DATASET_FOLDER`
- Upload and finalize  
  `clearml-data close --verbose`

### 3. Create, add all the files of the directory structure, upload and finalize/close in one command

- Create, add all the files of the directory structure, upload and finalize/close in one command !  
  `clearml-data sync --folder ./DATA/MNIST/TRAIN --project Datasets_Examples_CLI --name MNIST_training_dataset_CLI`

### 4. Creating a dataset with child

- Create, add all the files of the directory structure, upload and finalize/close in one command !  
  `clearml-data sync --folder ./YOUR_DATASET/TRAIN --project Datasets_Examples_CLI --name MNIST_training_dataset_CLI_2`  
  `clearml-data sync --folder ./YOUR_DATASET/TEST --project Datasets_Examples_CLI --name MNIST_testing_dataset_CLI_2`

- Create the child version  
  `clearml-data create --project Datasets_Examples_CLI --name MNIST_complete_dataset_CLI_2 --parents ID_OF_TRAIN_DATASET ID_OF_TEST_DATASET`    
  `clearml-data close --verbose`  
 
