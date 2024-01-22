from clearml import PipelineDecorator, Task


@PipelineDecorator.component(cache=True)
def create_dataset(source_url: str, project: str, dataset_name: str) -> str:
    print("starting create_dataset")
    from clearml import StorageManager, Dataset
    import pandas as pd
    local_file = StorageManager.get_local_copy(source_url)
    df = pd.read_csv(local_file, header=None)
    df.to_csv(path_or_buf="./dataset.csv", index=False)
    dataset = Dataset.create(dataset_project=project, dataset_name=dataset_name)
    dataset.add_files("./dataset.csv")
    dataset.get_logger().report_table(title="sample", series="head", table_plot=df.head())
    dataset.finalize(auto_upload=True)

    print("done create_dataset")
    return dataset.id


@PipelineDecorator.component(cache=True)
def preprocess_dataset(dataset_id: str):
    print("starting preprocess_dataset")
    from clearml import Dataset
    from pathlib import Path
    import pandas as pd
    dataset = Dataset.get(dataset_id=dataset_id)
    local_folder = dataset.get_local_copy()
    df = pd.read_csv(Path(local_folder) / "dataset.csv", header=None)
    # "preprocessing" - adding columns
    df.columns = [
        'age', 'workclass', 'fnlwgt', 'degree', 'education-yrs', 'marital-status',
        'occupation', 'relationship', 'ethnicity', 'gender', 'capital-gain',
        'capital-loss', 'hours-per-week', 'native-country', 'income-cls',
    ]
    df.to_csv(path_or_buf="./dataset.csv", index=False)

    # store in a new dataset
    new_dataset = Dataset.create(
        dataset_project=dataset.project, dataset_name="{} v2".format(dataset.name),
        parent_datasets=[dataset]
    )
    new_dataset.add_files("./dataset.csv")
    new_dataset.get_logger().report_table(title="sample", series="head", table_plot=df.head())
    new_dataset.finalize(auto_upload=True)

    print("done preprocess_dataset")
    return new_dataset.id


@PipelineDecorator.component(cache=True)
def verify_dataset_integrity(dataset_id: str, expected_num_columns: int):
    print("starting verify_dataset_integrity")
    from clearml import Dataset, Logger
    from pathlib import Path
    import numpy as np
    import pandas as pd
    dataset = Dataset.get(dataset_id=dataset_id)
    local_folder = dataset.get_local_copy()
    df = pd.read_csv(Path(local_folder) / "dataset.csv")
    print("Verifying dataset")
    assert len(df.columns) == expected_num_columns
    print("PASSED")
    # log some stats on the age column
    Logger.current_logger().report_histogram(
        title="histogram", series="age", values=np.histogram(df["age"])
    )

    print("done verify_dataset_integrity")
    return True


@PipelineDecorator.component(output_uri=True)
def train_model(dataset_id: str, training_args: dict):
    print("starting train_model")
    from clearml import Dataset, OutputModel, Task
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    dataset = Dataset.get(dataset_id=dataset_id)
    local_folder = dataset.get_local_copy()
    df = pd.read_csv(Path(local_folder) / "dataset.csv")

    # prepare data (i.e. select specific columns)
    columns = ["age", "fnlwgt", "education-yrs", "capital-gain", "capital-loss", "hours-per-week"]
    X = df[columns].drop("age", axis=1)
    y = df["age"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # create matrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # train with XGBoost
    params = {"objective": "reg:squarederror", "eval_metric": "rmse"}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=training_args.get("num_boost_round", 100),
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=0,
    )
    # evaluate
    y_pred = bst.predict(dtest)
    plt.plot(y_test, 'r')
    plt.plot(y_pred, 'b')

    # let's store the eval score
    error = np.linalg.norm(y_test-y_pred)
    bst.save_model("a_model.xgb")

    Task.current_task().reload()
    model_id = Task.current_task().models['output'][-1].id
    print("done train_model")
    return dict(error=error, model_id=model_id)


@PipelineDecorator.component(monitor_models=["best"])
def select_best_model(models_score: list):
    print("starting select_best_model:", models_score)
    from clearml import OutputModel, Task
    best_model = None
    for m in models_score:
        if not best_model or m["error"] < best_model["error"]:
            best_model = m

    print("The best model is {}".format(best_model))
    # lets store it on the pipeline
    best_model = OutputModel(base_model_id=best_model["model_id"])
    # let's make sure we have it
    best_model.connect(task=Task.current_task(), name="best")

    print("done select_best_model")
    return best_model.id


@PipelineDecorator.pipeline(
    name='xgboost_pipeline',
    project='xgboost_pipe_demo',
    version='0.1'
)
def pipeline(data_url: str, project: str):

    dataset_id = create_dataset(source_url=data_url, project=project, dataset_name="mock")

    preprocessed_dataset_id = preprocess_dataset(dataset_id=dataset_id)

    if not bool(verify_dataset_integrity(
            dataset_id=preprocessed_dataset_id,
            expected_num_columns=15)
    ):
        print("Verification Failed!")
        return False

    print("start training models")
    models_score = []
    for i in [100, 150]:
        model_score = train_model(
            dataset_id=preprocessed_dataset_id, training_args=dict(num_boost_round=i)
        )
        models_score.append(model_score)

    model_id = select_best_model(models_score=models_score)
    print("selected model_id = {}".format(model_id))


if __name__ == '__main__':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    # comment to run the entire pipeline remotely
    if Task.running_locally():
        # this is for demonstration purpose only,
        # it will run the entire pipeline logic and components locally
        PipelineDecorator.run_locally()

    pipeline(data_url=url, project="xgboost_pipe_demo")
