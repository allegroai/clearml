from clearml import PipelineController


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_one(pickle_data_url):
    # make sure we have scikit-learn for this step, we need it to use to unpickle the object
    import sklearn  # noqa
    import pickle
    import pandas as pd
    from clearml import StorageManager
    pickle_data_url = \
        pickle_data_url or \
        'https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl'
    local_iris_pkl = StorageManager.get_local_copy(remote_url=pickle_data_url)
    with open(local_iris_pkl, 'rb') as f:
        iris = pickle.load(f)
    data_frame = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    data_frame.columns += ['target']
    data_frame['target'] = iris['target']
    return data_frame


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_two(data_frame, test_size=0.2, random_state=42):
    # make sure we have pandas for this step, we need it to use the data_frame
    import pandas as pd  # noqa
    from sklearn.model_selection import train_test_split
    y = data_frame['target']
    X = data_frame[(c for c in data_frame.columns if c != 'target')]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_three(data):
    # make sure we have pandas for this step, we need it to use the data_frame
    import pandas as pd  # noqa
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = data
    model = LogisticRegression(solver='liblinear', multi_class='auto')
    model.fit(X_train, y_train)
    return model


if __name__ == '__main__':

    # create the pipeline controller
    pipe = PipelineController(
        project='examples',
        name='Pipeline demo',
        version='1.1',
        add_pipeline_tags=False,
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue('default')

    # add pipeline components
    pipe.add_parameter(
        name='url',
        description='url to pickle file',
        default='https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl'
    )
    pipe.add_function_step(
        name='step_one',
        function=step_one,
        function_kwargs=dict(pickle_data_url='${pipeline.url}'),
        function_return=['data_frame'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='step_two',
        # parents=['step_one'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=step_two,
        function_kwargs=dict(data_frame='${step_one.data_frame}'),
        function_return=['processed_data'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='step_three',
        # parents=['step_two'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=step_three,
        function_kwargs=dict(data='${step_two.processed_data}'),
        function_return=['model'],
        cache_executed_step=True,
    )

    # For debugging purposes run on the pipeline on current machine
    # Use run_pipeline_steps_locally=True to further execute the pipeline component Tasks as subprocesses.
    # pipe.start_locally(run_pipeline_steps_locally=False)

    # Start the pipeline on the services queue (remote machine, default on the clearml-server)
    pipe.start()

    print('pipeline completed')
