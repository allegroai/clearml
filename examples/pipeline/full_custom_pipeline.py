from clearml.automation.controller import PipelineDecorator


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(return_values=['data_frame'], cache=True)
def step_one(pickle_data_url: str, extra: int = 43):
    print('step_one')
    # make sure we have scikit-learn for this step, we need it to use to unpickle the object
    import sklearn  # noqa
    import pickle
    import pandas as pd
    from clearml import StorageManager
    local_iris_pkl = StorageManager.get_local_copy(remote_url=pickle_data_url)
    with open(local_iris_pkl, 'rb') as f:
        iris = pickle.load(f)
    data_frame = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    data_frame.columns += ['target']
    data_frame['target'] = iris['target']
    return data_frame


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step.
# Specifying `return_values` makes sure the function step can return an object to the pipeline logic
# In this case, the returned tuple will be stored as an artifact named "processed_data"
@PipelineDecorator.component(return_values=['processed_data'], cache=True,)
def step_two(data_frame, test_size=0.2, random_state=42):
    print('step_two')
    # make sure we have pandas for this step, we need it to use the data_frame
    import pandas as pd  # noqa
    from sklearn.model_selection import train_test_split
    y = data_frame['target']
    X = data_frame[(c for c in data_frame.columns if c != 'target')]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
# Specifying `return_values` makes sure the function step can return an object to the pipeline logic
# In this case, the returned object will be stored as an artifact named "model"
@PipelineDecorator.component(return_values=['model'], cache=True,)
def step_three(data):
    print('step_three')
    # make sure we have pandas for this step, we need it to use the data_frame
    import pandas as pd  # noqa
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = data
    model = LogisticRegression(solver='liblinear', multi_class='auto')
    model.fit(X_train, y_train)
    return model


# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(name='custom pipeline logic', project='examples', version='0.0.1')
def executing_pipeline(pickle_url, mock_parameter='mock'):
    print('pipeline args:', pickle_url, mock_parameter)

    # Use the pipeline argument to start the pipeline and pass it ot the first step
    data_frame = step_one(pickle_url)

    # Use the returned data from the first step (`step_one`), and pass it to the next step (`step_two`)
    # Notice! unless we actually access the `data_frame` object,
    # the pipeline logic does not actually load the artifact itself.
    # When actually passing the `data_frame` object into a new step,
    # It waits for the creating step/function (`step_one`) to complete the execution
    processed_data = step_two(data_frame)

    # Notice we can actually process/modify the returned values inside the pipeline logic context.
    # This means the modified object will be stored on the pipeline Task.
    processed_data = [processed_data[0], processed_data[1]*2, processed_data[2], processed_data[3]]
    model = step_three(processed_data)

    # Notice since we are "printing" the `model` object,
    # we actually deserialize the object from the third step, and thus wait for the third step to complete.
    print('pipeline completed with model: {}'.format(model))


if __name__ == '__main__':
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    PipelineDecorator.set_default_execution_queue('default')
    # run the pipeline steps as subprocess on the current machine, for debugging purposes
    # PipelineDecorator.debug_pipeline()

    # Start the pipeline execution logic.
    executing_pipeline(
        pickle_url='https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl',
    )

    print('process completed')
