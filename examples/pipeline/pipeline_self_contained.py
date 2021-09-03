from clearml import PipelineController


def step_one(pickle_data_url):
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


def step_two(data_frame, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    y = data_frame['target']
    X = data_frame[(c for c in data_frame.columns if c != 'target')]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def step_three(data):
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = data
    model = LogisticRegression(solver='liblinear', multi_class='auto')
    model.fit(X_train, y_train)
    return model


def debug_testing_our_pipeline(pickle_url):
    data_frame = step_one(pickle_url)
    processed_data = step_two(data_frame)
    model = step_three(processed_data)
    print(model)


pipe = PipelineController(
    project='examples',
    name='pipeline demo',
    version='1.1',
    add_pipeline_tags=False,
)

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

# for debugging purposes use local jobs
# pipe.start_locally()

# Starting the pipeline on the services queue (remote machine, default on the clearml-server)
pipe.start()

print('pipeline done')
