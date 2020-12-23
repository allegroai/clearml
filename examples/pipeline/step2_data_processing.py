import pickle
from clearml import Task, StorageManager
from sklearn.model_selection import train_test_split


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="examples", task_name="pipeline step 2 process dataset")

# program arguments
# Use either dataset_task_id to point to a tasks artifact or
# use a direct url with dataset_url
args = {
    'dataset_task_id': '',
    'dataset_url': '',
    'random_state': 42,
    'test_size': 0.2,
}

# store arguments, later we will be able to change them from outside the code
task.connect(args)
print('Arguments: {}'.format(args))

# only create the task, we will actually execute it later
task.execute_remotely()

# get dataset from task's artifact
if args['dataset_task_id']:
    dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
    print('Input task id={} artifacts {}'.format(args['dataset_task_id'], list(dataset_upload_task.artifacts.keys())))
    # download the artifact
    iris_pickle = dataset_upload_task.artifacts['dataset'].get_local_copy()
# get the dataset from a direct url
elif args['dataset_url']:
    iris_pickle = StorageManager.get_local_copy(remote_url=args['dataset_url'])
else:
    raise ValueError("Missing dataset link")

# open the local copy
iris = pickle.load(open(iris_pickle, 'rb'))

# "process" data
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args['test_size'], random_state=args['random_state'])

# upload processed data
print('Uploading process dataset')
task.upload_artifact('X_train', X_train)
task.upload_artifact('X_test', X_test)
task.upload_artifact('y_train', y_train)
task.upload_artifact('y_test', y_test)

print('Notice, artifacts are uploaded in the background')
print('Done')
