import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from clearml import Task


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="examples", task_name="pipeline step 3 train model")

# Arguments
args = {
    'dataset_task_id': 'REPLACE_WITH_DATASET_TASK_ID',
}
task.connect(args)

# only create the task, we will actually execute it later
task.execute_remotely()

print('Retrieving Iris dataset')
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
X_train = dataset_task.artifacts['X_train'].get()
X_test = dataset_task.artifacts['X_test'].get()
y_train = dataset_task.artifacts['y_train'].get()
y_test = dataset_task.artifacts['y_test'].get()
print('Iris dataset loaded')

model = LogisticRegression(solver='liblinear', multi_class='auto')
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl', compress=True)

loaded_model = joblib.load('model.pkl')
result = loaded_model.score(X_test, y_test)

print('model trained & stored')

x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
plt.figure(1, figsize=(4, 3))

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.title('Iris Types')
plt.show()

print('Done')
