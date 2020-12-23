import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import plot_tree

from clearml import Task


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='XGBoost simple example')

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations

# noinspection PyBroadException
try:
    # try to load a model
    bst = xgb.Booster(params=param, model_file='xgb.01.model')
    bst.load_model('xgb.01.model')
except Exception:
    bst = None

# if we dont have one train a model
if bst is None:
    bst = xgb.train(param, dtrain, num_round)

# store trained model model v1
bst.save_model('xgb.01.model')
bst.dump_model('xgb.01.raw.txt')

# build classifier
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# store trained classifier model
model.save_model('xgb.02.model')

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
labels = dtest.get_label()

# plot results
xgb.plot_importance(model)
plt.show()
plot_tree(model)
plt.show()
