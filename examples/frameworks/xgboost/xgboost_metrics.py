import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from clearml import Task

task = Task.init(project_name="examples", task_name="XGBoost metric auto reporting")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {"objective": "reg:squarederror", "eval_metric": "rmse"}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=0,
)

bst.save_model("best_model")
