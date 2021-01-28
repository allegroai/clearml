# ClearML - Example of LightGBM integration
#
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="examples", task_name="LIGHTgbm")

print('Loading data...')

# Load or create your dataset


df_train = pd.read_csv(
    'https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.train',
    header=None, sep='\t'
)
df_test = pd.read_csv(
    'https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.test',
    header=None, sep='\t'
)

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# Create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')

# Train
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=20,
    valid_sets=lgb_eval,
    early_stopping_rounds=5
)

print('Saving model...')

# Save model to file
gbm.save_model('model.txt')

print('Loading model to predict...')

# Load model to predict
bst = lgb.Booster(model_file='model.txt')

# Can only predict with the best iteration (or the saving iteration)
y_pred = bst.predict(X_test)

# Eval with loaded model
print("The rmse of loaded model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)
