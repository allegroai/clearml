# ClearML - Example of CatBoost training, saving model and loading model
#
import argparse

from catboost import CatBoostRegressor, Pool
from catboost.datasets import msrank

from clearml import Task

import numpy as np

from sklearn.model_selection import train_test_split


def main(iterations):
    # Download train and validation datasets
    train_df, test_df = msrank()
    # Column 0 contains label values, column 1 contains group ids.
    X_train, y_train = train_df.drop([0, 1], axis=1).values, train_df[0].values
    X_test, y_test = test_df.drop([0, 1], axis=1).values, test_df[0].values

    # Split train data into two parts. First part - for baseline model,
    # second part - for major model
    splitted_data = train_test_split(X_train, y_train, test_size=0.5)
    X_train_first, X_train_second, y_train_first, y_train_second = splitted_data

    catboost_model = CatBoostRegressor(iterations=iterations, verbose=False)

    # Prepare simple baselines (just mean target on first part of train pool).
    baseline_value = y_train_first.mean()
    train_baseline = np.array([baseline_value] * y_train_second.shape[0])
    test_baseline = np.array([baseline_value] * y_test.shape[0])

    # Create pools
    train_pool = Pool(X_train_second, y_train_second, baseline=train_baseline)
    test_pool = Pool(X_test, y_test, baseline=test_baseline)

    # Train CatBoost model
    catboost_model.fit(train_pool, eval_set=test_pool, verbose=True, plot=False, save_snapshot=True)
    catboost_model.save_model("example.cbm")

    catboost_model = CatBoostRegressor()
    catboost_model.load_model("example.cbm")

    # Apply model on pool with baseline values
    preds1 = catboost_model.predict(test_pool)

    # Apply model on numpy.array and then add the baseline values
    preds2 = test_baseline + catboost_model.predict(X_test)

    # Check that preds have small diffs
    assert (np.abs(preds1 - preds2) < 1e-6).all()


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="CatBoost simple example")
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", default=200)
    args = parser.parse_args()
    main(args.iterations)
