"""
Utils for sktransf
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression

from sktransf.logger import LogTransformer


def get_titanic() -> tuple[pd.DataFrame, pd.Series]:
    """Get titanic dataset from github gist"""

    base = "https://gist.githubusercontent.com/AlexandreGazagnes/"
    url = base + "9018022652ba0933dd39c9df8a600292/raw/"
    url += "0845ef4c2df4806bb05c8c7423dc75d93e37400f/titanic_train_raw_csv"

    df = pd.read_csv(url)
    y = df.Survived

    X = df.iloc[:, 2:].select_dtypes(include="number")

    return X, y


def use_case() -> GridSearchCV:
    """run a simple use case"""

    # pipeline
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("logger", LogTransformer()),
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
        ]
    )

    # param grid
    param_grid = {
        "logger__threshold": [1, 1.5, 3],
        "scaler": [StandardScaler(), "passthrough"],
    }

    # grid
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        refit=True,
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    # data
    X, y = get_titanic()

    # fit
    grid.fit(X, y)

    return grid
