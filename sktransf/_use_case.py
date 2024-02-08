"""
Utils for use case
"""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sktransf.logger import LogTransformer

from ._get_titanic import get_titanic


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
