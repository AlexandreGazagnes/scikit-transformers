"""
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


class LogTransformer(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self, threshold=3, out="np"):
        """ """

        if not isinstance(threshold, (float, int)):
            raise TypeError("threshold must be a float or an integer")

        if not isinstance(out, str):
            raise TypeError("out must be a string")

        if out not in ["df", "np"]:
            raise ValueError("out must be 'df' or 'array'")

        self.out = out
        self.threshold = threshold

    def fit(self, X, y=None):
        """ """

        return self

    def transform(self, X, y=None):
        """ """

        if not isinstance(X, (pd.DataFrame, np.ndarray, np.array, list)):
            raise TypeError(
                "X must be a (pd.DataFrame, np.ndarray, np.array, list)"
            )

        try:
            _X = pd.DataFrame(X).copy()
        except Exception as e:
            raise Exception(f"Impossible to make df/np.array : {e}")

        skew = _X.skew().round(4).to_dict()

        for col in skew:
            if skew[col] >= self.threshold:
                _X[col] = np.log1p(_X[col])

        if self.out == "df":
            return _X

        return _X.values


def _use_case():
    """ """

    import numpy as np
    import pandas as pd

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import KNNImputer, SimpleImputer

    base = "https://gist.githubusercontent.com/AlexandreGazagnes/"
    url = base + "9018022652ba0933dd39c9df8a600292/raw/"
    url += "0845ef4c2df4806bb05c8c7423dc75d93e37400f/titanic_train_raw_csv"

    df = pd.read_csv(url)
    y = df.Survived

    X = df.iloc[:, 2:].select_dtypes(include="number")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("logger", LogTransformer()),
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
        ]
    )

    param_grid = {
        "logger__threshold": [1, 1.5, 3],
        "scaler": [StandardScaler(), "passthrough"],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        refit=True,
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X, y)
