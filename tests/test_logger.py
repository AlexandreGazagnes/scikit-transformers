import pytest

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

from sktransf.logger import LogTransformer, _use_case


@pytest.fixture
def X_y() -> tuple:
    """Load the data"""

    base = "https://gist.githubusercontent.com/AlexandreGazagnes/"
    url = base + "9018022652ba0933dd39c9df8a600292/raw/"
    url += "0845ef4c2df4806bb05c8c7423dc75d93e37400f/titanic_train_raw_csv"

    df = pd.read_csv(url)
    y = df.Survived

    X = df.iloc[:, 2:].select_dtypes(include="number")
    return X, y


@pytest.fixture
def pipeline():
    """Create a pipeline"""

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("logger", LogTransformer()),
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
        ]
    )

    return pipeline


class TestLogTransformer:
    """Test class for skres package"""

    def test_integration(self, X_y: tuple, pipeline: Pipeline):
        """ " Test the integration of the package"""

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

        X, y = X_y
        grid.fit(X, y)

    def test_use_case(self):
        """Test the use case"""

        res = _use_case()
