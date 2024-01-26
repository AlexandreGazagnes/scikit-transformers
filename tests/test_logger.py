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

from sktransf import LogTransformer
from sktransf import get_titanic


@pytest.fixture
def X_y() -> tuple:
    """Load the data"""

    return get_titanic()


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
        """Test the integration of the package"""

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
