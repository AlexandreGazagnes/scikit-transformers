import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sktransf import LogTransformer, get_titanic


@pytest.fixture
def X_y() -> tuple:
    """Load the data"""

    return get_titanic()


@pytest.fixture
def pipeline():
    """Create a pipeline"""

    pipeline = Pipeline(
        [
            ('imputer', SimpleImputer()),
            ('logger', LogTransformer()),
            ('scaler', StandardScaler()),
            ('estimator', LogisticRegression()),
        ]
    )

    return pipeline


class TestLogTransformer:
    """Test class for skres package"""

    def test_integration(self, X_y: tuple, pipeline: Pipeline):
        """Test the integration of the package"""

        param_grid = {
            'logger__threshold': [1, 1.5, 3],
            'scaler': [StandardScaler(), 'passthrough'],
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
