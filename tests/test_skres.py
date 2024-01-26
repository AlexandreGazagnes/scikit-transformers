import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skres import SkRes, _use_case


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

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("estimator", DummyClassifier()),
        ]
    )
    return pipe


class TestSkres:
    """Test class for skres package"""

    def test_integration(self, X_y: tuple, pipeline: Pipeline):
        """ " Test the integration of the package"""

        param_grid = {
            "imputer__strategy": ["mean", "median", "most_frequent"],
            "scaler__with_mean": [True, False],
            "estimator__strategy": ["most_frequent", "stratified", "uniform"],
        }

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            refit=True,
            return_train_score=True,
            n_jobs=-1,
            verbose=2,
        )

        X, y = X_y
        grid.fit(X, y)

        res = SkRes(grid)

        assert isinstance(res, pd.DataFrame)
        assert res.shape[0] > 5
        assert res.shape[1] > 5

    def test_use_case(self):
        """Test the use case"""

        res = _use_case()

        assert isinstance(res, pd.DataFrame)
        assert res.shape[0] > 5
        assert res.shape[1] > 5
