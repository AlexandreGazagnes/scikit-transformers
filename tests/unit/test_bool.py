import numpy as np
import pandas as pd
import pytest

from sktransf import BoolColumnTransformer, get_titanic


@pytest.fixture
def X() -> pd.DataFrame:
    """Load the data"""

    X, _ = X_y = get_titanic()

    X["bool_col"] = np.random.choice(["a", "b"], size=X.shape[0])

    return X


class TestBoolColumnTransformer:
    """Test for BoolColumnTransformer"""

    def test_X(self, X: pd.DataFrame):
        """Test X"""

        assert isinstance(X, pd.DataFrame)

        assert X.bool_col.nunique() == 2

    def test_init(self):
        """Test the init method"""

        # create the transformer
        transformer = BoolColumnTransformer()

        assert transformer.bool_cols is None

    def test_fit(self, X: pd.DataFrame):
        """Test the fit method"""

        # create the transformer
        transformer = BoolColumnTransformer()

        # fit the transformer
        transformer.fit(X)

        assert transformer.bool_cols == ["bool_col"]

    def test_transform(self, X: pd.DataFrame):
        """Test the transform method"""

        # create the transformer
        transformer = BoolColumnTransformer()

        # fit the transformer
        transformer.fit(X)

        # transform
        X_ = transformer.transform(X)

        assert isinstance(X_, pd.DataFrame)
        assert X_.bool_col.nunique() == 2
        vals = X_.bool_col.unique().tolist()
        assert sorted(vals) == [0, 1]
