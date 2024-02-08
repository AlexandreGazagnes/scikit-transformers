"""
Unique Bool transformer
"""

import pandas as pd
import pytest

from sktransf import DropUniqueColumnTransformer, get_titanic


@pytest.fixture
def X() -> pd.DataFrame:
    """Load the data"""

    X, _ = X_y = get_titanic()

    X['unique_col'] = 'hello'

    return X


class TestDropUniqueColumnTransformer:
    """Test for DropUniqueColumnTransformer"""

    def test_X(self, X: pd.DataFrame):
        """Test the fit method"""

        assert isinstance(X, pd.DataFrame)

        assert X.unique_col.nunique() == 1

    def test_init(self):
        """Test the init method"""

        # create the transformer
        transformer = DropUniqueColumnTransformer()

        assert transformer.unique_cols is None

    def test_fit(self, X: pd.DataFrame):
        # create the transformer
        transformer = DropUniqueColumnTransformer()

        # fit the transformer
        transformer.fit(X)

        assert transformer.unique_cols == ['unique_col']

    def test_transform(self, X: pd.DataFrame):
        # create the transformer
        transformer = DropUniqueColumnTransformer()

        # fit the transformer
        transformer.fit(X)

        # transform
        X_ = transformer.transform(X)

        # assert
        assert isinstance(X_, pd.DataFrame)
        assert 'unique_col' not in X_.columns
