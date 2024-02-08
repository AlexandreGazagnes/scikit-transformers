import logging
import random

import pytest

import numpy as np
import pandas as pd

from sktransf import BoolColumnTransformer
from sktransf import get_titanic


@pytest.fixture
def X() -> pd.DataFrame:
    """Load the data"""

    X, _ = X_y = get_titanic()

    X["bool_col"] = np.random.choice(["a", "b"], size=X.shape[0])

    return X


class TestBoolColumnTransformer:
    """Test for BoolColumnTransformer"""

    def test_X(self, X: pd.DataFrame):
        """Test the fit method"""

        assert isinstance(X, pd.DataFrame)

        assert X.bool_col.nunique() == 2

    def test_fit(self, X: pd.DataFrame):

        # create the transformer
        transformer = BoolColumnTransformer()

        # fit the transformer
        transformer.fit(X)

        assert transformer.bool_cols == ["bool_col"]

    def test_transform(self, X: pd.DataFrame):

        # create the transformer
        transformer = BoolColumnTransformer()

        # fit the transformer
        transformer.fit(X)

        # transform
        X_ = transformer.transform(X)

        assert "bool_cols" not in X_.columns
