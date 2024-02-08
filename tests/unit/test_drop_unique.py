import logging
import random

import pytest

import numpy as np
import pandas as pd

from sktransf import DropUniqueColumnTransformer
from sktransf import get_titanic


@pytest.fixture
def X() -> pd.DataFrame:
    """Load the data"""

    X, _ = X_y = get_titanic()

    X["unique_col"] = "hello"

    return X


class TestDropUniqueColumnTransformer:
    """Test for DropUniqueColumnTransformer"""

    def test_X(self, X: pd.DataFrame):
        """Test the fit method"""

        assert isinstance(X, pd.DataFrame)

        assert X.unique_col.nunique() == 1

    def test_fit(self, X: pd.DataFrame):

        # create the transformer
        transformer = DropUniqueColumnTransformer()

        # fit the transformer
        transformer.fit(X)

        assert transformer.unique_cols == ["unique_col"]

    def test_transform(self, X: pd.DataFrame):

        # create the transformer
        transformer = DropUniqueColumnTransformer()

        # fit the transformer
        transformer.fit(X)

        # transform
        X_ = transformer.transform(X)

        assert "unique_col" not in X_.columns
