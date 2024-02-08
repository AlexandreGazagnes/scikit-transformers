"""
Unique Bool transformer
"""

import pandas as pd

from sktransf import DropUniqueColumnTransformer


class TestDropUniqueColumnTransformer:
    """Test for DropUniqueColumnTransformer"""

    def test_X(self, X_unique: pd.DataFrame):
        """Test the fit method"""

        assert isinstance(X_unique, pd.DataFrame)

        assert X_unique.unique_col.nunique() == 1

    def test_init(self):
        """Test the init method"""

        # create the transformer
        transformer = DropUniqueColumnTransformer()

        assert transformer.unique_cols is None

    def test_fit(self, X_unique: pd.DataFrame):
        # create the transformer
        transformer = DropUniqueColumnTransformer()

        # fit the transformer
        transformer.fit(X_unique)

        assert transformer.unique_cols == ["unique_col"]

    def test_transform(self, X_unique: pd.DataFrame):
        # create the transformer
        transformer = DropUniqueColumnTransformer()

        # fit the transformer
        transformer.fit(X_unique)

        # transform
        X_unique_ = transformer.transform(X_unique)

        # assert
        assert isinstance(X_unique_, pd.DataFrame)
        assert "unique_col" not in X_unique_.columns
