"""
Unique Bool transformer
"""

import pytest

import numpy as np
import pandas as pd

from sktransf.selector import DropUniqueColumnSelector


class TestDropUniqueColumnSelector:
    """Test for DropUniqueColumnSelector"""

    def test_X(self, X_unique: pd.DataFrame):
        """Test the fit method"""

        assert isinstance(X_unique, pd.DataFrame)

        assert X_unique.unique_col.nunique() == 1

    def test_init(self):
        """Test the init method"""

        # create the transformer
        transformer = DropUniqueColumnSelector(force_df_out=True)

        assert transformer._unique_cols is None

    def test_fit(self, X_unique: pd.DataFrame):
        # create the transformer
        transformer = DropUniqueColumnSelector(force_df_out=True)

        # fit the transformer
        transformer.fit(X_unique)

        assert transformer._unique_cols == ["unique_col"]

    def test_transform(self, X_unique: pd.DataFrame):
        # create the transformer
        transformer = DropUniqueColumnSelector(force_df_out=True)

        # fit the transformer
        transformer.fit(X_unique)

        # transform
        X_unique_ = transformer.transform(X_unique)

        # assert
        assert isinstance(X_unique_, pd.DataFrame)
        assert "unique_col" not in X_unique_.columns

    @pytest.mark.parametrize(
        "force_df_out,_type",
        [
            (True, pd.DataFrame),
            (False, np.ndarray),
        ],
    )
    def test_force_df_out(
        self,
        X: pd.DataFrame,
        force_df_out: bool,
        _type: None,
    ):
        """Test the force_df_out attribute"""

        # create the transformer
        transformer = DropUniqueColumnSelector(force_df_out=force_df_out)

        # fit the transformer
        transformer.fit(X)
        X_ = transformer.transform(X)

        # type  check
        assert isinstance(X_, _type)
