"""
Unique Bool transformer
"""

import pytest

import numpy as np
import pandas as pd

from sktransf.selector import DropSkuColumnSelector


class TestDropSkuColumnSelector:
    """Test for DropSkuColumnSelector"""

    def test_X(self, X: pd.DataFrame):
        """Test the fit method"""

        assert isinstance(X, pd.DataFrame)

    def test_init(self):
        """Test the init method"""

        # create the transformer
        transformer = DropSkuColumnSelector(force_df_out=True)

        assert transformer._sku_cols is None

    def test_fit(self, X: pd.DataFrame):
        # create the transformer
        transformer = DropSkuColumnSelector(force_df_out=True)

        # fit the transformer
        transformer.fit(X)

        assert "PassengerId" in transformer._sku_cols

    def test_transform(self, X: pd.DataFrame):
        # create the transformer
        transformer = DropSkuColumnSelector(force_df_out=True)

        # fit the transformer
        transformer.fit(X)

        # transform
        X_sku_ = transformer.transform(X)

        # assert
        assert isinstance(X_sku_, pd.DataFrame)
        assert "PassengerId" not in X_sku_.columns

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
        transformer = DropSkuColumnSelector(force_df_out=force_df_out)

        # fit the transformer
        transformer.fit(X)
        X_ = transformer.transform(X)

        # type  check
        assert isinstance(X_, _type)
