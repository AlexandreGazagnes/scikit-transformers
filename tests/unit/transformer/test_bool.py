"""
TestBoolColumnTransformer
"""

import pytest

import numpy as np
import pandas as pd

from sktransf.transformer import BoolColumnTransformer


class TestBoolColumnTransformer:
    """Test for BoolColumnTransformer"""

    def test_X(self, X_bool: pd.DataFrame):
        """Test X_bool"""

        assert isinstance(X_bool, pd.DataFrame)

        assert X_bool.bool_col.nunique() == 2

    def test_init(self):
        """Test the init method"""

        # create the transformer
        transformer = BoolColumnTransformer(force_df_out=True)

        assert transformer._bool_cols is None

    def test_fit(self, X_bool: pd.DataFrame):
        """Test the fit method"""

        # create the transformer
        transformer = BoolColumnTransformer(force_df_out=True)

        # fit the transformer
        transformer.fit(X_bool)

        assert transformer._bool_cols == ["bool_col"]

    def test_transform(self, X_bool: pd.DataFrame):
        """Test the transform method"""

        # create the transformer
        transformer = BoolColumnTransformer(force_df_out=True)

        # fit the transformer
        transformer.fit(X_bool)

        # transform
        X_bool_ = transformer.transform(X_bool)

        assert isinstance(X_bool_, pd.DataFrame)
        assert X_bool_.bool_col.nunique() == 2
        vals = X_bool_.bool_col.unique().tolist()
        assert sorted(vals) == [0, 1]

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
        transformer = BoolColumnTransformer(force_df_out=force_df_out)

        # fit the transformer
        transformer.fit(X)
        X_ = transformer.transform(X)

        # type  check
        assert isinstance(X_, _type)
