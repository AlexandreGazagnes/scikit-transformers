"""
TestStandardScaler
"""

import numpy as np
import pandas as pd
import pytest

from sktransf.scaler import StandardScaler


class TestStandardScaler:
    """Test for StandardScaler"""

    def test_X(self, X: pd.DataFrame):
        """Test the fit method"""

        assert isinstance(X, pd.DataFrame)

    def test_init(self):
        """Test the init method"""

        # create the transformer
        transformer = StandardScaler(force_df_out=True)

    def test_fit(self, X: pd.DataFrame):
        """Test the fit method"""

        # create the transformer
        transformer = StandardScaler(force_df_out=True)

        # fit the transformer
        transformer.fit(X)

    def test_transform(self, X: pd.DataFrame):
        """Test the transform method"""

        # create the transformer
        transformer = StandardScaler(force_df_out=True)

        # fit the transformer
        transformer.fit(X)

        # transform
        X_ = transformer.transform(X)

        # assert
        assert isinstance(X_, pd.DataFrame)

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
        transformer = StandardScaler(force_df_out=force_df_out)

        # fit the transformer
        transformer.fit(X)
        X_ = transformer.transform(X)

        # type  check
        assert isinstance(X_, _type)
