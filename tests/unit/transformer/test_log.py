"""
Test LogColumnTransformer
"""

import logging

import numpy as np
import pandas as pd
import pytest

from sktransf.transformer import LogColumnTransformer


class TestLogColumnTransformer:
    """Test for LogColumnTransformer"""

    def test_init(self):
        """Test the init method"""

        # create the transformer
        transformer = LogColumnTransformer()

        assert transformer.threshold == 3
        assert transformer.force_df_out is False

    def test_fit(self, X: pd.DataFrame):
        """Test the fit method"""

        # create the transformer
        transformer = LogColumnTransformer()

        # fit the transformer
        transformer.fit(X)

        # skew = {
        #     "Pclass": -0.63,
        #     "Age": 0.39,
        #     "SibSp": 3.7,
        #     "Parch": 2.75,
        #     "Fare": 4.79,
        # }

        assert transformer._log_cols == ["SibSp", "Fare"]

    def test_transform_3(self, X: pd.DataFrame):
        """Test the transform method"""

        # create the transformer
        transformer = LogColumnTransformer(force_df_out=True)

        # fit the transformer
        transformer.fit(X)

        assert transformer._log_cols == ["SibSp", "Fare"]

        # transform
        X_ = transformer.transform(X)

        assert isinstance(X_, pd.DataFrame)

        logging.warning(X_.head())

        assert X_.SibSp.values.tolist() != X.SibSp.values.tolist()
        assert X_.Fare.values.tolist() != X.Fare.values.tolist()
        assert X_.Pclass.values.tolist() == X.Pclass.values.tolist()

    @pytest.mark.parametrize(
        "threshold,cols",
        [(0, ["Age", "Fare", "Parch", "PassengerId", "SibSp"]), (100, [])],
    )
    def test_treshold(self, X: pd.DataFrame, threshold: int, cols: list):
        """Test the threshold attribute"""

        # create the transformer
        transformer = LogColumnTransformer(
            force_df_out=False,
            threshold=threshold,
        )

        # fit the transformer
        transformer.fit(X)

        assert sorted(transformer._log_cols) == cols

        # transform
        X_ = transformer.transform(X)

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
        transformer = LogColumnTransformer(force_df_out=force_df_out)

        # fit the transformer
        transformer.fit(X)
        X_ = transformer.transform(X)

        # type  check
        assert isinstance(X_, _type)
