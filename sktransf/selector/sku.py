"""
DropSkuColumnSelector
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..validators import manage_input, manage_output, manage_nan

pd.set_option("future.no_silent_downcasting", True)


class DropSkuColumnSelector(BaseEstimator, TransformerMixin):
    """Drops columns too many unique values"""

    def __init__(
        self,
        threshold: float = 0.99,
        ignore_float: bool = True,
        ignore_nan: bool = True,
        force_df_out: bool = False,
    ) -> None:
        """Init method"""

        self.threshold = threshold
        self.ignore_float = ignore_float
        self.ignore_nan = ignore_nan
        self.force_df_out = force_df_out
        self._sku_cols = None

    def fit(
        self,
        X: pd.DataFrame | list | np.ndarray,
        y=None,
    ):
        """Fit method"""

        _X = manage_input(X)

        _X = manage_nan(_X, self.ignore_nan)

        if self.ignore_float:
            _X = _X.select_dtypes(exclude="float").copy()

        percentages = [
            (col, _X[col].nunique() / len(_X)) for col in _X.columns
        ]

        self._sku_cols = [
            col for col, perc in percentages if perc >= self.threshold
        ]

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y=None,
    ) -> pd.DataFrame:
        """Transform method"""

        _X = manage_input(X)

        _X = _X.drop(columns=self._sku_cols, errors="ignore")

        return manage_output(_X, self.force_df_out)


def test():

    from sktransf.utils import get_titanic
    from sktransf import SkuColumnTransformer

    X, y = get_titanic(only_num=False)

    X_ = SkuColumnTransformer(force_df_out=True).fit(X).transform(X)
