"""
DropSkuColumnSelector
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..validators import (
    Bool,
    Number,
    SkuThreshold,
    manage_columns,
    manage_input,
    manage_nan,
    manage_output,
)

pd.set_option("future.no_silent_downcasting", True)


class DropSkuColumnSelector(BaseEstimator, TransformerMixin):
    """Drops columns too many unique values"""

    threshold = SkuThreshold()
    ignore_float = Bool()
    ignore_nan = Bool()
    force_df_out = Bool()

    def __init__(
        self,
        threshold: float | int = 0.99,
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
        self.fitted_columns = None

    def fit(
        self,
        X: pd.DataFrame | list | np.ndarray,
        y=None,
    ):
        """Fit method"""

        _X = manage_input(X)
        self.fitted_columns = _X.columns.tolist()

        _X = manage_nan(_X, self.ignore_nan)

        if self.ignore_float:
            _X = _X.select_dtypes(exclude="float").copy()

        if isinstance(self.threshold, float):
            percentages = [
                (col, _X[col].nunique() / len(_X)) for col in _X.columns
            ]
            self._sku_cols = [
                col for col, perc in percentages if perc >= self.threshold
            ]

        elif isinstance(self.threshold, int):
            _X = _X.select_dtypes(exclude=["number"]).copy()
            self._sku_cols = [
                col
                for col in _X.columns
                if _X[col].nunique() >= self.threshold
            ]

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y=None,
    ) -> pd.DataFrame | np.ndarray:
        """Transform method"""

        _X = manage_input(X)
        _X = manage_columns(_X, self.fitted_columns)

        _X = _X.drop(columns=self._sku_cols, errors="ignore")

        return manage_output(_X, self.force_df_out)
