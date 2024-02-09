"""
LogColumnTransformer
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..validators import (
    Bool,
    Number,
    SkewThreshold,
    manage_columns,
    manage_input,
    manage_nan,
    manage_negatives,
    manage_output,
)

pd.set_option("future.no_silent_downcasting", True)


class LogColumnTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer for columns with high skewness"""

    threshold = SkewThreshold()
    ignore_int = Bool()
    force_df_out = Bool()

    def __init__(
        self,
        threshold: int | float = 3,
        ignore_int: bool = False,
        force_df_out: bool = False,
    ) -> None:
        """Init method"""

        if not isinstance(threshold, (float, int)):
            raise TypeError("threshold must be a float or an integer")

        if not isinstance(force_df_out, (int, bool)):
            raise TypeError("out must be a boolean")

        self.force_df_out = force_df_out
        self.ignore_int = ignore_int
        self.threshold = threshold
        self._log_cols = None
        self._standard_cols = None
        self.fitted_columns = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y: None = None,
    ):
        """Fit method"""

        _X = manage_input(X)
        self.fitted_columns = _X.columns.tolist()

        manage_negatives(_X)

        _X = _X.select_dtypes(include=["number"])

        if self.ignore_int:
            _X = _X.select_dtypes(exclude=["int"])

        # compute skew
        skew = _X.skew().round(3).to_dict()

        self._log_cols = []
        self._standard_cols = []

        # use threshold
        for col in skew:
            if skew[col] >= self.threshold:
                self._log_cols.append(col)
            else:
                self._standard_cols.append(col)

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y: None = None,
    ) -> pd.DataFrame | np.ndarray:
        """Transform method"""

        _X = manage_input(X)
        _X = manage_columns(_X, self.fitted_columns)

        for col in self._log_cols:
            if not col in _X.columns:
                continue

            _X[col] = np.log1p(_X[col])

        return manage_output(_X, self.force_df_out)
