"""
LogColumnTransformer
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..validators import manage_input, manage_output, manage_negatives


pd.set_option("future.no_silent_downcasting", True)


class LogColumnTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer

    Opt args:
    - threshold: int or float, default=3
        Threshold to apply the log transformation
    - force_df_out: bool, default=False
        If True, the output will be a pd.DataFrame
        if False, the output will be a np.ndarray
    """

    def __init__(
        self,
        threshold: int | float = 3,
        force_df_out: bool = False,
    ):
        """Initialize the transformer"""

        if not isinstance(threshold, (float, int)):
            raise TypeError("threshold must be a float or an integer")

        if not isinstance(force_df_out, (int, bool)):
            raise TypeError("out must be a boolean")

        self.force_df_out = force_df_out
        self.threshold = threshold
        self._log_cols = None
        self._standard_cols = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: None = None,
    ):
        """Fit the transformer (does nothing)

        Pos args:
            X: pd.DataFrame or np.ndarray
                The data to fit
        Opt args:
            y: None
                The target to fit
        """

        _X = manage_input(X)

        manage_negatives(_X)

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
        """Transform the data

        Pos args:
            X: pd.DataFrame or np.ndarray
                The data to transform
        Opt args:
            y: None
                The target to transform
        """

        _X = manage_input(X)

        for col in self._log_cols:
            if not col in _X.columns:
                continue

            _X[col] = np.log1p(_X[col])

        return manage_output(_X, self.force_df_out)
