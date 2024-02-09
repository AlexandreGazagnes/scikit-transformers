"""
BoolColumnTransformer
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..validators import (
    Bool,
    Number,
    manage_columns,
    manage_input,
    manage_nan,
    manage_output,
)

pd.set_option("future.no_silent_downcasting", True)


class BoolColumnTransformer(BaseEstimator, TransformerMixin):
    """Boleanizer for columns with 2 unique values"""

    ignore_nan = Bool()
    force_df_out = Bool()

    def __init__(
        self,
        ignore_nan: bool = True,
        force_df_out: bool = False,
    ) -> None:
        """Init method"""

        self._bool_cols = None
        self.ignore_nan = ignore_nan
        self.force_df_out = force_df_out
        self._values = None
        self.fitted_columns = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y=None,
    ):
        """Fit method"""

        _X = manage_input(X)
        self.fitted_columns = sorted(_X.columns.tolist())

        _X = manage_nan(_X, self.ignore_nan)

        # find bool cols
        self._bool_cols = [col for col in _X.columns if _X[col].nunique() == 2]

        self._values = {}
        for col in self._bool_cols:
            # find values
            values = _X[col].unique()

            # store values
            self._values[col] = {values[0]: 0, values[1]: 1}

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y=None,
    ) -> pd.DataFrame | np.ndarray:
        """Transform method"""

        _X = manage_input(X)
        _X = manage_columns(_X, self.fitted_columns)

        for col in self._bool_cols:
            if col not in _X.columns:
                continue

            # replace values
            dd = self._values[col]

            # TODO : This line will be deprecated in the next version
            _X[col] = _X[col].replace(dd, errors="ignore")

            # Manage errors
            # example for sex 'm' / 'f' in train df and in test df we have 'other'
            _X[col] = _X[col].apply(lambda x: x if x in [0, 1] else np.nan)

        return manage_output(_X, self.force_df_out)
