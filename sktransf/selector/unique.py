"""
DropUniqueColumnSelector
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


class DropUniqueColumnSelector(BaseEstimator, TransformerMixin):
    """Drops columns with only one unique value"""

    ignore_nan = Bool()
    force_df_out = Bool()

    def __init__(
        self,
        ignore_nan: bool = True,
        force_df_out: bool = False,
    ) -> None:
        """Init method"""

        self._unique_cols = None
        self.ignore_nan = ignore_nan
        self.force_df_out = force_df_out
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

        # find bool cols
        self._unique_cols = [
            col for col in _X.columns if _X[col].nunique() == 1
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

        # drop it
        _X = _X.drop(columns=self._unique_cols, errors="ignore")

        return manage_output(_X, self.force_df_out)
