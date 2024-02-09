"""
BoolColumnTransformer
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..validators import manage_input, manage_nan, manage_output

pd.set_option("future.no_silent_downcasting", True)


class BoolColumnTransformer(BaseEstimator, TransformerMixin):
    """Boleanizer for columns with 2 unique values

    Args :
        Optional :
            - bool_cols : List[str] | None : list of columns to booleanize
            default : None => will be found during fit
    """

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

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y=None,
    ):
        """Fit method"""

        _X = manage_input(X)
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

        for col in self._bool_cols:
            # check if column exists
            if col not in _X.columns:
                continue

            # replace values
            dd = self._values[col]

            # TODO : This line will be deprecated in the next version
            _X[col] = _X[col].replace(dd)

        return manage_output(_X, self.force_df_out)
