"""
DropUniqueColumnSelector
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..validators import manage_input, manage_nan, manage_output

pd.set_option("future.no_silent_downcasting", True)


class DropUniqueColumnSelector(BaseEstimator, TransformerMixin):
    """Drops columns with only one unique value

    Agrs :
        Optional :
            - unique_cols : List[str] | None : list of columns to drop if
            they have only one unique value
            default : None => will be found during fit
    """

    def __init__(
        self,
        ignore_nan: bool = True,
        force_df_out: bool = False,
    ) -> None:
        """Init method"""

        self._unique_cols = None
        self.ignore_nan = ignore_nan
        self.force_df_out = force_df_out

    def fit(
        self,
        X: pd.DataFrame | list | np.ndarray,
        y=None,
    ):
        """Fit method"""

        _X = manage_input(X)

        _X = manage_nan(_X, self.ignore_nan)

        # find bool cols
        self._unique_cols = [
            col for col in _X.columns if _X[col].nunique() == 1
        ]

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y=None,
    ) -> pd.DataFrame:
        """Transform method"""

        _X = manage_input(X)

        # drop it
        _X = _X.drop(columns=self._unique_cols, errors="ignore")

        return manage_output(_X, self.force_df_out)
