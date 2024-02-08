"""
Unique drop transformer
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


pd.set_option("future.no_silent_downcasting", True)


class DropUniqueColumnTransformer(BaseEstimator, TransformerMixin):
    """Drops columns with only one unique value

    Agrs :
        Optional :
            - unique_cols : List[str] | None : list of columns to drop if
            they have only one unique value
            default : None => will be found during fit
    """

    def __init__(
        self,
        unique_cols: list[str] | None = None,
        force_df_out: bool = False,
    ) -> None:
        """Init method"""

        self.unique_cols = unique_cols
        self.force_df_out = force_df_out

    def fit(
        self,
        X: pd.DataFrame | list | np.ndarray,
        y=None,
    ):
        """Fit method"""

        if self.unique_cols:
            return self

        # manage df
        if not isinstance(X, pd.DataFrame):
            _X = pd.DataFrame(X)
        else:
            _X = X.copy()

        # find bool cols
        self.unique_cols = [
            col for col in _X.columns if _X[col].nunique() == 1
        ]

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y=None,
    ) -> pd.DataFrame:
        """Transform method"""

        # manage df
        if not isinstance(X, pd.DataFrame):
            _X = pd.DataFrame(X)
        else:
            _X = X.copy()

        # drop it
        _X = _X.drop(columns=self.unique_cols, errors="ignore")

        if self.force_df_out:
            return pd.DataFrame(_X)

        return _X.values
