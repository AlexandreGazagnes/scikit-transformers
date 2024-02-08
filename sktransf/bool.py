"""
Boolean column transformer
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
        bool_cols: list[str] | None = None,
        force_df_out: bool = False,
    ) -> None:
        """init method"""

        self.bool_cols = bool_cols
        self.force_df_out = force_df_out

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y=None,
    ):
        """Fit method"""

        if self.bool_cols:
            return self

        # manage df
        if not isinstance(X, pd.DataFrame):
            _X = pd.DataFrame(X)
        else:
            _X = X.copy()

        # find bool cols
        self.bool_cols = [col for col in _X.columns if _X[col].nunique() == 2]

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

        for col in self.bool_cols:
            # check if column exists
            if col not in _X.columns:
                continue

            # find values
            values = _X[col].unique()

            # enumerate
            dd = {values[0]: 0, values[1]: 1}

            # replace
            _X[col] = _X[col].replace(dd)

        if self.force_df_out:
            return pd.DataFrame(_X)

        return _X.values
