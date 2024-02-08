"""
Unique drop and Bool transformer
"""

import numpy as np
import pandas as pd

from typing import List

from sklearn.base import BaseEstimator, TransformerMixin


class DropUniqueColumnTransformer(BaseEstimator, TransformerMixin):
    """Drops columns with only one unique value

    Agrs :
        Optional :
            - unique_cols : List[str] | None : list of columns to drop if they have only one unique value
            default : None => will be found during fit
    """

    def __init__(self, unique_cols: List[str] | None = None) -> None:
        """init method"""

        self.unique_cols = unique_cols

    def fit(self, X: pd.DataFrame | list | np.array, y=None):
        """fit"""

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

    def transform(self, X, y=None) -> pd.DataFrame:
        """transform"""

        # manage df
        if not isinstance(X, pd.DataFrame):
            _X = pd.DataFrame(X)
        else:
            _X = X.copy()

        # drop it
        _X = _X.drop(columns=self.unique_cols, errors="ignore")

        return _X


class BoolColumnTransformer(BaseEstimator, TransformerMixin):
    """Boleanizer for columns with 2 unique values

    Args :
        Optional :
            - bool_cols : List[str] | None : list of columns to booleanize
            default : None => will be found during fit
    """

    def __init__(self, bool_cols: List[str] | None = None) -> None:
        """init method"""

        self.bool_cols = bool_cols

    def fit(self, X: pd.DataFrame | list | np.array, y=None):
        """fit"""

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

    def transform(self, X, y=None) -> pd.DataFrame:
        """transform"""

        # manage df
        if not isinstance(X, pd.DataFrame):
            _X = pd.DataFrame(X)
        else:
            _X = X.copy()

        for col in self.bool_cols:

            # check if column exists
            if not col in _X.columns:
                continue

            # find values
            values = _X[col].unique()

            # enumerate
            dd = {values[0]: 0, values[1]: 1}

            # replace
            _X[col] = _X[col].replace(dd)

        return _X
