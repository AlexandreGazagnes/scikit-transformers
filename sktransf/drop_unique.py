"""
Unique drop transformer
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
    ) -> None:
        """Init method"""

        self.unique_cols = unique_cols

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

        return _X
