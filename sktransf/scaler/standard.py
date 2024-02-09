"""
BoolColumnTransformer
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler as _StandardScaler

from ..validators import manage_input, manage_nan, manage_output

pd.set_option("future.no_silent_downcasting", True)


class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
        force_df_out: bool = False,
        ignore_nan: bool = True,
    ) -> None:
        # super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

        self.sca = _StandardScaler(
            copy=copy, with_mean=with_mean, with_std=with_std
        )
        self.ignore_nan = ignore_nan
        self.force_df_out = force_df_out

    def fit(self, X, y=None):
        _X = manage_input(X)
        _X = manage_nan(_X, self.ignore_nan)

        self.sca.fit(_X)

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y=None,
    ) -> pd.DataFrame | np.ndarray:
        """Transform method"""

        _X = manage_input(X)

        _X = self.sca.transform(_X)

        _X = pd.DataFrame(_X, columns=X.columns)

        return manage_output(_X, self.force_df_out)
