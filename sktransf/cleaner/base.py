"""
BaseDataCleaner
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..selector import DropSkuColumnSelector, DropUniqueColumnSelector
from ..transformer import BoolColumnTransformer, LogColumnTransformer
from ..validators import (
    Bool,
    Number,
    SkewThreshold,
    SkuThreshold,
    manage_columns,
    manage_input,
    manage_nan,
    manage_output,
)

pd.set_option("future.no_silent_downcasting", True)


class BaseDataCleaner(BaseEstimator, TransformerMixin):
    """Data cleaner"""

    skew_threshold = SkewThreshold()
    sku_threshold = SkuThreshold()
    ignore_int = Bool()
    ignore_float = Bool()
    ignore_nan = Bool()
    force_df_out = Bool()

    def __init__(
        self,
        skew_threshold: float = 1.5,
        sku_threshold: float = 0.99,
        ignore_int: bool = False,
        ignore_float: bool = True,
        ignore_nan: bool = True,
        force_df_out: bool = False,
    ) -> None:
        """Init method"""

        self.skew_threshold = skew_threshold
        self.ignore_int = ignore_int
        self.ignore_float = ignore_float
        self.sku_threshold = sku_threshold
        self.ignore_nan = ignore_nan
        self.force_df_out = force_df_out
        self.fitted_columns = None

        self.sku = DropSkuColumnSelector(
            threshold=self.sku_threshold,
            ignore_nan=self.ignore_nan,
            ignore_float=self.ignore_float,
            force_df_out=True,
        )
        self.unique = DropUniqueColumnSelector(
            ignore_nan=self.ignore_nan,
            force_df_out=True,
        )
        self.bool = BoolColumnTransformer(
            ignore_nan=self.ignore_nan,
            force_df_out=True,
        )
        self.log = LogColumnTransformer(
            threshold=self.skew_threshold,
            ignore_int=self.ignore_int,
            force_df_out=True,
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y=None,
    ):
        """Fit method"""

        _X = manage_input(X)
        self.fitted_columns = sorted(_X.columns.tolist())

        _X = manage_nan(_X, self.ignore_nan)

        self.sku.fit(_X)
        self.unique.fit(_X)
        self.bool.fit(_X)
        self.log.fit(_X.select_dtypes(include=[np.number]))

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y=None,
    ) -> pd.DataFrame | np.ndarray:
        """Transform method"""

        _X = manage_input(X)
        _X = manage_columns(_X, self.fitted_columns)

        _X = self.sku.transform(_X).copy()
        _X = self.unique.transform(_X).copy()
        _X = self.bool.transform(_X).copy()
        _X = self.log.transform(_X).copy()

        return manage_output(_X, self.force_df_out)
