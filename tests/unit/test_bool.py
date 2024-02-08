import pandas as pd

from sktransf import BoolColumnTransformer


class TestBoolColumnTransformer:
    """Test for BoolColumnTransformer"""

    def test_X(self, X_bool: pd.DataFrame):
        """Test X_bool"""

        assert isinstance(X_bool, pd.DataFrame)

        assert X_bool.bool_col.nunique() == 2

    def test_init(self):
        """Test the init method"""

        # create the transformer
        transformer = BoolColumnTransformer()

        assert transformer.bool_cols is None

    def test_fit(self, X_bool: pd.DataFrame):
        """Test the fit method"""

        # create the transformer
        transformer = BoolColumnTransformer()

        # fit the transformer
        transformer.fit(X_bool)

        assert transformer.bool_cols == ["bool_col"]

    def test_transform(self, X_bool: pd.DataFrame):
        """Test the transform method"""

        # create the transformer
        transformer = BoolColumnTransformer()

        # fit the transformer
        transformer.fit(X_bool)

        # transform
        X_bool_ = transformer.transform(X_bool)

        assert isinstance(X_bool_, pd.DataFrame)
        assert X_bool_.bool_col.nunique() == 2
        vals = X_bool_.bool_col.unique().tolist()
        assert sorted(vals) == [0, 1]
