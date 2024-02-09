"""
Test get_titanic function
"""

import logging

import pytest

import pandas as pd

from sktransf.utils import get_titanic


class TestGetTitanic:
    """Test get_titanic function"""

    @pytest.mark.parametrize("only_num", [True, False])
    def test_return_type(self, only_num):
        """Test return type"""

        X, y = get_titanic()

        logging.warning(f"X: {X.head()}")
        logging.warning(f"y: {y.head()}")

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
