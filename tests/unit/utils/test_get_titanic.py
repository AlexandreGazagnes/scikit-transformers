"""
Test get_titanic function
"""

import logging

import pandas as pd
import pytest

from sktransf.utils import get_titanic


class TestGetTitanic:
    """Test get_titanic function"""

    @pytest.mark.parametrize("only_num", [True, False])
    def test_return_type(self, only_num):
        """Test return type"""

        X, y = get_titanic()

        logging.debug(f"X: {X.head()}")
        logging.debug(f"y: {y.head()}")

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
