"""
Test get_titanic function
"""

import pandas as pd

from sktransf._get_titanic import get_titanic


class TestGetTitanic:
    """Test get_titanic function"""

    def test_return_type(self):
        """Test return type"""

        X, y = get_titanic()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
