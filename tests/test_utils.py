"""
Test _utils module
"""

import pytest

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from sktransf import get_titanic, use_case


class TestGetTitanic:
    """Test get_titanic function"""

    def test_return_type(self):
        X, y = get_titanic()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)


class TestUseCase:
    """Test use_case function"""

    def test_return_type(self):
        """Test return type"""

        grid = use_case()
        assert isinstance(grid, GridSearchCV)
