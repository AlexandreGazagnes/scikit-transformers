"""
TestUseCase
"""

import pytest
from sklearn.model_selection import GridSearchCV

from sktransf.utils import use_case


class TestUseCase:
    """Test use_case function"""

    def test_return_type(self):
        """Test a complete use case"""

        grid = use_case()
        assert isinstance(grid, GridSearchCV)
