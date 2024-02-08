"""
Test _utils module
"""

from sklearn.model_selection import GridSearchCV

from sktransf._use_case import use_case


class TestUseCase:
    """Test use_case function"""

    def test_return_type(self):
        """Test a complete use case"""

        grid = use_case()
        assert isinstance(grid, GridSearchCV)
