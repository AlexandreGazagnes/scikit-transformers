__version__ = "0.2.1"

__doc__ = """
Basic package to enable usefull transformers in scikit-learn pipelines.

First transformer implemented is a LogTransformer, which is a simple wrapper
around the numpy log function.
"""

from ._get_titanic import get_titanic
from ._use_case import use_case
from .bool import BoolColumnTransformer
from .unique import DropUniqueColumnTransformer
from .log import LogColumnTransformer


class utils:
    get_titanic = get_titanic
    use_case = use_case


__all__ = [
    "DropUniqueColumnTransformer",
    "BoolColumnTransformer",
    "LogColumnTransformer",
    "utils",
]
