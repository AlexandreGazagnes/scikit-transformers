__version__ = "0.1.0"

__doc__ = """
Basic package to enable usefull transformers in scikit-learn pipelines.

First transformer implemented is a LogTransformer, which is a simple wrapper around the numpy log function.
"""

from .logger import LogTransformer
from ._utils import get_titanic, use_case

__all__ = [
    "LogTransformer",
    "get_titanic",
    "use_case",
]
