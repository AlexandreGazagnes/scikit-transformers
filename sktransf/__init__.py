"""
Init file for sktransf package.
"""

# import importlib.metadata

# __version__ = importlib.metadata.metadata(__package__)["version"]
# __description__ = importlib.metadata.metadata(__package__)["description"]
# __doc__ = importlib.metadata.metadata(__package__)["description"]

__version__ = "0.2.2"
__description__ = "Very usefull package to enable and provide custom transformers such as LogColumnTransformer, BoolColumnTransformers and others fancy transformers."
__doc__ = """
Very usefull package to enable and provide custom transformers such as LogColumnTransformer, BoolColumnTransformers and others fancy transformers.

It was created to provide a simple way to use custom transformers in scikit-learn pipelines, and allow to use them in a scikit-learn model, using GridSearchCV for testing and tuning hyperparameters.

The starting point was to provide a simple LogColumnTransformer, which is a simple wrapper around the numpy log function, making possible to use a skew threshold to apply the log transformation only on columns with a skew superior to a given threshold inside a GridSearchCV.
"""

import utils
from .bool import BoolColumnTransformer
from .unique import DropUniqueColumnTransformer
from .log import LogColumnTransformer


__all__ = [
    "DropUniqueColumnTransformer",
    "BoolColumnTransformer",
    "LogColumnTransformer",
    "utils",
]
