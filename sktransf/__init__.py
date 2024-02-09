"""
Init file for sktransf package.
"""

# import importlib.metadata

# __version__ = importlib.metadata.metadata(__package__)["version"]
# __description__ = importlib.metadata.metadata(__package__)["description"]
# __doc__ = importlib.metadata.metadata(__package__)["description"]

__version__ = "0.3.0"
__description__ = "scikit-transformers is a very usefull package to enable and provide custom transformers such as LogColumnTransformer, BoolColumnTransformers and others fancy transformers."
__doc__ = """

**scikit-transformers** is a very usefull package to enable and provide custom transformers such as ```LogColumnTransformer```, ```BoolColumnTransformers``` and others fancy transformers.

It was created to provide a simple way to use custom transformers in ```scikit-learn``` pipelines, and allow to use them in a ```scikit-learn ```model, using ```GridSearchCV``` for testing and tuning hyperparameters.

The starting point was to provide a simple ```LogColumnTransformer```, which is a simple wrapper around the numpy log function, making possible to use a skew threshold to apply the log transformation only on columns with a skew superior to a given threshold.

With ```scikit-transformers```, it is now possible to use this ```LogColumnTransformer``` in transformer in a ```GridSearchCV``` using a skew threshold as hyperparameter to find what columns are good to log or not.

```LogColumnTransformer``` is one of the many transformers implemented in ```scikit-transformers```.
"""

from . import cleaner, scaler, selector, transformer, utils

__all__ = [
    "transformer",
    "selector",
    "scaler",
    "cleaner",
    "utils",
]
