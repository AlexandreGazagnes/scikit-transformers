"""
Module for validators and decorators 
"""

import logging
import pandas as pd
import numpy as np


def manage_input(X):
    """
    Decorator to manage input data
    """

    # def decorator(func):
    #     def wrapper(*args, **kwargs):
    #         return func(*args, **kwargs)

    #     return wrapper

    # return decorator

    # manage dtypes
    if not isinstance(X, (pd.DataFrame, np.ndarray, list)):
        raise TypeError(
            f"X must be a (pd.DataFrame, np.ndarray, list), received {type(X)}"
        )

    # force cast df
    try:
        _X = pd.DataFrame(X).copy()
    except Exception as e:
        raise Exception(f"Impossible to make df/np.array : {e}")

    return _X


def manage_output(_X: pd.DataFrame, force_df_out: bool = True):
    # """
    # Decorator to manage output data
    # """

    # def decorator(func):
    #     def wrapper(*args, **kwargs):
    #         return func(*args, **kwargs)

    #     return wrapper

    # return decorator

    return pd.DataFrame(_X) if force_df_out else _X.values


def manage_nan(_X: pd.DataFrame, ignore_nan: bool = True):
    # """
    # Decorator to manage nan values
    # """
    # def decorator(func):
    #     def wrapper(*args, **kwargs):
    #         return func(*args, **kwargs)

    #     return wrapper

    # return decorator

    return _X.dropna(axis=0).copy() if ignore_nan else _X


def manage_negatives(_X: pd.DataFrame):

    # no negative values
    min_ = _X.select_dtypes("number").min()
    if min_.min() < 0:
        raise ValueError(
            f"All values must be >=0 to apply log : {min_.to_dict()}"
        )
