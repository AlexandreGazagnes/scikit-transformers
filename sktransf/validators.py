"""
Module for validators and decorators 
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def manage_input(X):
    """Manage input data type"""

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
    """Manage output data type"""

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
    """Manage nan values in input data"""

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
    """Manage negative values in input data"""

    # no negative values
    min_ = _X.select_dtypes("number").min()
    if min_.min() < 0:
        raise ValueError(
            f"All values must be >=0 to apply log : {min_.to_dict()}"
        )


def manage_columns(
    _X: pd.DataFrame,
    fitted_columns: list,
) -> pd.DataFrame:
    """Manage columns drift between fit and transform"""

    columns = sorted(_X.columns.tolist())
    if not columns == fitted_columns:
        logging.warning(
            "fColumns in input data are different from fitted columns fit : {fitted_columns} != transform : {columns}"
        )

    # drop cols
    drop_cols = [i for i in columns if i not in fitted_columns]
    _X = _X.drop(columns=drop_cols, errors="ignore")
    logging.warning(f"Columns dropped: {drop_cols}")

    # create nan cols
    nan_cols = [i for i in fitted_columns if i not in columns]
    _X.loc[:, nan_cols] = np.nan
    logging.warning(f"Columns with NaN added: {nan_cols}")

    return _X


class Validator(ABC):
    """Abstract class for validators"""

    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass


class Number(Validator):
    """Validator for numbers"""

    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Expected {self.private_name[1:]} to be an int or float : received {type(value)} for {value}"
            )


class Df(Validator):
    """Validator for dataframes"""

    def validate(self, value):
        if not isinstance(value, (pd.DataFrame, list, np.array)):
            raise TypeError(
                f"Expected {self.private_name[1:]} to be an int or float : received {type(value)} for {value}"
            )

        try:
            pd.DataFrame(value)
        except Exception as e:
            raise Exception(
                f"Impossible to make df/np.array for {self.private_name[1:]} received {type(value)} for {value} : {e}"
            )


class Bool(Validator):
    """Validator for bools"""

    def validate(self, value):
        if not isinstance(value, (bool, int)):
            raise TypeError(
                f"Expected {self.private_name[1:]} to be a bool : received {type(value)} for {value}"
            )
        try:
            bool(value)
        except Exception as e:
            raise ValueError(
                f"Impossible to cast {self.private_name[1:]} to bool received {type(value)} for {value} : {e}"
            )


class SkuThreshold(Validator):
    """Threshold validator for float or int between 0 and 1 or greater than 1"""

    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Expected {self.private_name[1:]} to be an int or float : received {type(value)} for {value}"
            )

        if isinstance(value, float) and (value < 0 or value > 1):
            raise ValueError(
                f"Expected {self.private_name[1:]} float must be between 0 and 1 : received {type(value)} for {value}"
            )

        if isinstance(value, int) and value < 1:
            raise ValueError(
                f"Expected {self.private_name[1:]} int must be greater than 1 : received {type(value)} for {value}"
            )


class SkewThreshold(Validator):
    """Threshold validator for float or int between 0 and 1 or greater than 1"""

    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Expected {self.private_name[1:]} to be an int or float : received {type(value)} for {value}"
            )

        if isinstance(value, int) and value < 0:
            raise ValueError(
                f"Expected {self.private_name[1:]} int must be greater than 0 : received {type(value)} for {value}"
            )
