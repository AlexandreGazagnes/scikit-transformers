"""
Module for validation of input data and decorators 
"""


def manage_input():
    """
    Decorator to manage input data
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def manage_output():
    """
    Decorator to manage output data
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
