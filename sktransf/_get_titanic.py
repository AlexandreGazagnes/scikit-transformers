"""
Utils get_titanic
"""

import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


def get_titanic() -> tuple[pd.DataFrame, pd.Series]:
    """Get titanic dataset from github gist"""

    base = "https://gist.githubusercontent.com/AlexandreGazagnes/"
    url = base + "9018022652ba0933dd39c9df8a600292/raw/"
    url += "0845ef4c2df4806bb05c8c7423dc75d93e37400f/titanic_train_raw_csv"

    df = pd.read_csv(url)
    y = df.Survived

    X = df.iloc[:, 2:].select_dtypes(include="number")

    return X, y
