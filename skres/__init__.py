__version__ = "0.1.1"


import pandas as pd
from sklearn.model_selection import GridSearchCV


class SkRes(pd.DataFrame):
    """Main class of the package"""

    def __init__(self, grid: GridSearchCV) -> pd.DataFrame:
        """initialisation of the class"""

        if not isinstance(grid, GridSearchCV):
            raise TypeError("grid must be a GridSearchCV object")

        # get the results
        res = pd.DataFrame(grid.cv_results_)

        # sort the results
        res.sort_values("mean_test_score", ascending=False, inplace=True)

        # remove the split columns
        cols = [i for i in res.columns if "split" not in i]
        res = res.loc[:, cols]

        # round the results
        res = res.round(4)

        # drop cols
        res.drop(
            columns=["std_fit_time", "std_score_time", "rank_test_score"],
            inplace=True,
            errors="ignore",
        )

        super().__init__(res)


def _use_case():
    """use case of the package"""

    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    base = "https://gist.githubusercontent.com/AlexandreGazagnes/"
    url = base + "9018022652ba0933dd39c9df8a600292/raw/"
    url += "0845ef4c2df4806bb05c8c7423dc75d93e37400f/titanic_train_raw_csv"

    df = pd.read_csv(url)
    y = df.Survived

    X = df.iloc[:, 2:].select_dtypes(include="number")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("estimator", DummyClassifier()),
        ]
    )

    param_grid = {
        "imputer__strategy": ["mean", "median", "most_frequent"],
        "scaler__with_mean": [True, False],
        "estimator__strategy": ["most_frequent", "stratified", "uniform"],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        refit=True,
        return_train_score=True,
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X, y)

    res = SkRes(grid)

    return res
