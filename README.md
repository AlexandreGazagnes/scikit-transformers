![image](./.assets/img.png)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)  ![Python](https://img.shields.io/badge/python-3.10.x-green.svg) ![Repo Size](https://img.shields.io/github/repo-size/Sulstice/global-chem)  [![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/) [![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/) ![Coverage](./.assets/cov.svg) ![CI](https://github.com/AlexandreGazagnes/scikit-transformers/actions/workflows/ci.yaml/badge.svg) ![statics](https://github.com/AlexandreGazagnes/scikit-transformers/actions/workflows/statics.yaml/badge.svg)

# Scikit-transformers : Scikit-learn + transformers

## About

### Context

Very Basic package to enable usefull transformers in scikit-learn pipelines.


## Installation

### Clone the repository

Please clone the repository using the following command :

* for https :
```bash
git clone https://github.com/AlexandreGazagnes/scikit-transformers.git
```
* for ssh :
```bash
git clone git@github.com:AlexandreGazagnes/scikit-transformers.git
```

### Install the dependencies

The project uses [Poetry](https://python-poetry.org/) to manage its dependencies. Please install it using the following command :

```bash
pip install poetry
```

Then, please install the dependencies using the following command :

```bash
poetry install
```

Alternatively, you can install the dependencies using the following command :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r .utils/requirements-dev.txt
```

## Usage

```python
    import numpy as np
    import pandas as pd

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import KNNImputer, SimpleImputer

    base = "https://gist.githubusercontent.com/AlexandreGazagnes/"
    url = base + "9018022652ba0933dd39c9df8a600292/raw/"
    url += "0845ef4c2df4806bb05c8c7423dc75d93e37400f/titanic_train_raw_csv"

    df = pd.read_csv(url)
    y = df.Survived

    X = df.iloc[:, 2:].select_dtypes(include="number")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("logger", LogTransformer()),
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
        ]
    )

    param_grid = {
        "logger__threshold": [1, 1.5, 3],
        "scaler": [StandardScaler(), "passthrough"],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        refit=True,
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X, y)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License

[GPLv3](LICENSE)