![image](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/assets/img/img.png?raw=true)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-3.10.x-green.svg)
![Repo Size](https://img.shields.io/github/repo-size/AlexandreGazagnes/scikit-transformers)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
![Coverage](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/assets/img/cov.svg?raw=true)
![Tests](https://github.com/AlexandreGazagnes/scikit-transformers/actions/workflows/tests.yaml/badge.svg)
![Statics](https://github.com/AlexandreGazagnes/scikit-transformers/actions/workflows/statics.yaml/badge.svg)
![Doc](https://github.com/AlexandreGazagnes/scikit-transformers/actions/workflows/docs.yaml/badge.svg)
![Pypi](https://github.com/AlexandreGazagnes/scikit-transformers/actions/workflows/publish.yaml/badge.svg)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/AlexandreGazagnes/scikit-transformers)

# Scikit-transformers : Scikit-learn + Custom transformers


## About

**scikit-transformers** is a very usefull package to enable and provide custom transformers such as ```LogColumnTransformer```, ```BoolColumnTransformers``` and others fancy transformers.

It was created to provide a simple way to use custom transformers in ```scikit-learn``` pipelines, and allow to use them in a ```scikit-learn ```model, using ```GridSearchCV``` for testing and tuning hyperparameters.

The starting point was to provide a simple ```LogColumnTransformer```, which is a simple wrapper around the numpy log function, making possible to use a skew threshold to apply the log transformation only on columns with a skew superior to a given threshold.

With ```scikit-transformers```, it is now possible to use this ```LogColumnTransformer``` in transformer in a ```GridSearchCV``` using a skew threshold as hyperparameter to find what columns are good to log or not.

```LogColumnTransformer``` is one of the many transformers implemented in ```scikit-transformers```.



## Installation

Using regular pip and venv tools :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install scikit-transformers
```


## Usage

For a very basic usage :
```python
import pandas as pd

from sktransf.trasnformer import LogColumnTransformer

df = pd.DataFrame(
    { "a": range(10),
      "b": range(10)
    }
)

logger = LogColumnTransformer()
logger.fit_transform(df)
df_transf = logger.transform(df)
```

Using common transformers : 

```python
import pandas as pd

from sktransf.transformer import LogColumnTransformer, BoolColumnTransformer
from sktransf.selector import DropUniqueColumnSelector

df = pd.DataFrame(
    { "a": range(10),
      "b": range(10)
    }
)

df_bool = BoolColumnTransformer().fit_transform(df)
df_unique = DropUniqueColumnTransformer().fit_transform(df)
df_logged = LogColumnTransformer().fit_transform(df)
```

Using a pipeline with a scikit-learn model : 

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sktransf.transformer import LogColumnTransformer, BoolColumnTransformer
from sktransf.selector import DropUniqueColumnSelector

pipe = Pipeline([
    ('bool', BoolColumnTransformer()),
    ('unique', DropUniqueColumnTransformer()),
    ('log', LogColumnTransformer()),
    ('model', LinearRegression())
])

X = pd.DataFrame(
    { "a": range(10),
      "b": range(10)
    }
)

y = range(10)

pipe.fit(X, y)

y_pred = pipe.predict(X)
```


## Documentation

For more specific information, please refer to the notebooks: 

* Transformers : 
  * [LogColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/transformer/LogColumnTransformer.ipynb)
  * [BoolColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/transformer/BoolColumnTransformer.ipynb)
* Selectors : 
  * [DropUniqueColumnSelector notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/selector/DropUniqueColumnSelector.ipynb)
  * [DropSkuColumnSelector notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/selector/DropSkuColumnSelector.ipynb)
* Pipelines :
  * [Pipelines notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/Pipelines.ipynb)


A complete documentation is be available on the  [github page](https://alexandregazagnes.github.io/scikit-transformers/).


## Changelog, Releases and Roadmap

Please refer to the [changelog](https://alexandregazagnes.github.io/scikit-transformers/CHANGELOG/) page for more information.


## Contributing

Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.

For more information, please refer to the [contributing](https://alexandregazagnes.github.io/scikit-transformers/CONTRIBUTING/) page.


## License

[GPLv3](https://raw.githubusercontent.com/AlexandreGazagnes/scikit-transformers/main/LICENSE)
