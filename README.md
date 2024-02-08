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

Basic package to enable usefull transformers in scikit-learn pipelines.

First transformer implemented is a LogTransformer, which is a simple wrapper around the numpy log function.


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

from sktransf import LogTransformer

df = pd.DataFrame(
    { "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
)

logger = LogTransformer()
logger.fit_transform(df)
df_transf = logger.transform(df)
```

Using common transformers : 

```python
import pandas as pd

from sktransf import LogTransformer, DropUniqueColumnTransformer, BoolColumnTransformer

df = pd.DataFrame(
    { "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
)

df_bool = BoolColumnTransformer().fit_transform(df)
df_unique = DropUniqueColumnTransformer().fit_transform(df)
df_logged = LogTransformer().fit_transform(df)
```

Using a pipeline : 

```python

import pandas as pd
from sklearn.pipeline import Pipeline

from sktransf import LogTransformer, DropUniqueColumnTransformer, BoolColumnTransformer

pipe = Pipeline([
    ('bool', BoolColumnTransformer()),
    ('unique', DropUniqueColumnTransformer()),
    ('log', LogTransformer())
])

df = pd.DataFrame(
    { "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
)

df_transf = pipe.fit_transform(df)
```

Using a pipeline with a scikit-learn model : 

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sktransf import LogTransformer, DropUniqueColumnTransformer, BoolColumnTransformer

pipe = Pipeline([
    ('bool', BoolColumnTransformer()),
    ('unique', DropUniqueColumnTransformer()),
    ('log', LogTransformer()),
    ('model', LinearRegression())
])

X = pd.DataFrame(
    { "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
)

y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

pipe.fit(X, y)

y_pred = pipe.predict(X)
```


## Documentation

For more specific information, please refer to the notebooks: 
- [Pipelines notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/Pipelines.ipynb)
- [BoolColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/BoolColumnTransformer.ipynb)
- [DropUniqueColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/DropUniqueColumnTransformer.ipynb)
- [LogColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/LogColumnTransformer.ipynb)


<!-- For more specific use case, please refer to this [notebook](docs/detailed_example.ipynb). -->

<!-- For more detailed information, please refer to the [documentation](https://alexandregazagnes.github.io/scikit-transformers/). -->

A complete documentation is be available on the  [github page](https://alexandregazagnes.github.io/scikit-transformers/).


## Changelog, Releases and Roadmap

Please refer to the [changelog](https://alexandregazagnes.github.io/scikit-transformers/CHANGELOG/) page for more information.


## Contributing

Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.

For more information, please refer to the [contributing](https://alexandregazagnes.github.io/scikit-transformers/CONTRIBUTING/) page.


## License

[GPLv3](https://raw.githubusercontent.com/AlexandreGazagnes/scikit-transformers/main/LICENSE)
