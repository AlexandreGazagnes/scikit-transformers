# Usage


## Most basic usage

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


## Using common transformers

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

## Using a pipeline

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

## Using a pipeline with a scikit-learn model

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


## Notebooks

For more specific information, please refer to the notebooks: 
- [Pipelines notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/Pipelines.ipynb)
- [BoolColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/BoolColumnTransformer.ipynb)
- [DropUniqueColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/DropUniqueColumnTransformer.ipynb)
- [LogColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/LogColumnTransformer.ipynb)
