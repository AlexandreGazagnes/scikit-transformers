# Usage


## Basic usage

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

## Using common transformers

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

## Using a pipeline with a scikit-learn model

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

## Notebooks

For more specific information, please refer to the notebooks: 

* Transformers : 
  * [LogColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/transformer/LogColumnTransformer.ipynb)
  * [BoolColumnTransformer notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/transformer/BoolColumnTransformer.ipynb)
* Selectors : 
  * [DropUniqueColumnSelector notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/selector/DropUniqueColumnSelector.ipynb)
  * [DropSkuColumnSelector notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/selector/DropSkuColumnSelector.ipynb)
* Pipelines :
  * [Pipelines notebook](https://github.com/AlexandreGazagnes/scikit-transformers/blob/main/docs/notebooks/Pipelines.ipynb)

