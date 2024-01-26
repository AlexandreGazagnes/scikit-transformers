![image](./.assets/img.png)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)  ![Python](https://img.shields.io/badge/python-3.10.x-green.svg) ![Repo Size](https://img.shields.io/github/repo-size/Sulstice/global-chem)  [![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/) [![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/) ![Coverage](./.assets/cov.svg) ![CI](https://github.com/AlexandreGazagnes/scikit-res/actions/workflows/ci.yaml/badge.svg) ![statics](https://github.com/AlexandreGazagnes/scikit-res/actions/workflows/statics.yaml/badge.svg)

# Scikit-res : Scikit-learn + results

## About

### Context

Very Basic package to store results of ML models
Grid search results are hard to exploit. This package aims to store them in a more convenient way.


## Installation

### Clone the repository

Please clone the repository using the following command :

* for https :
```bash
git clone https://github.com/AlexandreGazagnes/skres.git
```
* for ssh :
```bash
git clone git@github.com:AlexandreGazagnes/skres.git
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
from skres import SkRes
grid_search = GridSearchCV(...)
grid_search.fit(X_train, y_train)
skres = SkRes(grid_search)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License

[GPLv3](LICENSE)