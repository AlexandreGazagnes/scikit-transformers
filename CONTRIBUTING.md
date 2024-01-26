# Contributing

## Local development

- The complete test suite depends on having at least the following installed
  (possibly not a complete list)
  - git (Version 2.24.0 or above is required )
  - python3.10.x (Required by a test which checks different python versions)
  - tox (or venv)
  - poetry, pip, pipeenv, virtualenv, or similar

### Setting up an environment

The project uses [Poetry](https://python-poetry.org/) to manage its dependencies. Please install it using the following command :

```bash
pip install poetry
```

Then, please install the dependencies using the following command :

```bash
poetry install
```

### Running a specific test

Running a specific test with the environment activated is as easy as:
`pytest tests -k test_the_name_of_your_test`

### Running all the tests

With the environment activated you can run all of the tests
using:
`pytest tests`


## Documentation

For generic use case, please refer to this [notebook](docs/simple_example.ipynb).

<!-- For more specific use case, please refer to this [notebook](docs/detailed_example.ipynb). -->

<!-- For more detailed information, please refer to the [documentation](https://alexandregazagnes.github.io/scikit-transformers/). -->

A complete documentation will be soon available [here](https://alexandregazagnes.github.io/scikit-transformers/).


## Contributing

Best way to contribute is to open an issue and discuss about it.

Feel free to fork the project and make a pull request.
