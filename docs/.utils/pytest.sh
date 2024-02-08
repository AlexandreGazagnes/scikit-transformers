#! /bin/sh


# .venv/bin/python3 -m pytest .
rm -f ./docs/assets/img/cov.svg
rm -f .coverage
.venv/bin/coverage run -m pytest -vvx --capture=tee-sys --log-cli-level=INFO tests/
.venv/bin/coverage-badge -fo ./docs/assets/img/cov.svg
