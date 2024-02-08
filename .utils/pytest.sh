#! /bin/sh


# .venv/bin/python3 -m pytest .
rm -f .assets/cov.svg
rm -f .coverage
coverage run -m pytest -vvx --capture=tee-sys --log-cli-level=INFO tests/
coverage-badge -fo .assets/cov.svg
