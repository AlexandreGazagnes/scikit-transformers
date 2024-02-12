#! /bin/sh


rm -f ./docs/assets/img/cov.svg
# rm -f .coverage
coverage run -m pytest -vvx --capture=tee-sys --log-cli-level=INFO tests/
coverage-badge -fo ./docs/assets/img/cov.svg
