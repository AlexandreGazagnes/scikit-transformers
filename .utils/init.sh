#! /bin/bash

python3 -m poetry install
python3 -m poetry shell
pre-commit install
