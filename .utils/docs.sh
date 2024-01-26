#! /bin/sh


.venv/bin/python -m mkdocs build
.venv/bin/python -m mkdocs gh-deploy