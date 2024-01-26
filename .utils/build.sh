#! /bin/sh

rm -rf build
rm -rf dist
python3 setup.py bdist_wheel
twine upload -u $TWINE_USER -p $TWINE_PASS  dist/*