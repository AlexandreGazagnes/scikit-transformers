#! /bin/bash

# ###########################
# # clean notebooks output
# ###########################

# for f in *.ipynbp
# do
#   jupyter nbconvert --clear-output --inplace $f
# done

# for f in ./*/*.ipynb
# do
#   jupyter nbconvert --clear-output --inplace $f
# done

# for f in ./*/*/*.ipynb
# do
#   jupyter nbconvert --clear-output --inplace $f
# done


###########################
# copy .py from .ipynb
###########################

# => COMMAND TO MAKE .py FRM .ipynb FILES AND COPY ALL IN ./src/ FOLDER
# => UNCOMMENT IF NEEDED

# for f in *.ipynb
# do
#   jupytext --to py:percent $f
#   fn=$(basename $f); #   echo "FN => $fn"
#   new="./gbs/notebooks/"$fn ; #   echo "new => $new"
#   mv $f $new
# done

# for f in ./*/*.ipynb
# do
#   jupytext --to py:percent $f
#   fn=$(basename $f); #   echo "FN => $fn"
#   new="./gbs/notebooks/"$fn ; #   echo "new => $new"
#   # mv $f $new
# done


# for f in ./*/*.ipynb
# do
#   jupytext --to pdf $f
#   fn=$(basename $f); #   echo "FN => $fn"
#   # new="./gbs/notebooks/"$fn ; #   echo "new => $new"
#   # mv $f $new
# done

###########################
# black and flake8
###########################


.venv/bin/python3 -m isort ./skres/ ./tests/
# .venv/bin/python3 -m flake8 .
.venv/bin/python3 -m black ./skres/ ./tests/


# ###########################
# # pytest & coverage
# ###########################

# # .venv/bin/python3 -m pytest
.venv/bin/coverage -m pytest -vv -x -s tests/
.venv/bin/coverage html
rm .assets/cov.svg
.venv/bin/coverage-badge -o .assets/cov.svg
