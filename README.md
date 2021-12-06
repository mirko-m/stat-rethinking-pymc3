# About
This repo is my attempt at solving the assignments from `Statistical Rethinking`
by Richard McElreath. The original material can be found here
https://github.com/rmcelreath/statrethinking_winter2019 . Instead of
using R I will be using pymc3.

# Useful Links
1. The original material: https://github.com/rmcelreath/statrethinking_winter2019 
2. The books code ported to `pymc3` https://github.com/pymc-devs/resources/tree/master/Rethinking
3. Solutions to the assignments using `pymc3` https://github.com/gbosquechacon/statrethink_course_in_pymc3 

# Dependencies
```
pip install pymc3
sudo apt install libblas-dev
```

# Data
Most of the data was taken from https://github.com/rmcelreath/statrethinking_winter2019 .
It can also be scraped using the `get_data.py` script.

The only exception is 'data/hapiness.csv' which is the result of a simulation from
the `rethinking` package and was taken from
https://github.com/gbosquechacon/statrethink_course_in_pymc3 .

Note that the seperator for the columns is ";" instead of "," .  

# A Note on Version Control
Instead of pushing the jupyter notebooks directly I use `jupytext` to convert
them to python files. This makes it easier to follow changes that have been
made to the code which would otherwise be drowned out by all the changes to the
output of the cells. Since the python files do not contain any visualizations
I am also pushing the notebooks as HTML files.

Note that there are other methods to do version control with jupyter notebooks,
for example `nbdime` (https://nbdime.readthedocs.io/en/latest/) which also
works very well.

