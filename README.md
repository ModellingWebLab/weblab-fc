# Functional Curation backend for the Modelling Web Lab

This package is currently in transition, migrating the functional curation code from being
a Chaste extension project to a standalone Python package.

The original project readme is in README.txt.

## Installation

In order to build the package you need Cython and numpy. These can be installed with:
```sh
pip install -r requirements/setup.txt
```

You also need to have CVODE (from Sundials) installed. If you do this with your system package
manager no further setup is (probably) needed. Just make sure to install version 2; we do not
(yet) support version 3. Alternatively you can install it using `conda`:
```sh
conda install sundials=2.7.0 -c conda-forge
```
In this case, or if it is installed in another non-standard location on your machine, you'll
need to set environment variables to tell Cython where to find it, e.g.:
```sh
export CFLAGS="-I$HOME/anaconda3/envs/weblab/include"
export LDFLAGS="-L$HOME/anaconda3/envs/weblab/lib"
```

You can then install this package with developer dependencies:
```sh
pip install .[dev,test]
```
