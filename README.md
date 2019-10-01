[![Build Status](https://travis-ci.org/ModellingWebLab/weblab-fc.svg?branch=master)](https://travis-ci.org/ModellingWebLab/weblab-fc)
[![codecov](https://codecov.io/gh/ModellingWebLab/weblab-fc/branch/master/graph/badge.svg)](https://codecov.io/gh/ModellingWebLab/weblab-fc)


# Functional Curation backend for the Modelling Web Lab

This is a Python implementation of the Functional Curation (FC) language, intended for use with the Cardiac Electrophysiology Web Lab.

Documentation on FC can be found [here](https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration), while the syntax of FC protocols is described [here](https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax).

An ongoing attempt to document the Web Lab and all its interconnected technologies is given at https://github.com/ModellingWebLab/weblab_docs


## Transition!

This package is currently in transition, migrating the functional curation code from being
a Chaste extension project to a standalone Python package.

## Installation

In order to build the package you need Cython and numpy. These can be installed with:
```sh
pip install -r requirements/setup.txt
```

For the temporary weblab_cg stuff, also run
```sh
requirements/weblab_cg.sh
```

You also need to have CVODE (from Sundials) installed. If you do this with your system package
manager no further setup is (probably) needed. Alternatively you can install it using `conda`:
```sh
conda install sundials -c conda-forge
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
