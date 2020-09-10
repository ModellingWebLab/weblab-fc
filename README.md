[![Build Status](https://travis-ci.org/ModellingWebLab/weblab-fc.svg?branch=master)](https://travis-ci.org/ModellingWebLab/weblab-fc)
[![codecov](https://codecov.io/gh/ModellingWebLab/weblab-fc/branch/master/graph/badge.svg)](https://codecov.io/gh/ModellingWebLab/weblab-fc)


# Functional Curation backend for the Modelling Web Lab

This is a Python implementation of the Functional Curation (FC) language, intended for use with the Cardiac Electrophysiology Web Lab.

Documentation on FC can be found [here](https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration), while the syntax of FC protocols is described [here](https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax).

An ongoing attempt to document the Web Lab and all its interconnected technologies can be viewed in the [weblab docs repository](https://github.com/ModellingWebLab/weblab_docs).

## No Windows support

**FC is tested/developed on Linux and OS/X. There are no plans to run it work on Windows**.

It _might_ run on Windows, if you have installed CVODE with the shared libraries, and an MSVC compiler that matches your Python installation (see [here](https://wiki.python.org/moin/WindowsCompilers)).

## Transition!

This package is currently in transition, migrating the functional curation code from being a Chaste extension project to a standalone Python package.
In particular, we're replacing the CellML and code-generation tool PyCml with [cellmlmanip](https://github.com/ModellingWebLab/cellmlmanip) to read and manipulate CellML code and [weblab_cg](https://github.com/ModellingWebLab/weblab-cg) to generate Model code.

The code from before this transition started can be seen at the tag [`pycml-version`](https://github.com/ModellingWebLab/weblab-fc/tree/pycml-version), but crucial parts of PyCml weblab code are also temporarily stored in the [pycml](./pycml) directory.
Most of the code from [pycml_protocol.py](./pycml/pycml_protocol.py) will have to be replaced by (1) changes to `fc` so that it extracts and stores _all_ the protocol information, and (2) updates to `weblab_cg` so that it can use information provides by `fc` to generate a model.
(At the moment both `fc` and `pycml` read the protocol, but in the new code `fc` should gather all the information and then pass it to `weblab_cg`.)

## Installation

In order to build the package you need Cython and numpy. These can be installed with:
```sh
pip install -r requirements/setup.txt
```

The ontologies used are in a separate module, to install this run

```sh
git submodule init
git submodule update
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

Because the `weblab_fc` module has Cython components, it needs to be compiled before you can use it.
Compilation is performed using Python's [`distutils`](https://docs.python.org/3/library/distutils.html) and [`setuptools`](https://setuptools.readthedocs.io/en/latest/), and happens automatically when you install the package using `setup.py`.
For developers, this can be done using:

```sh
pip install -e .[dev,test]
```

Note that you'll need to repeat this step after any changes to Cython files (e.g. `.pyx` or `.pxd` files), because these don't automatically get recompiled.

## Full installation steps on Jonathan's Macbook (slightly out of date)

```sh
export CONDA_ENV=weblab-fc-py36
conda create -n $CONDA_ENV python=3.6
conda activate $CONDA_ENV
conda install -c conda-forge sundials=4 pytables scipy numpy numexpr
pip install -r requirements/setup.txt
./requirements/weblab_cg.sh
export CFLAGS="-I/anaconda3/envs/$CONDA_ENV/include"
export LDFLAGS="-L/anaconda3/envs/$CONDA_ENV/lib"
pip install -e .[dev,test]
```
