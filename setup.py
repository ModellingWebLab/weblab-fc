"""
Setup for the Python implementation of Functional Curation.

This builds our Cython SUNDIALS wrapper. If SUNDIALS is installed in a
non-standard location, it requires environment variables (CFLAGS and LDFLAGS)
to have been set up before running.
"""

import numpy
from cython import inline
from Cython.Build import cythonize
from setuptools import Extension, setup

# Detect major sundials version
SUNDIALS_MAJOR = inline(
    '''
    cdef extern from "sundials/sundials_config.h":
        int SUNDIALS_VERSION_MAJOR

    return SUNDIALS_VERSION_MAJOR
    '''
)

assert SUNDIALS_MAJOR >= 3, f"Unsupported Sundials version: {SUNDIALS_MAJOR}"

print(f"Building for Sundials {SUNDIALS_MAJOR}")

# Define Cython modules
extensions = [
    Extension(
        name="fc.sundials.solver",
        sources=["fc/sundials/solver.pyx"],
        include_dirs=[".", numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        libraries=["sundials_cvode", "sundials_nvecserial"],
    ),
]

# Setup
setup(
    name="fc",
    include_package_data=True,  # Include non-python files via MANIFEST.in
    zip_safe=False,
    ext_modules=cythonize(
        extensions,
        compile_time_env={"SUNDIALS_MAJOR": SUNDIALS_MAJOR},
    ),
)
