"""
Setup for the Python implementation of Functional Curation.

This builds our Cython SUNDIALS wrapper. If SUNDIALS is installed in a
non-standard location, it requires environment variables (CFLAGS and LDFLAGS)
to have been set up before running.
"""

import numpy
import warnings
from cython import inline
from Cython.Build import cythonize
from setuptools import Extension, setup

# Detect major sundials version
sundials_major = inline(
    '''
    cdef extern from *:
        """
        #include <sundials/sundials_config.h>
        """
        int SUNDIALS_VERSION_MAJOR

    return SUNDIALS_VERSION_MAJOR
    ''',
    force=True,  # Always re-compile to pick up environment changes
)

if sundials_major < 3:
    warnings.warn(f"Unsupported SUNDIALS version {sundials_major}")

print(f"Building for Sundials {sundials_major}.x")

# Define Cython modules
extensions = [
    Extension(
        name="fc.sundials.solver",
        sources=["fc/sundials/solver.pyx"],
        include_dirs=[".", numpy.get_include()],
        libraries=["sundials_cvode", "sundials_nvecserial"],
    ),
]

# Setup
setup(
    name="fc",
    include_package_data=True,  # Include non-python files via MANIFEST.in
    zip_safe=False,
    ext_modules=cythonize(extensions, force=True),
)
