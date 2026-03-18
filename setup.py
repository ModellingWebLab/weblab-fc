"""
Setup for the Python implementation of Functional Curation.

This builds our Cython SUNDIALS wrapper. If SUNDIALS is installed in a 
non-standard location, it requires environment variables (CFLAGS and LDFLAGS) 
to have been set up before running.
"""
from setuptools import Extension, setup # Must come before Cython!
import numpy
from Cython.Build import cythonize
from cython import inline

# Detect major sundials version (defaults to 2)
fc_sundials_major = inline('''
    cdef extern from "<sundials/sundials_config.h>":
        """
        #ifndef SUNDIALS_VERSION_MAJOR
            #define SUNDIALS_VERSION_MAJOR 2
        #endif
        """
        int SUNDIALS_VERSION_MAJOR

    return SUNDIALS_VERSION_MAJOR
    ''')
print("Building for Sundials " + str(fc_sundials_major) + ".x")

# Define Cython modules
extensions = [
    Extension(name="fc.sundials.sundials",
              sources=["fc/sundials/sundials.pxd"],
              include_dirs=[".", numpy.get_include()],
              libraries=["sundials_cvode", "sundials_nvecserial"],
              ),
    Extension(name="fc.sundials.solver",
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
    ext_modules=cythonize(extensions, compile_time_env={'FC_SUNDIALS_MAJOR': fc_sundials_major},),
)
