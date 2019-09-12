
"""
Test distutils setup file for the Python implementation of Functional Curation.

At present, this just exists to allow us to build our Cython SUNDIALS wrapper.
If SUNDIALS is installed in a non-standard location, it requires environment variables
(CFLAGS and LDFLAGS) to have been set up before running.
"""
import numpy

from setuptools import find_packages, setup  # Must come before Cython!
from distutils.extension import Extension
from cython import inline
from Cython.Build import cythonize

# Detect major sundials version (defaults to 2)
sundials_major = inline('''
    cdef extern from *:
        """
        #include <sundials/sundials_config.h>

        #ifndef SUNDIALS_VERSION_MAJOR
            #define SUNDIALS_VERSION_MAJOR 2
        #endif
        """
        int SUNDIALS_VERSION_MAJOR

    return SUNDIALS_VERSION_MAJOR
    ''')
print('Building for Sundials ' + str(sundials_major) + '.x')

# Define Cython modules
extensions = [
    Extension('fc.sundials.sundials',
              sources=['fc/sundials/sundials.pxd'],
              include_dirs=['.', numpy.get_include()],
              libraries=['sundials_cvode', 'sundials_nvecserial'],
              ),
    Extension('fc.sundials.solver',
              sources=['fc/sundials/solver.pyx'],
              include_dirs=['.', numpy.get_include()],
              libraries=['sundials_cvode', 'sundials_nvecserial'],
              ),
]

# Load readme for use as long description
with open('README.md') as f:
    readme = f.read()

# Setup
setup(
    name='fc',
    version='0.1.0',
    description='Functional Curation backend for the Modelling Web Lab',
    long_description=readme,
    license='BSD',
    maintainer='Web Lab team',
    maintainer_email='j.p.cooper@ucl.ac.uk',
    url='https://github.com/ModellingWebLab/weblab-fc',
    packages=find_packages(exclude=('test', 'test.*')),
    include_package_data=True,  # Include non-python files via MANIFEST.in
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    ext_modules=cythonize(extensions,
                          compile_time_env={'FC_SUNDIALS_MAJOR': sundials_major}
    ),
    install_requires=[
        # 'cellmlmanip',    # Add this in when cellmlmanip is ready
        'cython',
        'lxml',
        'matplotlib',
        'numexpr',
        'numpy',
        'pyparsing<2.4',
        'scipy',
        'tables',
        # 'weblab_cg',      # Add this in when weblab_cg is ready
    ],
    extras_require={
        'dev': [
            # 'line_profiler',
            'setproctitle',
        ],
        'test': [
            'flake8>=3.6',
            'pytest>=3.6',
            'pytest-cov',
        ],
    },
    entry_points={
    },
)
