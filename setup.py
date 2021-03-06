
"""
Test distutils setup file for the Python implementation of Functional Curation.

At present, this just exists to allow us to build our Cython SUNDIALS wrapper.
If SUNDIALS is installed in a non-standard location, it requires environment variables
(CFLAGS and LDFLAGS) to have been set up before running.
"""
import numpy

from setuptools import find_packages, setup  # Must come before Cython!
from cython import inline
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

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
ext_modules = [
    Extension('fc.sundials.sundials',
              sources=['fc/sundials/sundials.pxd'],
              include_dirs=['.', numpy.get_include()],
              libraries=['sundials_cvode', 'sundials_nvecserial'],
              cython_compile_time_env={'FC_SUNDIALS_MAJOR': sundials_major},
              ),
    Extension('fc.sundials.solver',
              sources=['fc/sundials/solver.pyx'],
              include_dirs=['.', numpy.get_include()],
              libraries=['sundials_cvode', 'sundials_nvecserial'],
              cython_compile_time_env={'FC_SUNDIALS_MAJOR': sundials_major},
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
    packages=find_packages(exclude=['test', 'test.*']),
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
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    install_requires=[
        'cellmlmanip',
        'cython',
        'Jinja2>=2.10',
        'matplotlib',
        'numexpr',
        'numpy',
        'pyparsing!=2.4.2',
        'scipy',
        'tables',
    ],
    extras_require={
        'dev': [
            # 'line_profiler',
            'pytest-xdist[psutil]',
        ],
        'test': [
            'codecov',
            'flake8>=3.6',
            'pytest>=3.6',
            'pytest-cov',
            'pytest-profiling',
        ],
    },
    entry_points={
        'console_scripts': [
            'fc_run = fc.cli:run_protocol',
            'fc_extract_outputs = fc.cli:extract_outputs',
            'fc_check_syntax = fc.cli:check_syntax',
        ],
    },
)
