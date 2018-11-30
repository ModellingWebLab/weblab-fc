
"""
Test distutils setup file for the Python implementation of Functional Curation.

At present, this just exists to allow us to build our Cython SUNDIALS wrapper.
If SUNDIALS is installed in a non-standard location, it requires environment variables
(CFLAGS and LDFLAGS) to have been set up before running.
"""


import numpy
from Cython.Distutils import build_ext
# from distutils.core import setup
from distutils.extension import Extension
from setuptools import find_packages, setup


ext_modules = [
    Extension('fc.sundials.sundials',
              sources=['fc/sundials/sundials.pxd'],
              include_dirs=['.', numpy.get_include()],
              libraries=['sundials_cvode', 'sundials_nvecserial']),
    Extension('fc.sundials.solver',
              sources=['fc/sundials/solver.pyx'],
              include_dirs=['.', numpy.get_include()],
              libraries=['sundials_cvode', 'sundials_nvecserial'])
]
with open('README.md') as f:
    readme = f.read()

setup(
    name='fc',
    version='0.1.0',
    description='Functional Curation backend for the Modelling Web Lab',
    long_description=readme,
    license='BSD',
    maintainer='Web Lab team',
    maintainer_email='j.p.cooper@ucl.ac.uk',
    url='https://github.com/ModellingWebLab/weblab-fc',
    packages=find_packages(exclude=['tests']),
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
        'cython',
        'matplotlib',
        'numexpr',
        'numpy',
        'scipy',
        'tables',
    ],
    extras_require={
        'dev': [
            'line_profiler',
            'setproctitle',
        ],
        'test': [
            'flake8',
            'pytest',
            'pytest-cov',
        ],
    },
    entry_points={
    },
)
