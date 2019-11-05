"""
Methods for loading data from disk.
"""

import os
import numpy as np

from .language import values as V


def check_file_compression(file_path):
    """Return (real_path, is_compressed) if a .gz compressed version of file_path exists."""
    real_path = file_path
    is_compressed = False
    if file_path.endswith('.gz'):
        is_compressed = True
    elif os.path.exists(file_path + '.gz'):
        real_path += '.gz'
        is_compressed = True
    return real_path, is_compressed


def load2d(file_path):
    """Load the legacy data format for 2d arrays."""
    real_path, is_compressed = check_file_compression(file_path)
    array = np.loadtxt(real_path, dtype=float, delimiter=',', ndmin=2, unpack=True)  # unpack transposes the array
    assert array.ndim == 2
    return V.Array(array)


def load(file_path):
    """Load the legacy data format for arbitrary dimension arrays."""
    real_path, is_compressed = check_file_compression(file_path)
    if is_compressed:
        import gzip
        f = gzip.GzipFile(real_path, 'rb')
    else:
        f = open(real_path, 'r')
    f.readline()  # strip comment line
    dims = map(int, f.readline().split(','))[1:]
    array = np.loadtxt(f, dtype=float)
    f.close()
    return V.Array(array.reshape(tuple(dims)))
