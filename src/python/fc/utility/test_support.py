"""Copyright (c) 2005-2015, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
Routines of use in tests of Functional Curation.
"""

import os
import numpy as np

from ..language import values as V

def CheckResults(proto, expectedSpec, dataFolder):
    """Check protocol results against saved values.
    expectedSpec is a dictionary mapping result name to number of dimensions, so we can use the correct Load* method.
    """
    for name, ndims in expectedSpec.iteritems():
        data_file = os.path.join(dataFolder, 'outputs_' + name + '.csv')
        if ndims == 2:
            method = Load2d
        else:
            method = Load
        expected = method(data_file)
        actual = proto.outputEnv.LookUp(name)
        np.testing.assert_allclose(actual.array, expected.array, rtol=0.01)

def CheckFileCompression(filePath):
    """Return (real_path, is_compressed) if a .gz compressed version of filePath exists."""
    real_path = filePath
    if filePath.endswith('.gz'):
        is_compressed = True
    else:
        if os.path.exists(filePath):
            is_compressed = False
        elif os.path.exists(filePath + '.gz'):
            real_path += '.gz'
            is_compressed = True
    return real_path, is_compressed

def Load2d(filePath):
    """Load the legacy data format for 2d arrays."""
    real_path, is_compressed = CheckFileCompression(filePath)
    array = np.loadtxt(real_path, dtype=float, delimiter=',', unpack=True) # unpack transposes the array
    if array.ndim == 1:
        array = array[:, np.newaxis]
    return V.Array(array)

def Load(filePath):
    """Load the legacy data format for arbitrary dimension arrays."""
    real_path, is_compressed = CheckFileCompression(filePath)
    if is_compressed:
        import gzip
        f = gzip.GzipFile(real_path, 'rb')
    else:
        f = open(real_path, 'r')
    f.readline() # Strip comment line
    dims = map(int, f.readline().split(','))[1:]
    array = np.loadtxt(f, dtype=float)
    f.close()
    return V.Array(array.reshape(tuple(dims)))
