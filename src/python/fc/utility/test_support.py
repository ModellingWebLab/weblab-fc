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

def GetProcessNumber():
    """Get the number of the current process within a process pool.
    
    Numbering starts from 1.  If this process is not one started by the multiprocessing module, 0 is returned.
    """
    import multiprocessing
    name = multiprocessing.current_process().name
    try:
        num = int(name.split('-')[-1])
    except:
        num = 0
    return num

# The following utility methods for comparing floating point numbers are based on boost/test/floating_point_comparison.hpp

def WithinRelativeTolerance(arr1, arr2, tol):
    """Determine if two arrays are element-wise close within the given relative tolerance.
    
    :returns: a boolean array
    """
    with np.errstate(all='ignore'):
        difference = np.fabs(arr1 - arr2)
        d1 = np.nan_to_num(difference / np.fabs(arr1))
        d2 = np.nan_to_num(difference / np.fabs(arr2))
    return np.logical_and(d1 <= tol, d2 <= tol)

def WithinAbsoluteTolerance(arr1, arr2, tol):
    """Determine if two arrays are element-wise close within the given absolute tolerance.
    
    A difference of exactly the tolerance is considered to be OK.
    
    :returns: a boolean array
    """
    return np.fabs(arr1 - arr2) <= tol

def GetMaxErrors(arr1, arr2):
    """Compute the maximum relative and absolute pairwise errors between two arrays.
    
    :returns: (max relative error, max absolute error)
    """
    with np.errstate(all='ignore'):
        difference = np.fabs(arr1 - arr2)
        d1 = np.nan_to_num(difference / np.fabs(arr1))
        d2 = np.nan_to_num(difference / np.fabs(arr2))
    max_rel_err = np.amax(np.maximum(d1, d2))
    max_abs_err = np.amax(np.fabs(arr1 - arr2))
    return (max_rel_err, max_abs_err)

def WithinAnyTolerance(arr1, arr2, relTol=None, absTol=None):
    """Determine if two arrays are element-wise close within the given tolerances.
    
    If either the relative OR absolute tolerance is satisfied for a given pair of values, the result is true.
    
    :param relTol: relative tolerance. If omitted, machine epsilon is used to effect a comparison only under absolute tolerance.
    :param absTol: absolute tolerance. If omitted, machine epsilon is used to effect a comparison only under relative tolerance.
    :returns: a boolean array
    """
    if relTol is None:
        relTol = np.finfo(np.float).eps
    if absTol is None:
        absTol = np.finfo(np.float).eps
    return np.logical_or(WithinAbsoluteTolerance(arr1, arr2, absTol), WithinRelativeTolerance(arr1, arr2, relTol))

def CheckResults(proto, expectedSpec, dataFolder, rtol=0.01, atol=0, messages=None):
    """Check protocol results against saved values.
    
    Note that if the protocol is missing expected results, this is only an error if reference results are actually present
    on disk.  If no reference results are available for an 'expected' output, this indicates that the protocol is expected
    to fail (or at least, not produce this output).  Similarly, it is not an error if the protocol produces results but no
    reference results are available, although we do add a warning to messages (if supplied) in this case.
    
    :param proto: an instance of fc.Protocol that (hopefully) has results available to check
    :param expectedSpec: a dictionary mapping result name to number of dimensions, so we can use the correct Load* method
    :param rtol: relative tolerance
    :param atol: absolute tolerance
    :param messages: if provided, should be a list to which failure reports will be appended.  Otherwise any failure will raise AssertionError.
    :returns: a boolean indicating whether the results matched to within tolerances, or None if failure was expected.
    """
    results_ok = True
    for name, ndims in expectedSpec.iteritems():
        data_file = os.path.join(dataFolder, 'outputs_' + name + '.csv')
        try:
            actual = proto.outputEnv.LookUp(name)
        except KeyError:
            if os.path.exists(data_file):
                results_ok = False
                if messages is not None:
                    messages.append("Output %s not produced but reference result exists" % name)
            elif results_ok:
                results_ok = None # Indicate expected failure
            continue # Can't compare in this case
        if not os.path.exists(data_file):
            if messages is not None:
                messages.append("Output %s produced but no reference result available - please save for future comparison" % name)
            continue # Can't compare in this case
        if ndims == 2:
            method = Load2d
        else:
            method = Load
        expected = method(data_file)
        if messages is None:
            np.testing.assert_allclose(actual.array, expected.array, rtol=rtol, atol=atol)
        else:
            if actual.array.shape != expected.array.shape:
                messages.append("Output %s shape %s does not match expected shape %s" % (name, actual.array.shape, expected.array.shape))
                results_ok = False
            else:
                close_entries = WithinAnyTolerance(actual.array, expected.array, relTol=rtol, absTol=atol)
                if not close_entries.all():
                    max_rel_err, max_abs_err = GetMaxErrors(actual.array, expected.array)
                    bad_entries = np.logical_not(close_entries)
                    bad = actual.array[bad_entries]
                    first_bad = bad.flat[:10]
                    first_expected = expected.array[bad_entries].flat[:10]
                    messages.append("Output %s was not within tolerances (rel=%g, abs=%g) in %d of %d locations. Max rel error=%g, max abs error=%g.\nFirst <=10 mismatches: %s != %s" %
                                    (name, rtol, atol, bad.size, actual.array.size, max_rel_err, max_abs_err, first_bad, first_expected))
                    results_ok = False
    return results_ok

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
    array = np.loadtxt(real_path, dtype=float, delimiter=',', ndmin=2, unpack=True) # unpack transposes the array
    assert array.ndim == 2
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
