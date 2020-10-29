"""
Routines of use in tests of Functional Curation.
"""

import logging
import os
import numpy as np

from .data_loading import load, load2d


def get_process_number():
    """Get the number of the current process within a process pool.

    Numbering starts from 1.  If this process is not one started by the multiprocessing module, 0 is returned.
    """
    import multiprocessing
    name = multiprocessing.current_process().name
    try:
        num = int(name.split('-')[-1])
    except Exception:
        num = 0
    return num

# The following utility methods for comparing floating point numbers are
# based on boost/test/floating_point_comparison.hpp


def within_relative_tolerance(arr1, arr2, tol):
    """Determine if two arrays are element-wise close within the given relative tolerance.

    :returns: a boolean array
    """
    with np.errstate(all='ignore'):
        difference = np.fabs(arr1 - arr2)
        d1 = np.nan_to_num(difference / np.fabs(arr1))
        d2 = np.nan_to_num(difference / np.fabs(arr2))
    return np.logical_and(d1 <= tol, d2 <= tol)


def within_absolute_tolerance(arr1, arr2, tol):
    """Determine if two arrays are element-wise close within the given absolute tolerance.

    A difference of exactly the tolerance is considered to be OK.

    :returns: a boolean array
    """
    return np.fabs(arr1 - arr2) <= tol


def get_max_errors(arr1, arr2):
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


def within_any_tolerance(arr1, arr2, rel_tol=None, abs_tol=None):
    """Determine if two arrays are element-wise close within the given tolerances.

    If either the relative OR absolute tolerance is satisfied for a given pair of values, the result is true.

    :param rel_tol: relative tolerance.
        If omitted, machine epsilon is used to effect a comparison only under absolute tolerance.
    :param abs_tol: absolute tolerance.
        If omitted, machine epsilon is used to effect a comparison only under relative tolerance.
    :returns: a boolean array
    """
    if rel_tol is None:
        rel_tol = np.finfo(np.float).eps
    if abs_tol is None:
        abs_tol = np.finfo(np.float).eps
    return np.logical_or(
        within_absolute_tolerance(arr1, arr2, abs_tol),
        within_relative_tolerance(arr1, arr2, rel_tol))


def check_results(proto, expected_spec, data_folder, rel_tol=0.01, abs_tol=0):
    """Check protocol results against saved values.

    Note that if the protocol is missing expected results, this is only an error if reference results are actually
    present on disk. If no reference results are available for an 'expected' output, this indicates that the protocol is
    expected to fail (or at least, not produce this output). Similarly, it is not an error if the protocol produces
    results but no reference results are available, although we do add a warning to messages (if supplied) in this case.

    An assertion will be raised if any output is outside tolerances compared to the reference results, or if an expected
    output (with reference results available) was not produced.

    :param proto: an instance of fc.Protocol that (hopefully) has results available to check
    :param expected_spec: a dictionary mapping result name to number of dimensions, so we can use the correct load*
        method
    :param data_folder: location of the reference data
    :param rel_tol: relative tolerance
    :param abs_tol: absolute tolerance
    """
    for name, ndims in expected_spec.items():
        data_file = os.path.join(data_folder, 'outputs_' + name + '.csv')
        try:
            actual = proto.output_env.look_up(name)
        except KeyError:
            assert not os.path.exists(data_file), f"Output {name} not produced but reference result exists"
            continue  # Can't compare in this case
        if not os.path.exists(data_file):
            logging.warning(
                f"Output {name} produced but no reference result available - please save for future comparison")
            continue  # Can't compare in this case
        if ndims == 2:
            expected = load2d(data_file)
        else:
            expected = load(data_file)
        np.testing.assert_allclose(actual.array, expected.array, rtol=rel_tol, atol=abs_tol)
