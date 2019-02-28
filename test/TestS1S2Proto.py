
import os
import pytest
import unittest

import fc
import fc.test_support as TestSupport
from fc.simulations.solvers import CvodeSolver


@pytest.mark.skipif(os.getenv('FC_LONG_TESTS', '0') == '0',
                    reason="FC_LONG_TESTS not set to 1")
class TestS1S2Proto(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def testS1S2(self):
        proto = fc.Protocol('protocols/S1S2.txt')
        proto.SetOutputFolder('Py_TestS1S2Proto')
        proto.set_model('cellml/courtemanche_ramirez_nattel_1998.cellml')
        proto.model.SetSolver(CvodeSolver())
        proto.Run()
        data_folder = 'test/data/TestSpeedRealProto/S1S2'
        TestSupport.CheckResults(proto, {'raw_APD90': 2, 'raw_DI': 2, 'max_S1S2_slope': 1}, data_folder)
