
import os
import pytest
import unittest

import fc
from fc.utility import test_support
from fc.simulations.solvers import CvodeSolver


@pytest.mark.skipif(os.getenv('FC_LONG_TESTS', '0') == '0',
                    reason="FC_LONG_TESTS not set to 1")
class TestIcalProto(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def testIcal(self):
        proto = fc.Protocol('protocols/ICaL.txt')
        proto.SetOutputFolder('Py_TestIcalProto')
        proto.set_model('cellml/aslanidi_Purkinje_model_2009.cellml')
        proto.model.SetSolver(CvodeSolver())
        proto.Run()
        data_folder = 'test/data/TestSpeedRealProto/ICaL'
        test_support.CheckResults(
            proto, {'min_LCC': 2, 'final_membrane_voltage': 1}, data_folder)
