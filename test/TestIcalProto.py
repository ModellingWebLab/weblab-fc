import os
import pytest
import unittest

import fc
from fc import test_support
from fc.simulations.solvers import CvodeSolver


@pytest.mark.skipif(os.getenv('FC_LONG_TESTS', '0') == '0',
                    reason="FC_LONG_TESTS not set to 1")
class TestIcalProto(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def test_ical(self):
        proto = fc.Protocol('protocols/ICaL.txt')
        proto.set_output_folder('Py_TestIcalProto')
        proto.set_model('cellml/aslanidi_Purkinje_model_2009.cellml')
        proto.model.set_solver(CvodeSolver())
        proto.run()
        data_folder = 'test/data/TestSpeedRealProto/ICaL'
        test_support.check_results(
            proto, {'min_LCC': 2, 'final_membrane_voltage': 1}, data_folder)
