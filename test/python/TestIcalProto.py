
import os
import pytest
import unittest

import fc
import fc.utility.test_support as TestSupport
from fc.simulations.solvers import CvodeSolver


@pytest.mark.skipif(os.getenv('FC_LONG_TESTS', '0') == '0',
                    reason="FC_LONG_TESTS not set to 1")
class TestIcalProto(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""

    def testIcal(self):
        proto = fc.Protocol('projects/FunctionalCuration/protocols/ICaL.txt')
        proto.SetOutputFolder('Py_TestIcalProto')
        proto.SetModel('projects/FunctionalCuration/cellml/aslanidi_Purkinje_model_2009.cellml', useNumba=False)
        proto.model.SetSolver(CvodeSolver())
        proto.Run()
        data_folder = 'projects/FunctionalCuration/test/data/TestSpeedRealProto/ICaL'
        TestSupport.CheckResults(proto, {'min_LCC': 2, 'final_membrane_voltage': 1}, data_folder)
