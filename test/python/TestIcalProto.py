
try:
    import unittest2 as unittest
except ImportError:
    import unittest

import fc
import fc.utility.test_support as TestSupport
import fc.simulations.model as Model
from fc.simulations.solvers import CvodeSolver


class TestIcalProto(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""

    def TestIcal(self):
        proto = fc.Protocol('projects/FunctionalCuration/protocols/ICaL.txt')
        proto.SetOutputFolder('Py_TestIcalProto')
        proto.SetModel('projects/FunctionalCuration/cellml/aslanidi_Purkinje_model_2009.cellml', useNumba=False)
        proto.model.SetSolver(CvodeSolver())
        proto.Run()
        data_folder = 'projects/FunctionalCuration/test/data/TestSpeedRealProto/ICaL'
        TestSupport.CheckResults(proto, {'min_LCC': 2, 'final_membrane_voltage': 1}, data_folder)
