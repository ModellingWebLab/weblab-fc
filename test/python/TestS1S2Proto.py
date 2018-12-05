
try:
    import unittest2 as unittest
except ImportError:
    import unittest

import fc
import fc.utility.test_support as TestSupport
import fc.simulations.model as Model
from fc.simulations.solvers import CvodeSolver


class TestS1S2Proto(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""

    def TestS1S2(self):
        proto = fc.Protocol('projects/FunctionalCuration/protocols/S1S2.txt')
        proto.SetOutputFolder('Py_TestS1S2Proto')
        proto.SetModel('projects/FunctionalCuration/cellml/courtemanche_ramirez_nattel_1998.cellml')
        proto.model.SetSolver(CvodeSolver())
        proto.Run()
        data_folder = 'projects/FunctionalCuration/test/data/TestSpeedRealProto/S1S2'
        TestSupport.CheckResults(proto, {'raw_APD90': 2, 'raw_DI': 2, 'max_S1S2_slope': 1}, data_folder)
