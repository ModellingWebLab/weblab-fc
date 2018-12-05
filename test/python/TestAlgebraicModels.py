
import unittest

import fc


class TestAlgebraicModels(unittest.TestCase):
    """Test behaviour on models with no ODEs; based on TestClamping.hpp"""

    def testClampingToInitialValue(self):
        proto = fc.Protocol('projects/FunctionalCuration/test/protocols/test_clamping1.txt')
        proto.SetOutputFolder('Py_TestAlgebraicModels_TestClampingToInitialValue')
        proto.SetModel('projects/FunctionalCuration/cellml/beeler_reuter_model_1977.cellml')
        proto.Run()
        # Test assertions are within the protocol itself

    def testClampingToFixedValue(self):
        proto = fc.Protocol('projects/FunctionalCuration/test/protocols/test_clamping2.txt')
        proto.SetOutputFolder('Py_TestAlgebraicModels_TestClampingToInitialValue')
        proto.SetModel('projects/FunctionalCuration/cellml/beeler_reuter_model_1977.cellml')
        proto.Run()
        # Test assertions are within the protocol itself
