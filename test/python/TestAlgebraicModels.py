
import pytest
import unittest

import fc


class TestAlgebraicModels(unittest.TestCase):
    """Test behaviour on models with no ODEs; based on TestClamping.hpp"""

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def testClampingToInitialValue(self):
        proto = fc.Protocol('test/protocols/test_clamping1.txt')
        proto.SetOutputFolder('Py_TestAlgebraicModels_TestClampingToInitialValue')
        proto.SetModel('cellml/beeler_reuter_model_1977.cellml')
        proto.Run()
        # Test assertions are within the protocol itself

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def testClampingToFixedValue(self):
        proto = fc.Protocol('test/protocols/test_clamping2.txt')
        proto.SetOutputFolder('Py_TestAlgebraicModels_TestClampingToInitialValue')
        proto.SetModel('cellml/beeler_reuter_model_1977.cellml')
        proto.Run()
        # Test assertions are within the protocol itself
