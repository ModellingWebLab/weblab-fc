
import pytest
import unittest

import fc


class TestAlgebraicModels(unittest.TestCase):
    """Test behaviour on models with no ODEs; based on TestClamping.hpp"""

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def test_clamping_to_initial_value(self):
        proto = fc.Protocol('test/protocols/test_clamping1.txt')
        proto.set_output_folder('Py_TestAlgebraicModels_TestClampingToInitialValue')
        proto.set_model('cellml/beeler_reuter_model_1977.cellml')
        proto.run()
        # Test assertions are within the protocol itself

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def test_clamping_to_fixed_value(self):
        proto = fc.Protocol('test/protocols/test_clamping2.txt')
        proto.set_output_folder('Py_TestAlgebraicModels_TestClampingToInitialValue')
        proto.set_model('cellml/beeler_reuter_model_1977.cellml')
        proto.run()
        # Test assertions are within the protocol itself
