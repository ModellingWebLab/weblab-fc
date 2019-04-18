
import os
import pytest
import unittest

import fc


class TestMathmlEvaluation(unittest.TestCase):
    """Test implementation of voltage clamps in the Python code.

    Note that the first two cases in TestClamping.hpp are covered in TestAlgebraicModels.py!
    """

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def test_mathml_operations(self):
        proto = fc.Protocol('test/protocols/test_mathml_evaluation.txt')
        proto.set_output_folder('Py_TestMathmlEvaluation_TestMathmlOperations')
        proto.set_model('cellml/beeler_reuter_model_1977.cellml')
        proto.run()
        # Test assertions are within the protocol itself
        self.assertTrue(os.path.exists(os.path.join(proto.outputFolder.path, 'output.h5')))
