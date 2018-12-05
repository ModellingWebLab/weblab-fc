
import os
try:
    import unittest2 as unittest
except ImportError:
    import unittest

import fc


class TestMathmlEvaluation(unittest.TestCase):
    """Test implementation of voltage clamps in the Python code.

    Note that the first two cases in TestClamping.hpp are covered in TestAlgebraicModels.py!
    """

    def TestMathmlOperations(self):
        proto = fc.Protocol('projects/FunctionalCuration/test/protocols/test_mathml_evaluation.txt')
        proto.SetOutputFolder('Py_TestMathmlEvaluation_TestMathmlOperations')
        proto.SetModel('projects/FunctionalCuration/cellml/beeler_reuter_model_1977.cellml')
        proto.Run()
        # Test assertions are within the protocol itself
        self.assert_(os.path.exists(os.path.join(proto.outputFolder.path, 'output.h5')))
