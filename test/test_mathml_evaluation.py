"""Test implementation of voltage clamps.

Note that the first two cases in TestClamping.hpp are covered in TestAlgebraicModels.py!
"""
import os

import fc


def test_mathml_operations():

    proto = fc.Protocol('test/protocols/test_mathml_evaluation.txt')
    proto.set_output_folder('Py_TestMathmlEvaluation_TestMathmlOperations')
    proto.set_model(os.path.join('test', 'models', 'beeler_reuter_model_1977.cellml'))
    proto.run()
    # Test assertions are within the protocol itself
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

