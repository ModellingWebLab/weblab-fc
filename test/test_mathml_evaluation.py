#
# Test implementation of MathML operators and functions.
#
import os

import fc


def test_mathml_operations():

    proto = fc.Protocol('test/protocols/test_mathml_evaluation.txt')
    proto.set_output_folder('test_mathml_operations')
    proto.set_model(os.path.join('test', 'models', 'real', 'beeler_reuter_model_1977.cellml'))
    proto.run()

    # Test assertions are within the protocol itself
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

    # Note: There's not MathML:Piecewise support, but there is an Ifthenelse function in the protocol language. This is
    # used when clamping to irregularly spaced variables and tested in e.g. test_s1s2_proto.py.
