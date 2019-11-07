"""

Simplest test for using model generated with weblab_cg.

"""
import os

import fc
from fc import test_support


def test_generated_model_graphstate():

    # Select model & protocol
    model_name = 'hodgkin_huxley_squid_axon_model_1952_modified'
    proto_name = 'GraphState'

    # Create protocol (generates model)
    proto = fc.Protocol(os.path.join(
        'test', 'protocols', 'generated_model_graphstate.txt'))

    # Run protocol
    proto.set_output_folder('test_generated_model_graphstate')
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))
    proto.run()
    # The test assertions are within the protocol itself

    # Check output exists
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

    # Check output is correct
    assert test_support.check_results(
        proto,
        {'state': 2},   # Name and dimension of output to check
        os.path.join('test', 'data', 'historic', model_name, proto_name),
        rel_tol=0.005,
        abs_tol=2.5e-4
    )

