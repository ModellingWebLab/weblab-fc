"""

Temporary test for dynamically generated `pyx` model with weblab_cg.

"""
import os
# import pytest

import fc
from fc.utility import test_support


def test_static_pyx_file():

    model_name = 'hodgkin_huxley_squid_axon_model_1952_modified'
    proto_name = 'GraphState'

    proto = fc.Protocol(os.path.join(
        'test', 'protocols', 'dynamic_model_graphstate.txt'))
    proto.SetOutputFolder('test_dynamic_pyx_file')
    proto.set_model(os.path.join(
        'test', 'models', model_name + '.cellml'))
    proto.Run()
    # Test assertions are within the protocol itself

    # Check output exists
    assert os.path.exists(os.path.join(proto.outputFolder.path, 'output.h5'))

    # Check output is correct
    assert test_support.CheckResults(
        proto,
        {'state': 2},   # Name and dimension of output to check
        os.path.join('test', 'data', 'historic', model_name, proto_name),
        rtol=0.005,
        atol=2.5e-4
    )

