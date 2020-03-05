"""
Tests processing of the model interface section.
"""
import os

import pytest

import fc
from fc import data_loading
from fc.error_handling import ProtocolError
from fc.simulations import model


def test_input_output_units_mismatch():
    # Tests if an error is raised when a variable X is specified as input and output but with different units

    proto_file = 'test/protocols/test_invalid_interface.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_invalid_interface')
    proto.set_model('test/models/single_ode.cellml')
    with pytest.raises(ProtocolError, match='Some sensible error message'):
        proto.run()

