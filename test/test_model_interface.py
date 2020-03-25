"""
Tests processing of the model interface section.
"""
import pytest

import fc
from fc.error_handling import ProtocolError


def test_duplicate_input():
    # Tests if an error is raised when a variable is specified as input twice.

    proto_file = 'test/protocols/test_bad_interface_duplicate_input.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_invalid_interface')
    with pytest.raises(ProtocolError, match='as an input twice'):
        proto.set_model('test/models/single_ode.cellml')
        proto.run()


def test_duplicate_output():
    # Tests if an error is raised when a variable is specified as output twice.

    proto_file = 'test/protocols/test_bad_interface_duplicate_output.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_invalid_interface')
    with pytest.raises(ProtocolError, match='as an output twice'):
        proto.set_model('test/models/single_ode.cellml')
        proto.run()


def test_input_output_units_mismatch():
    # Tests if an error is raised when a variable is specified as input and output but with different units.

    proto_file = 'test/protocols/test_bad_interface_unit_mismatch.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_invalid_interface')
    with pytest.raises(ProtocolError, match='different unit'):
        proto.set_model('test/models/single_ode.cellml')
        proto.run()

