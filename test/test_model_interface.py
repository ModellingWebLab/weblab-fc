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
    proto.set_output_folder('test_bad_interface_duplicate_input')
    with pytest.raises(ProtocolError, match='as an input twice'):
        proto.set_model('test/models/single_ode.cellml')


def test_duplicate_output():
    # Tests if an error is raised when a variable is specified as output twice.

    proto_file = 'test/protocols/test_bad_interface_duplicate_output.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_bad_interface_duplicate_output')
    with pytest.raises(ProtocolError, match='as an output twice'):
        proto.set_model('test/models/single_ode.cellml')


def test_input_output_units_mismatch():
    # Tests if an error is raised when a variable is specified as input and output but with different units.

    proto_file = 'test/protocols/test_bad_interface_unit_mismatch.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_bad_interface_unit_mismatch')
    with pytest.raises(ProtocolError, match='with different units'):
        proto.set_model('test/models/single_ode.cellml')


def test_duplicate_clamp():
    # Tests if an error is raised when a variable is clamped twice

    proto_file = 'test/protocols/test_bad_interface_duplicate_clamp.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_bad_interface_duplicate_clamp')
    with pytest.raises(ProtocolError, match='multiple clamp statements'):
        proto.set_model('test/models/single_ode.cellml')


def test_duplicate_define():
    # Tests if an error is raised when a variable is redefined twice

    proto_file = 'test/protocols/test_bad_interface_duplicate_define.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_bad_interface_duplicate_define')
    with pytest.raises(ProtocolError, match='more than one clamp and/or define'):
        proto.set_model('test/models/single_ode.cellml')


def test_clamp_and_define_1():
    # Tests if an error is raised when a variable is clamped and set in a define statement
    # Clamp without an RHS

    proto_file = 'test/protocols/test_bad_interface_clamp_and_define_1.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_bad_interface_clamp_and_define_1')
    with pytest.raises(ProtocolError, match='more than one clamp and/or define'):
        proto.set_model('test/models/single_ode.cellml')


def test_clamp_and_define_2():
    # Tests if an error is raised when a variable is clamped and set in a define statement
    # Clamp with an RHS

    proto_file = 'test/protocols/test_bad_interface_clamp_and_define_2.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_bad_interface_clamp_and_define_2')
    with pytest.raises(ProtocolError, match='more than one clamp and/or define'):
        proto.set_model('test/models/single_ode.cellml')

