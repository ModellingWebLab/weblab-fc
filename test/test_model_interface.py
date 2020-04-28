"""
Tests processing of the model interface section.
"""
import pytest

import fc
from fc.error_handling import ProtocolError


def test_interface_inconsistent_input():
    # Tests if an error is raised when a variable is specified as input twice, with inconsistent information.

    proto_file = 'test/protocols/test_interface_inconsistent_input.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_interface_inconsistent_input')
    with pytest.raises(ProtocolError, match='Multiple initial values'):
        proto.set_model('test/models/single_ode.cellml')


def test_interface_inconsistent_output():
    # Tests if an error is raised when a variable is specified as output twice, with inconsistent information.

    proto_file = 'test/protocols/test_interface_inconsistent_output.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_interface_inconsistent_output')
    with pytest.raises(ProtocolError, match='Inconsistent units'):
        proto.set_model('test/models/single_ode.cellml')


def test_interface_inconsistent_input_output():
    # Tests if an error is raised when a variable is specified as input and output but with different units.

    proto_file = 'test/protocols/test_interface_inconsistent_input_output.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_interface_inconsistent_input_output')
    with pytest.raises(ProtocolError, match='Inconsistent units'):
        proto.set_model('test/models/single_ode.cellml')


def test_interface_inconsistent_clamp_1():
    # Tests if an error is raised when a variable is clamped in different ways

    proto_file = 'test/protocols/test_interface_inconsistent_clamp_1.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_interface_inconsistent_clamp_1')
    with pytest.raises(ProtocolError, match='more than one clamp and/or define'):
        proto.set_model('test/models/single_ode.cellml')


def test_interface_inconsistent_clamp_2():
    # Tests if an error is raised when a variable is clamped in different ways

    proto_file = 'test/protocols/test_interface_inconsistent_clamp_2.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_interface_inconsistent_clamp_2')
    with pytest.raises(ProtocolError, match='Multiple equations'):
        proto.set_model('test/models/single_ode.cellml')


def test_interface_inconsistent_define():
    # Tests if an error is raised when a variable is redefined in different ways

    proto_file = 'test/protocols/test_interface_inconsistent_define.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_interface_duplicate_define')
    with pytest.raises(ProtocolError, match='Multiple equations'):
        proto.set_model('test/models/single_ode.cellml')


def test_interface_clamp_and_define_1():
    # Tests if an error is raised when a variable is clamped and set in a define statement
    # Tests with clamp-to-initial-value

    proto_file = 'test/protocols/test_interface_clamp_and_define_1.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_interface_clamp_and_define_1')
    with pytest.raises(ProtocolError, match='more than one clamp and/or define'):
        proto.set_model('test/models/single_ode.cellml')


def test_interface_clamp_and_define_2():
    # Tests if an error is raised when a variable is clamped and set in a define statement
    # Tests with clamp-to

    proto_file = 'test/protocols/test_interface_clamp_and_define_2.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_interface_clamp_and_define_2')
    with pytest.raises(ProtocolError, match='Multiple equations'):
        proto.set_model('test/models/single_ode.cellml')

