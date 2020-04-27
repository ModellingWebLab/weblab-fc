"""
Test default expressions for optional variables.
"""
import pytest

import fc
from fc.error_handling import ProtocolError


def test_optional_with_default_expression():
    # Test creating an optional variable with a default expression

    proto_file = 'test/protocols/test_default_expression.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_optional_with_default_expression')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()
    # Assertions are within the protocol itself


def test_optional_with_default_expression_but_no_units():
    # Test creating an optional variable with a default expression, but without enough information to create (no units)

    proto_file = 'test/protocols/test_default_expression_no_units.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_optional_with_default_expression_but_no_units')
    with pytest.raises(ProtocolError, match='blaaaaaar'):
        proto.set_model('test/data/simple_ode.cellml')

