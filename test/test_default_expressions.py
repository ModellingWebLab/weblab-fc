"""
Test default expressions for optional variables.
"""
import fc


def test_optional_with_default_expression():
    # Test creating an optional variable with a default expression

    proto_file = 'test/protocols/test_default_expression.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_optional_with_default_expression')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()
    # Assertions are within the protocol itself
