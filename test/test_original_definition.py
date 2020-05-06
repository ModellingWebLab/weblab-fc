"""
Test using the original definition when replacing an equation.
"""
import fc


def test_original_definition():
    # Test using the original definition when replacing an equation.

    proto_file = 'test/protocols/test_original_definition.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_original_definition')
    proto.set_model('test/models/simple_ode.cellml')
    proto.run()
    # Assertions are within the protocol itself

