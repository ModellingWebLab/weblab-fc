"""
Tests the units_of() construct when giving units to numbers.
"""
import fc


def test_units_of():
    # Tests the units_of construct

    proto_file = 'test/protocols/test_units_of.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_units_of')
    proto.set_model('test/models/simple_ode.cellml')
    proto.run()
    # Test assertions are within the protocol itself
