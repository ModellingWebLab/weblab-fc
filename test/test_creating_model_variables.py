"""
Tests creating new annotated variables with optional+default, or via define.
"""
import fc


def test_create_annotated():
    # Test creating variables with optional+default and with define, with explicit unit

    proto_file = 'test/protocols/test_create_annotated.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_create_annotated')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()
    # Assertions are within the protocol itself


def test_create_annotated_no_units():
    # Test creating variables with optional+default and with define, but without unit information in either case

    proto_file = 'test/protocols/test_create_annotated_no_units.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_create_annotated_no_units')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()
    # Assertions are within the protocol itself

