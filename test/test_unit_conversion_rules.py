"""
Tests for unit conversion rules.
"""
import fc


def test_unit_conversion_rules():
    # Test unit conversion for transitive variables and within equations (with define statements)

    proto_file = 'test/protocols/test_unit_conversion_rules.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_rules')
    proto.set_model('test/models/luo_rudy_1991.cellml')
    proto.run()
    # Assertions are within the protocol itself

