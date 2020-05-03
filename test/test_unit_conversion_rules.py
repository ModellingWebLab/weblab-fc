"""
Tests for unit conversion rules.
"""
import fc
import pint
import pytest


def test_unit_conversion_rules():
    # Test unit conversion rules

    proto_file = 'test/protocols/test_unit_conversion_rules.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_rules')
    proto.set_model('test/models/unit_conversion_rules.cellml')
    proto.run()
    # Assertions are within the protocol itself

