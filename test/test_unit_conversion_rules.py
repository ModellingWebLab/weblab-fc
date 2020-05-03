"""
Tests for unit conversion rules.
"""
import fc
import pint


def test_unit_conversion_rules():
    # Test unit conversion rules

    proto_file = 'test/protocols/test_unit_conversion_rules.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_rules')
    proto.set_model('test/models/paci_hyttinen_aaltosetala_severi_atrialVersion.cellml')
    proto.run()
    # Assertions are within the protocol itself


def test_unit_conversion_rules_incompatible():
    # Test unit conversion rules won't work if model variables are in the wrong units.
    # In this case, the luo-rudy model seems to be badly annotated, having a uF/cm^2 variable annotated as capacitance

    proto_file = 'test/protocols/test_unit_conversion_rules.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_rules')
    with pytest.raises(pint.errors.DimensionalityError):
        proto.set_model('test/models/luo_rudy_1991.cellml')

