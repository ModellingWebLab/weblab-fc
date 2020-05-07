"""Test having optional protocol outputs."""
import pytest

import fc


def test_optional_outputs():
    # Test having optional protocol outputs that do or do not resolve to model variables (but aren't created)
    proto_file = 'test/protocols/test_optional_outputs.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_optional_outputs')
    proto.set_model('test/models/simple_ode.cellml')
    proto.run()

    # Voltage should be present, and can be looked up without error
    proto.output_env.look_up('V')

    # Chloride should not: this should fail
    with pytest.raises(KeyError, match='Cl_e is not defined'):
        proto.output_env.look_up('Cl_e')

