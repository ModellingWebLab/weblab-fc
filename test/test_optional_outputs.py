"""Test having optional protocol outputs."""
import pytest

import fc


def test_optional_outputs():
    # Test having optional protocol outputs that do or do not resolve to model variables (but aren't created)
    proto_file = 'test/protocols/test_optional_outputs.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_optional_outputs')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

    # Voltage should be present
    v = proto.output_env.look_up('V').array

    # Chloride should not: this should fail
    cle_e = proto.output_env.look_up('Cl_e')

