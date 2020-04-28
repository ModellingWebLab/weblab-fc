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

    #TODO: Check outputs?
    '''

    # Check the max slope
    slope = proto.output_env.look_up('max_S1S2_slope').array
    assert len(slope) == 1
    assert slope[0] == pytest.approx(0.212, abs=1e-3)

    # Check we did the right number of timesteps (overridden protocol input)
    v = proto.output_env.look_up('membrane_voltage').array
    assert v.shape == (len(intervals), 2001)
    '''
