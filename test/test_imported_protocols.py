"""Test protocol imports at the top level."""
import pytest

import fc
import fc.language.values as V


def test_s1_s2_lr91():
    """ Run a shortened S1S2 protocol on LR1991 and check the resulting slope. """

    proto = fc.Protocol('test/protocols/test_S1S2.txt')
    proto.set_output_folder('test_s1_s2_lr91')
    proto.set_model('test/real/models/luo_rudy_1991.cellml')
    intervals = [1000, 900, 800, 700, 600, 500]
    proto.set_input('s2_intervals', V.Array(intervals))
    proto.run()

    # Check the max slope
    slope = proto.output_env.look_up('max_S1S2_slope').array
    assert len(slope) == 1
    assert slope[0] == pytest.approx(0.212, abs=1e-3)

    # Check we did the right number of timesteps (overridden protocol input)
    v = proto.output_env.look_up('membrane_voltage').array
    assert v.shape == (len(intervals), 2001)


def test_s1_s2_noble():
    """ This model has time units in seconds, so we're checking that conversion works too. """

    proto = fc.Protocol('test/protocols/test_S1S2.txt')
    proto.set_output_folder('test_s1_s2_noble')
    proto.set_model('test/real/models/earm_noble_model_1990.cellml')
    intervals = [1000, 900, 800, 700, 600, 500]
    proto.set_input('s2_intervals', V.Array(intervals))
    proto.run()

    # Check the max slope
    slope = proto.output_env.look_up('max_S1S2_slope').array
    assert len(slope) == 1
    assert slope[0] == pytest.approx(0.0264, abs=1e-3)

    # Check we did the right number of timesteps (overridden protocol input)
    v = proto.output_env.look_up('membrane_voltage').array
    assert v.shape == (len(intervals), 2001)

