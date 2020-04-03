"""Test models, simulations, ranges, and modifiers."""
import os
import pytest

import fc
import fc.test_support
import fc.language.values as V


#@pytest.mark.skipif(os.getenv('FC_LONG_TESTS', '0') == '0', reason='FC_LONG_TESTS not set to 1')
def test_s1_s2():
    """Tests running the full S1S2 protocol on the Courtemanche model."""

    proto = fc.Protocol('protocols/S1S2.txt')
    proto.set_output_folder('test_s1_s2')
    proto.set_model('test/models/courtemanche_ramirez_nattel_model_1998.cellml')
    proto.run()
    data_folder = 'test/data/TestSpeedRealProto/S1S2'
    fc.test_support.check_results(
        proto,
        {'raw_APD90': 2, 'raw_DI': 2, 'max_S1S2_slope': 1},
        data_folder
    )


def test_s1_s2_lr91():
    """ Run a shortened S1S2 protocol on LR1991 and check the resulting slope. """

    proto = fc.Protocol('test/protocols/test_S1S2.txt')
    proto.set_output_folder('test_s1_s2_lr91')
    proto.set_model('test/models/luo_rudy_1991.cellml')
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
    proto.set_model('test/models/earm_noble_model_1990.cellml')
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

