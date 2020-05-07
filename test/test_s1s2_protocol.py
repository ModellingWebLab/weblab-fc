"""Test models, simulations, ranges, and modifiers."""
import os
import pytest

import fc
import fc.test_support


@pytest.mark.skipif(os.getenv('FC_LONG_TESTS', '0') == '0', reason='FC_LONG_TESTS not set to 1')
def test_s1_s2():
    """Tests running the full S1S2 protocol on the Courtemanche model."""

    proto = fc.Protocol('test/protocols/real/S1S2.txt')
    proto.set_output_folder('test_s1_s2')
    proto.set_model('test/models/real/courtemanche_ramirez_nattel_model_1998.cellml')
    proto.run()
    data_folder = 'test/data/TestSpeedRealProto/S1S2'
    fc.test_support.check_results(
        proto,
        {'raw_APD90': 2, 'raw_DI': 2, 'max_S1S2_slope': 1},
        data_folder
    )

