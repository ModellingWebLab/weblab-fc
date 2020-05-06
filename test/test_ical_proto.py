"""Test models, simulations, ranges, and modifiers."""

import os
import pytest

import fc
from fc import test_support


@pytest.mark.skipif(os.getenv('FC_LONG_TESTS', '0') == '0', reason="FC_LONG_TESTS not set to 1")
@pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
def test_ical():
    proto = fc.Protocol('test/real/protocols/ICaL.txt')
    proto.set_output_folder('test_ical')
    proto.set_model('test/real/models/aslanidi_Purkinje_model_2009.cellml')
    proto.run()
    data_folder = 'test/data/TestSpeedRealProto/ICaL'
    test_support.check_results(proto, {'min_LCC': 2, 'final_membrane_voltage': 1}, data_folder)
