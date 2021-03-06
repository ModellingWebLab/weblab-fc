"""
Test behaviour on models with no ODEs.
"""

import fc


def test_clamping_to_initial_value():
    proto = fc.Protocol('test/protocols/test_clamping1.txt')
    proto.set_output_folder('test_clamping_to_initial_value')
    proto.set_model('test/models/real/beeler_reuter_model_1977.cellml')
    proto.run()
    # Test assertions are within the protocol itself


def test_clamping_to_fixed_value():
    proto = fc.Protocol('test/protocols/test_clamping2.txt')
    proto.set_output_folder('test_clamping_to_fixed_value')
    proto.set_model('test/models/real/beeler_reuter_model_1977.cellml')
    proto.run()
    # Test assertions are within the protocol itself
