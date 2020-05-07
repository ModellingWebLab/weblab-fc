"""
Test the INa IV curve protocols, which clamps optional variables, and requires several other features including local
var declarations and unit conversion rules.
"""
import pytest

import fc


def test_clamping_with_old_model():

    proto = fc.Protocol('test/protocols/test_INa_IV_curves.txt')
    proto.set_output_folder('test_clamping_with_old_model')
    proto.set_model('test/models/real/beeler_reuter_model_1977.cellml')
    proto.run()

    # Check the key outputs haven't changed.
    peaks = proto.output_env.look_up('normalised_peak_currents').array
    expected = [1.0, 0.999998, 0.999981, 0.999906, 0.999564, 0.999062, 0.99795, 0.995344, 0.988466, 0.966859, 0.887918,
                0.624599, 0.192671, 0.0186169, 0.00193235, 0.00116954, 0.00114086, 0.00113976, 0.00113976]
    assert len(peaks) == len(expected)
    ipeaks = iter(peaks)
    for value in expected:
        assert next(ipeaks) == pytest.approx(value, 1e-3)


def test_clamping_with_default_expressions():

    # Note: Difference with test_clamping_with_old_model is the protocol use (test_INa.. vs INa...)
    proto = fc.Protocol('test/protocols/real/INa_IV_curves.txt')
    proto.set_output_folder('test_clamping_with_default_expressions')
    proto.set_model('test/models/real/beeler_reuter_model_1977.cellml')
    proto.run()

    # Check the key outputs haven't changed.
    peaks = proto.output_env.look_up('normalised_peak_currents').array
    expected = [1.0, 0.999998, 0.999981, 0.999906, 0.999564, 0.999062, 0.99795, 0.995344, 0.988466, 0.966859, 0.887918,
                0.624599, 0.192671, 0.0186169, 0.00193235, 0.00116954, 0.00114086, 0.00113976, 0.00113976]
    assert len(peaks) == len(expected)
    ipeaks = iter(peaks)
    for value in expected:
        assert next(ipeaks) == pytest.approx(value, 1e-3)


def test_clamping_computed_variable():

    proto = fc.Protocol('test/protocols/real/INa_IV_curves.txt')
    proto.set_output_folder('test_clamping_computed_variable')
    proto.set_model('test/models/real/matsuoka_model_2003.cellml')
    proto.run()

    # Check the key outputs haven't changed.
    peaks = proto.output_env.look_up('normalised_peak_currents').array
    expected = [0.893527, 0.946238, 0.980761, 0.999288, 1, 0.98876, 0.961362, 0.903678, 0.79484, 0.622032, 0.410937,
                0.224468, 0.104744, 0.044427, 0.018137, 0.00741457, 0.00312374, 0.000728363, 0.000343776]
    assert len(peaks) == len(expected)
    ipeaks = iter(peaks)
    for value in expected:
        assert next(ipeaks) == pytest.approx(value, 1e-3)
