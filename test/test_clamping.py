"""Test implementation of voltage clamps in the Python code."""
import os
import numpy as np
import pytest
import tables

import fc
from fc.file_handling import extract_output


def test_clamping_with_variable_units(tmpdir):
    # Test clamping state variables to a fixed value, with units taken from another model variable (units_of construct)

    proto = fc.Protocol('test/protocols/test_clamping3.txt')
    proto.set_output_folder('test_clamping_with_variable_units')
    proto.set_model('test/models/real/beeler_reuter_model_1977.cellml')
    proto.run()
    # Test assertions are within the protocol itself
    h5_path = os.path.join(proto.output_folder.path, 'output.h5')
    assert os.path.exists(h5_path)

    # Also check the extract_output helper
    extract_output(h5_path, 'V')
    location = os.path.join(proto.output_folder.path, 'outputs_V.csv')
    assert os.path.exists(location)
    array = np.loadtxt(location, dtype=float, delimiter=',')
    np.testing.assert_allclose(array, proto.output_env.look_up('V').array, rtol=0.01, atol=0.0)

    # Other call path
    with tables.open_file(h5_path, 'r') as h5_file:
        extract_output(h5_file, 'V', output_folder=str(tmpdir))
        assert tmpdir.join('outputs_V.csv').exists()
        array = np.loadtxt(str(tmpdir.join('outputs_V.csv')), dtype=float, delimiter=',')
        np.testing.assert_allclose(array, proto.output_env.look_up('V').array, rtol=0.01, atol=0.0)


def test_clamping_optional_variable():
    # Simple and fast test for clamping an optional variable to its initial value.

    proto = fc.Protocol('test/protocols/test_clamping_optional.txt')
    proto.set_output_folder('test_clamping_optional_variable')
    proto.set_model('test/models/simple_ode.cellml')
    proto.run()
    # Test assertions are within the protocol itself
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

    # V should be available, but Cl should not
    proto.output_env.look_up('V')
    with pytest.raises(KeyError, match='Cl is not defined'):
        proto.output_env.look_up('Cl')


def test_clamp_to_data_file():
    proto_file = 'test/protocols/real/timecourse_voltage_clamp.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_clamp_to_data_file')
    proto.set_model('test/models/real/ten_tusscher_model_2004_epi.cellml')
    proto.run()
    # Test assertions are within the protocol itself
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))


def test_interpolation_clamp():
    proto_file = 'test/protocols/test_model_interpolation.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_interpolation_clamp')
    proto.set_model('test/models/simple_ode.cellml')
    proto.run()
    # Test assertions are within the protocol itself
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))
