"""
Tests for unit conversion of inputs and defines.
For further unit conversion tests see test_graphstate.py
"""
import pytest

import fc


@pytest.mark.xfail(strict=True, reason='Unit conversion not implemented yet.')
def test_unit_conversion_inputs():
    # Test unit conversion for the initial value of a state variable

    proto_file = 'test/protocols/test_unit_conversion_inputs.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_inputs')

    # This model has v in mV, with dv/dt=1 mV/ms, and v(0) = 2mV
    # The protocol will change this to v(0) = 10 V, and then run for 5ms
    proto.set_model('test/models/single_ode.cellml')
    proto.run()

    # Check output
    v = proto.output_env.look_up('membrane_voltage').array
    print(v)
    assert len(v) == 2
    assert v[0] == 10
    assert v[1] == 10.005

