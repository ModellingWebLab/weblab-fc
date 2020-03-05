"""
Tests for unit conversion of inputs and defines.
For further unit conversion tests see test_graphstate.py
"""
import pytest

import fc


#@pytest.mark.xfail(strict=True, reason='Unit conversion not implemented yet.')
def test_unit_conversion_inputs():
    # Test unit conversion for initial values of states and constants.

    proto_file = 'test/protocols/test_unit_conversion_inputs.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_inputs')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()

    # Check initial value of constant was set correctly
    # The model has Na in mM with Na(0)=0mM, and dNa/dt=n, where n has initial value 1 mM/ms
    # The protocol will change this to 3 M/ms, and then run for 5ms
    Na = proto.output_env.look_up('cytosolic_sodium_concentration').array
    assert len(Na) == 2
    assert Na[0] == 0
    assert Na[1] == 15000

    # Check initial value of state was set correctly
    # The model has v in mV, with dv/dt=1 mV/ms, and v(0)=2mV
    # The protocol will change this to v(0)=10 V, and then run for 5ms
    v = proto.output_env.look_up('membrane_voltage').array
    assert len(v) == 2
    assert v[0] == 10
    assert v[1] == 10.005

