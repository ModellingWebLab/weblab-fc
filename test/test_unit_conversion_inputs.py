"""
Tests for unit conversion of inputs and defines.
For further unit conversion tests see test_graphstate.py
"""
import pytest

import fc


@pytest.mark.xfail(strict=True, reason='Unit conversion not implemented yet.')
def test_unit_conversion_inputs_initial_values():
    # Test unit conversion for initial values of states and constants.

    proto_file = 'test/protocols/test_unit_conversion_inputs_initial_values.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_inputs_initial_values')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()

    # Check initial value of constant was set correctly
    # The model has Na in mM with Na(0)=0mM, and dNa/dt=n, where n has initial value 1 mM/ms
    # The protocol will change n's initial value to 3 M/ms, and then run for 5ms
    Na = proto.output_env.look_up('cytosolic_sodium_concentration').array
    assert list(Na) == [0, 15000]

    # Check initial value of state was set correctly
    # The model has v in mV, with dv/dt=1 mV/ms, and v(0)=2mV
    # The protocol will change V's units to volts, set v(0)=10 V, and then run for 5ms
    v = proto.output_env.look_up('membrane_voltage').array
    assert list(v) == [10, 10.005]


@pytest.mark.xfail(strict=True, reason='Unit conversion not implemented yet.')
def test_unit_conversion_inputs_defines():
    # Test unit conversion for input variables (states and constants) modified with define statements.

    proto_file = 'test/protocols/test_unit_conversion_inputs_defines.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_inputs_defines')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()

    # Check RHS of constant was set correctly
    # The model has Na in mM with Na(0)=0mM, and dNa/dt=n, where n has initial value 1 mM/ms
    # The protocol will define n=5M/ms, and then run for 5ms
    Na = proto.output_env.look_up('cytosolic_sodium_concentration').array
    assert list(Na) == [0, 25000]

    # Check initial value of state was set correctly
    # The model has v in mV, with dv/dt=1 mV/ms, and v(0)=2mV
    # The protocol will change V's units to volts, define dV/dt=3V/ms, and then run for 5ms
    v = proto.output_env.look_up('membrane_voltage').array
    assert list(v) == [0, 15]

