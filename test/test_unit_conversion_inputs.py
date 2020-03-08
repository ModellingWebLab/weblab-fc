"""
Tests for unit conversion of inputs and defines.
For further unit conversion tests see test_graphstate.py
"""
import pytest

import fc


def test_unit_conversion_inputs_initial_values():
    # Test unit conversion for initial values of states and constants.

    proto_file = 'test/protocols/test_unit_conversion_inputs_initial_values.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_inputs_initial_values')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()

    # Check initial value of constant was set correctly
    # The model has Na in mM with Na(0)=0 mM, and dNa/dt=n, where n has initial value 1 mM/ms
    # The protocol will change n's initial value to 3 M/ms, and then run for 5 ms
    Na = proto.output_env.look_up('cytosolic_sodium_concentration').array
    assert len(Na) == 2
    assert Na[0] == 0
    assert Na[1] == pytest.approx(15000, rel=1e-15)

    # Check initial value of state was set correctly
    # The model has v in mV, with dv/dt=1 mV/ms, and v(0)=0 mV
    # The protocol will change v's units to volts, set initial value v(0)=10 V, and then run for 5 ms
    v = proto.output_env.look_up('membrane_voltage').array
    assert len(v) == 2
    assert v[0] == 10
    assert v[1] == pytest.approx(10.005, rel=1e-15)


def test_unit_conversion_inputs_defines():
    # Test unit conversion for input variables (states and constants) modified with define statements.

    proto_file = 'test/protocols/test_unit_conversion_inputs_defines.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_inputs_defines')
    proto.set_model('test/data/simple_ode.cellml')
    proto.run()

    # Check RHS of constant was set correctly
    # The model has Na in mM with Na(0)=0 mM, and dNa/dt=n, where n has initial value 1 mM/ms
    # The protocol will define n=5 M/ms, and then run for 5 ms
    Na = proto.output_env.look_up('cytosolic_sodium_concentration').array
    assert len(Na) == 2
    assert Na[0] == 0
    assert Na[1] == pytest.approx(25000, rel=1e-15)

    # Check initial value of state was set correctly
    # The model has v in mV, with dv/dt=1 mV/ms, and v(0)=0 mV
    # The protocol will change v's units to volts, define dv/dt=3 V/ms, and then run for 5 ms
    v = proto.output_env.look_up('membrane_voltage').array
    assert len(v) == 2
    assert v[0] == 0
    assert v[1] == pytest.approx(15, rel=1e-15)

