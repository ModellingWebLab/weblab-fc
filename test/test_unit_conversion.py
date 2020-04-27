"""
Tests for unit conversion.
For further unit conversion tests see the GraphState test in test_code_generation.py
"""
import os
import pytest

import fc
import fc.test_support
from fc.error_handling import ProtocolError


def test_unit_conversion_time():
    """ Tests the graph state protocol in a model requiring time units conversion. """

    # Create protocol
    proto = fc.Protocol(os.path.join('protocols', 'GraphState.txt'))

    # Set model (generates & compiles model)
    model_name = 'difrancesco_noble_model_1985'  # has time in seconds, not milliseconds
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))

    # Run protocol
    proto.set_output_folder('test_unit_conversion_time')
    proto.run()
    # Some test assertions are within the protocol itself

    # Check output exists
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

    # Check output is correct
    assert fc.test_support.check_results(
        proto,
        {'state': 2},   # Name and dimension of output to check
        os.path.join('test', 'data', 'historic', model_name, 'GraphState'),
        rel_tol=0.005,
        abs_tol=2.5e-4
    )


def test_unit_conversion_state_variable():
    """ Tests the graph state protocol in a model requiring state variable units conversion. """

    # Create protocol
    proto = fc.Protocol(os.path.join('protocols', 'GraphState.txt'))

    # Set model (generates & compiles model)
    model_name = 'paci_hyttinen_aaltosetala_severi_ventricularVersion'  # has voltage in volt, not millivolt
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))

    # Run protocol
    proto.set_output_folder('test_unit_conversion_state_variable')
    proto.run()
    # Some test assertions are within the protocol itself

    # Check output exists
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

    # Check output is correct
    assert fc.test_support.check_results(
        proto,
        {'state': 2},   # Name and dimension of output to check
        os.path.join('test', 'data', 'historic', model_name, 'GraphState'),
        rel_tol=0.005,
        abs_tol=2.5e-4
    )


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


def test_unit_conversion_transitive_and_within_equations():
    # Test unit conversion for transitive variables and within equations (with define statements)

    proto_file = 'test/protocols/test_transitive_variables.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_transitive_and_within_equations')
    proto.set_model('test/models/transitive_variables.cellml')
    proto.run()
    # Assertions are within the protocol itself


def test_unit_conversion_transitive_variables_clash():
    # Tests an error is raised if units are set from a direct annotation and a vector annotation

    proto_file = 'test/protocols/test_transitive_variables_clash.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_unit_conversion_transitive_variables_clash')
    with pytest.raises(ProtocolError, match='set individually as'):
        proto.set_model('test/models/transitive_variables.cellml')

