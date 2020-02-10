"""

Run the (relatively simple) GraphState protocol on a model generated with weblab_cg.

"""
import os

import fc
from fc import test_support


def test_generated_model_graphstate():
    # Create protocol
    proto = fc.Protocol(os.path.join(
        'test', 'protocols', 'generated_model_graphstate.txt'))

    # Set model (generates & compiles model)
    model_name = 'hodgkin_huxley_squid_axon_model_1952_modified'
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))

    # Run protocol
    proto.set_output_folder('test_generated_model_graphstate')
    proto.run()
    # Some test assertions are within the protocol itself

    # Check output exists
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

    # Check output is correct
    assert test_support.check_results(
        proto,
        {'state': 2},   # Name and dimension of output to check
        os.path.join('test', 'data', 'historic', model_name, 'GraphState'),
        rel_tol=0.005,
        abs_tol=2.5e-4
    )


def test_graphstate_time_conversion():
    # Create protocol
    proto = fc.Protocol(os.path.join(
        'test', 'protocols', 'generated_model_graphstate.txt'))

    # Set model (generates & compiles model)
    model_name = 'difrancesco_noble_model_1985'  # has time in seconds, not milliseconds
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))

    # Run protocol
    proto.set_output_folder('test_graphstate_time_conversion')
    proto.run()
    # Some test assertions are within the protocol itself

    # Check output exists
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

    # Check output is correct
    assert test_support.check_results(
        proto,
        {'state': 2},   # Name and dimension of output to check
        os.path.join('test', 'data', 'historic', model_name, 'GraphState'),
        rel_tol=0.005,
        abs_tol=2.5e-4
    )


def test_graphstate_voltage_conversion():
    # Create protocol
    proto = fc.Protocol(os.path.join(
        'test', 'protocols', 'generated_model_graphstate.txt'))

    # Set model (generates & compiles model)
    model_name = 'paci_hyttinen_aaltosetala_severi_ventricularVersion'  # has time in seconds, not milliseconds
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))

    # Run protocol
    proto.set_output_folder('test_graphstate_voltage_conversion')
    proto.run()
    # Some test assertions are within the protocol itself

    # Check output exists
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))

    # Check output is correct
    assert test_support.check_results(
        proto,
        {'state': 2},   # Name and dimension of output to check
        os.path.join('test', 'data', 'historic', model_name, 'GraphState'),
        rel_tol=0.005,
        abs_tol=2.5e-4
    )
