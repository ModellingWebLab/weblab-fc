"""
Test the protocol interface analysis for the Web Lab front-end.
"""

from cellmlmanip import load_model

from fc import Protocol
from fc.parsing.rdf import get_used_annotations


def test_get_required_annotations_clamp():
    proto = Protocol('test/protocols/test_required_annotations.txt')
    exp_required = set([
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#time',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#extracellular_sodium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#cytosolic_sodium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_voltage',
        'urn:fc:local#funky_converter',
    ])
    exp_optional = set([
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#cytosolic_potassium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_fast_sodium_current',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#extracellular_calcium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#cytosolic_calcium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_rapid_delayed_rectifier_potassium_current',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_inward_rectifier_potassium_current',
        'urn:fc:local#newvar',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_capacitance',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#extracellular_potassium_concentration',
    ])
    actual_req, actual_opt = proto.get_required_model_annotations()
    assert exp_required == actual_req
    assert exp_optional == actual_opt


def test_get_model_annotations():
    model = load_model('test/models/simple_ode.cellml')
    model_terms = get_used_annotations(model)
    expected = set([
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#time',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_voltage',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#cytosolic_sodium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#state_variable',
        'urn:test-ns#parameter_a',
        'urn:test-ns#parameter_b',
        'urn:test-ns#parameter_n',
    ])
    assert expected == model_terms


def test_model_compatibility():
    proto = Protocol('test/protocols/test_required_annotations.txt')
    missing_terms, missing_optional_terms = proto.check_model_compatibility('test/models/simple_ode.cellml')
    assert [
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#extracellular_sodium_concentration',
        'urn:fc:local#funky_converter',
    ] == missing_terms
    assert [
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#cytosolic_calcium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#cytosolic_potassium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#extracellular_calcium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#extracellular_potassium_concentration',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_capacitance',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_fast_sodium_current',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_inward_rectifier_potassium_current',
        'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#membrane_rapid_delayed_rectifier_potassium_current',
        'urn:fc:local#newvar',
    ] == missing_optional_terms


def test_protocol_interface():
    proto = Protocol('test/protocols/test_required_annotations.txt')
    actual = proto.get_protocol_interface()
    expected = [
        {'kind': 'output', 'name': 'missing_units', 'units': ''},
        {'kind': 'output', 'name': 'model_interface_units', 'units': '0.001 second'},
        {'kind': 'output', 'name': 'pp_defined_units', 'units': '0.001 second'},
        {'kind': 'output', 'name': 'sim_defined_units', 'units': '1 second'},
        {'kind': 'output', 'name': 'unknown_units', 'units': ''},
    ]
    actual.sort(key=lambda d: d['name'])
    assert expected == actual
