"""
Test the protocol interface analysis for the Web Lab front-end.
"""

from cellmlmanip import load_model

from fc.parsing.rdf import get_used_annotations


def test_get_required_annotations():
    pass


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
    pass
