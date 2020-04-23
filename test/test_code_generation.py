#
# Tests templating functionality for the Cardiac Electrophysiology Web Lab
#
import cellmlmanip
import logging
import os
import re

import fc
import fc.code_generation
import fc.test_support
from fc.parsing.actions import ProtocolVariable
from fc.parsing.rdf import OXMETA_NS, PRED_IS_VERSION_OF, create_rdf_node


# Show more logging output
logging.getLogger().setLevel(logging.INFO)


def test_unique_name_generation():
    # Tests if unique variable names are generated correctly

    # Load cellml model, get unique names
    model = cellmlmanip.load_model(
        os.path.join('test', 'models', 'conflicting_names.cellml'))

    # Test unique names
    unames = fc.code_generation.get_unique_names(model)
    assert len(unames) == 9
    variables = [v for v in model.graph]
    variables.sort(key=str)

    assert unames[variables[0]] == 'time'         # env.time
    assert unames[variables[1]] == 'x__a'         # x.a
    assert unames[variables[2]] == 'b'            # x.b
    assert unames[variables[3]] == 'x__y__z_1'    # x.y__z
    assert unames[variables[4]] == 'x__y__a'      # x__y.a
    assert unames[variables[5]] == 'x__y__z'      # x__y.x__y__z
    assert unames[variables[6]] == 'z'            # x__y.z
    assert unames[variables[7]] == 'z__a'         # z.a
    assert unames[variables[8]] == 'z__y__z'      # z.y__z


def test_generate_weblab_model(tmp_path):
    # Tests the create_weblab_model() method

    # Select output path (in temporary dir)
    path = tmp_path / 'model.pyx'

    # Select class name
    class_name = 'TestModel'

    # Load cellml model
    model = os.path.join('test', 'models', 'hodgkin_huxley_squid_axon_model_1952_modified.cellml')
    model = cellmlmanip.load_model(model)

    # Combined output and parameter information
    protocol_variables = []
    variables = [
        (True, False, 'membrane_fast_sodium_current_conductance'),
        (True, False, 'membrane_potassium_current_conductance'),
        (False, True, 'membrane_fast_sodium_current'),
        (False, True, 'membrane_voltage'),
        (False, True, 'time'),
    ]
    for input, output, name in variables:
        rdf_term = create_rdf_node((OXMETA_NS, name))
        pvar = ProtocolVariable(OXMETA_NS + ':' + name, name, rdf_term)
        pvar.update(input=input, output=output)
        pvar.update(model_variable=model.get_variable_by_ontology_term(rdf_term))
        protocol_variables.append(pvar)

    # State variable output
    rdf_term = create_rdf_node((OXMETA_NS, 'state_variable'))
    pvar = ProtocolVariable(OXMETA_NS + ':' + 'state_variable', 'state_variable', rdf_term)
    pvar.update(output=True, output_category=True, transitive_variables=model.get_state_variables())
    protocol_variables.append(pvar)

    # Annotate state variables with the magic oxmeta:state_variable term
    state_annotation = create_rdf_node((OXMETA_NS, 'state_variable'))
    vector_orderings = {state_annotation: {}}
    for i, state_var in enumerate(model.get_state_variables()):
        model.add_cmeta_id(state_var)
        model.rdf.add((state_var.rdf_identity, PRED_IS_VERSION_OF, state_annotation))
        vector_orderings[state_annotation][state_var.rdf_identity] = i

    # Create weblab model at path
    fc.code_generation.create_weblab_model(
        str(path),
        class_name,
        model,
        ns_map={'oxmeta': OXMETA_NS},
        protocol_variables=protocol_variables,
        vector_orderings=vector_orderings,
    )

    # Read expected output from file
    expected = os.path.join('test', 'code_generation', 'weblab_model.pyx')
    with open(expected, 'r') as f:
        expected = f.read()

    # Read generated output from file
    generated = path.read_text()

    # Store locally to update test output file
    if os.environ.get('WEBLAB_REGENERATE_REF'):
        print('REGENERATING CODE GENERATION REFERENCE FILE')
        with open(expected, 'w') as f:
            f.write(generated)

    # Remove line about creation date and version
    p = re.compile('# Generated by .+')
    expected = p.sub('# Generated by me', expected).strip()
    generated = p.sub('# Generated by me', generated).strip()

    # Now they should match
    assert generated == expected


def test_graphstate():
    """ Tests the graph state protocol on a generated model. """

    # Create protocol
    proto = fc.Protocol(os.path.join('protocols', 'GraphState.txt'))

    # Set model (generates & compiles model)
    model_name = 'hodgkin_huxley_squid_axon_model_1952_modified'
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))

    # Run protocol
    proto.set_output_folder('test_graphstate')
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
