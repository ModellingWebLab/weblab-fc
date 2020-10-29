
import os
import pytest

import fc
from fc import test_support


def proto_id_fn(protocol_settings):
    """Generate a pytest parameter ID using just the protocol name."""
    proto_name, expected_outputs = protocol_settings
    return proto_name


@pytest.mark.skipif(os.getenv('FC_LONG_TESTS', '0') == '0',
                    reason='FC_LONG_TESTS not set to 1')
@pytest.mark.parametrize(
    'model_name',
    [
        'aslanidi_atrial_model_2009',
        'aslanidi_Purkinje_model_2009',
        'beeler_reuter_model_1977',
        'carro_2011_epi',
        'decker_2009',
        'difrancesco_noble_model_1985',
        'fink_noble_giles_model_2008',
        'grandi_pasqualini_bers_2010_ss',
        'li_mouse_2010',
        'luo_rudy_1991',
        'mahajan_shiferaw_2008',
        'ohara_rudy_2011_epi',
        'shannon_wang_puglisi_weber_bers_2004',
        'ten_tusscher_model_2006_epi',
    ]
)
@pytest.mark.parametrize(
    'protocol_settings',
    [
        # (protocol name, expected outputs)
        ('ExtracellularPotassiumVariation', {'scaled_APD90': 1,
                                             'scaled_resting_potential': 1,
                                             'detailed_voltage': 2}),
        ('GraphState', {'state': 2}),
        ('ICaL', {'min_LCC': 2, 'final_membrane_voltage': 1}),
        ('IK1_IV_curve', {'normalised_low_K1': 1, 'normalised_high_K1': 1}),
        ('IKr_block', {'scaled_APD90': 1, 'detailed_voltage': 2}),
        ('IKr_IV_curve', {'normalised_peak_Kr_tail': 1}),
        ('INa_block', {'scaled_APD90': 1, 'detailed_voltage': 2}),
        ('INa_IV_curves', {'normalised_peak_currents': 1, 'current_activation': 2}),
        ('NCX_block', {'scaled_resting_potential': 1, 'scaled_APD90': 1, 'detailed_voltage': 2}),
        ('RyR_block', {'scaled_APD90': 1, 'detailed_voltage': 2}),
        ('S1S2', {'S1S2_slope': 1}),
        ('SteadyStateRestitution', {'APD': 2, 'restitution_slope': 1}),
        ('SteadyStateRunner', {'num_paces': 0, 'detailed_voltage': 1}),
        ('SteadyStateRunner4Hz', {'num_paces': 0, 'detailed_voltage': 1}),
    ],
    ids=proto_id_fn
)
def test_fc_experiment(model_name, protocol_settings, tmpdir, request):
    """Test reproducibility of an experiment, i.e. application of a protocol to a model.

    :param model_name: name of model to run, i.e. no path or extension
    :param protocol_settings: pair with details of a protocol to run and expected results:
        first element gives the name of protocol to run, i.e. no path or extension;
        second is a dictionary of outputs to check against reference data,
        mapping output name to number of dimensions
    """
    proto_name, expected_outputs = protocol_settings
    print(f'Applying {proto_name} to {model_name}')

    # If there are missing reference results then this is expected to fail
    data_folder = f'test/output/real/{model_name}/{proto_name}'
    for name in expected_outputs:
        if not os.path.exists(os.path.join(data_folder, 'outputs_' + name + '.csv')):
            request.applymarker(pytest.mark.xfail(reason=f'Missing reference result {name}', strict=True))
            break

    # Try to run the protocol
    proto = fc.Protocol('test/protocols/real/%s.txt' % proto_name)
    os.environ['CHASTE_TEST_OUTPUT'] = str(tmpdir)
    proto.set_output_folder(str(tmpdir / 'output'))
    proto.set_model('test/models/real/%s.cellml' % model_name)
    for input in ['max_paces', 'max_steady_state_beats']:
        try:
            proto.set_input(input, fc.language.values.Simple(1000))
        except Exception:
            pass  # Input doesn't exist
    proto.run()

    if expected_outputs:
        print('Checking results against reference data')
        test_support.check_results(
            proto,
            expected_outputs,
            data_folder,
            rel_tol=0.005,
            abs_tol=2.5e-4,
        )
    print(f'Protocol {proto_name} succeeded on {model_name} and results match where available')
