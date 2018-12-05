
try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from setproctitle import setproctitle
except ImportError:
    def setproctitle(t):
        return None

import argparse
import multiprocessing
import os
import sys
import traceback
from io import StringIO

import fc
import fc.utility.test_support as TestSupport


class RedirectStdStreams(object):
    """Context manager to redirect standard streams, from http://stackoverflow.com/a/6796752/2639299.

    TODO: Consider if we can redirect for child processes (model compilation) too.
    It's trickier as they write to os-level file handles.
    """

    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def WorkerInit():
    """Function run on initialization of a Pool worker process.

    If the setproctitle module is installed, this will adjust the process title (shown by ps) to be more informative.
    """
    setproctitle('python worker %d' % TestSupport.GetProcessNumber())


def RunExperiment(modelName, protoName, expectedOutputs):
    """Worker function to run a single experiment, i.e. application of a protocol to a model.

    :param modelName: name of model to run, i.e. no path or extension
    :param protoName: name of protocol to run, i.e. no path or extension
    :param expectedOutputs: dictionary of outputs to check against reference data,
        mapping output name to number of dimensions
    :returns: a tuple (modelName, protoName, boolean result, output text, error/warning messages)
    """
    messages = []
    result = True
    output = StringIO()
    proto = None
    with RedirectStdStreams(output, output):
        try:
            print("Applying", protoName, "to", modelName, "on process", TestSupport.GetProcessNumber(), "of", CHASTE_NUM_PROCS)
            setproctitle('python worker %d running %s on %s' % (TestSupport.GetProcessNumber(), protoName, modelName))
            proto = fc.Protocol('projects/FunctionalCuration/protocols/%s.txt' % protoName)
            proto.SetOutputFolder(os.path.join(CHASTE_TEST_OUTPUT, 'Py_FunctionalCuration', modelName, protoName))
            proto.SetModel('projects/FunctionalCuration/cellml/%s.cellml' % modelName)
            for input in ['max_paces', 'max_steady_state_beats']:
                try:
                    proto.SetInput(input, fc.language.values.Simple(1000))
                except:
                    pass  # Input doesn't exist
            proto.Run()
        except:
            result = False
            messages.append(traceback.format_exc())
        try:
            if expectedOutputs and proto:
                outputs_match = TestSupport.CheckResults(proto, expectedOutputs,
                                                         'projects/FunctionalCuration/test/data/historic/%s/%s' % (
                                                             modelName, protoName),
                                                         rtol=0.005, atol=2.5e-4, messages=messages)
                if outputs_match is None:
                    if result:
                        messages.append("Experiment succeeded but produced no results, and this was expected!")
                    else:
                        messages.append("Note: experiment was expected to fail - no reference results stored.")
                        result = True
                else:
                    result = result and outputs_match
        except:
            result = False
            messages.append(traceback.format_exc())
    return (modelName, protoName, result, output.getvalue(), messages)


class Defaults(object):
    """Simple encapsulation of default settings for the test suite below."""
    # List of model names
    models = ["aslanidi_atrial_model_2009",
              "aslanidi_Purkinje_model_2009",
              "beeler_reuter_model_1977",
              # "benson_epicardial_2008",
              # "bernus_wilders_zemlin_verschelde_panfilov_2002",
              # "bondarenko_szigeti_bett_kim_rasmusson_2004_apical",
              "carro_2011_epi",
              # "clancy_rudy_2002",
              # "courtemanche_ramirez_nattel_1998",
              "decker_2009",
              "difrancesco_noble_model_1985",
              # "earm_noble_model_1990",
              "fink_noble_giles_model_2008",
              "grandi_pasqualini_bers_2010_ss",
              # "iyer_2004",
              # "iyer_model_2007",
              "li_mouse_2010",
              "luo_rudy_1991",
              "mahajan_shiferaw_2008",
              # "maleckar_model_2009",
              # "matsuoka_model_2003",
              # "noble_model_1991",
              # "noble_model_1998",
              "ohara_rudy_2011_epi",
              # "priebe_beuckelmann_1998",
              # "sachse_moreno_abildskov_2008_b",
              "shannon_wang_puglisi_weber_bers_2004",
              # "ten_tusscher_model_2004_endo",
              "ten_tusscher_model_2006_epi"
              # "winslow_model_1999",
              # "zhang_SAN_model_2000_0D_capable"
              ]

    # Map from protocol name to expected outputs (names & dimensions)
    protocolOutputs = {"ExtracellularPotassiumVariation": {"scaled_APD90": 1, "scaled_resting_potential": 1, "detailed_voltage": 2},
                       "GraphState": {"state": 2},
                       "ICaL": {"min_LCC": 2, "final_membrane_voltage": 1},
                       # "ICaL_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       # "ICaL_IV_curve": {"normalised_peak_currents": 1},
                       # "IK1_block": {"scaled_resting_potential": 1, "scaled_APD90": 1, "detailed_voltage": 2},
                       "IK1_IV_curve": {"normalised_low_K1": 1, "normalised_high_K1": 1},
                       "IKr_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       "IKr_IV_curve": {"normalised_peak_Kr_tail": 1},
                       # "IKs_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       # "IKs_IV_curve": {"normalised_peak_Ks_tail": 1},
                       "INa_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       "INa_IV_curves": {"normalised_peak_currents": 1, "current_activation": 2},
                       # "Ito_block": {"scaled_resting_potential": 1, "scaled_APD90": 1, "detailed_voltage": 2},
                       "NCX_block": {"scaled_resting_potential": 1, "scaled_APD90": 1, "detailed_voltage": 2},
                       "RyR_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       "S1S2": {"S1S2_slope": 1},
                       "SteadyStateRestitution": {"APD": 2, "restitution_slope": 1},
                       "SteadyStateRunner": {"num_paces": 0, "detailed_voltage": 1},
                       "SteadyStateRunner0_5Hz": {"num_paces": 0, "detailed_voltage": 1},
                       # "SteadyStateRunner2Hz": {"num_paces": 0, "detailed_voltage": 1},
                       # "SteadyStateRunner3Hz": {"num_paces": 0, "detailed_voltage": 1},
                       "SteadyStateRunner4Hz": {"num_paces": 0, "detailed_voltage": 1},
                       }


class TestPythonReproducibility(unittest.TestCase):
    """Test reproducibility of the Python implementation of Functional Curation.

    We compare the results of many protocols applied to many models against stored results produced with
    the original (C++) implementation.

    This module also demonstrates how to handle optional run-time flags passed to a Python test,
    and test execution in parallel.
    """
    # We keep a record of all failures to summarise at the end
    failures = []

    def TestExperimentReproducibility(self):
        # Get the first result, when available
        model, protocol, resultCode, output, messages = self.results.pop().get()
        if resultCode and len(output) > 450:
            print(output[:100] + "\n...\n" + output[-300:], end=' ')
        else:
            print(output, end=' ')
        print("Applied", protocol, "to", model)
        for i, message in enumerate(messages):
            print("%d) %s" % (i+1, message))
        if not resultCode:
            self.failures.append(model + " / " + protocol)
        self.assertTrue(resultCode, "Experiment %s / %s failed" % (model, protocol))

    @classmethod
    def setUpClass(cls):
        """Set up a pool of workers for executing experiments, and submit all experiments to the pool."""
        # Set up parallel execution if available (if not, we have just one worker)
        cls.pool = multiprocessing.Pool(processes=CHASTE_NUM_PROCS, initializer=WorkerInit)
        if CHASTE_NUM_PROCS > 1:
            # If we're parallelising at this level, don't parallelise internally
            import numexpr
            numexpr.set_num_threads(1)

        # Loop over experiments and submit to the pool
        cls.results = []
        for model in cls.options.models:
            try:
                os.makedirs(os.path.join(CHASTE_TEST_OUTPUT, 'Py_FunctionalCuration', model))
            except os.error:
                pass
            for protocol in cls.options.protocols:
                cls.results.append(cls.pool.apply_async(RunExperiment,
                                                        args=(model, protocol, Defaults.protocolOutputs[protocol])))

        # Disallow further job submission to the pool
        cls.pool.close()

    @classmethod
    def tearDownClass(cls):
        """Wait for all workers to exit, and summarise failures."""
        cls.pool.join()
        if cls.failures:
            print("\nThe following model/protocol combinations failed unexpectedly:")
            for failure in sorted(cls.failures):
                print("  ", failure)


def MakeTestSuite():
    """Build a suite where each test checks one experiment.

    This instantiates the TestPythonReproducibility class once for each model/protocol combination.
    """
    suite = unittest.TestSuite()
    # Get list of models & protocols, either defaults or from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help="list of model names to test (i.e. no path or extension)")
    parser.add_argument('--protocols', nargs='+', help="list of protocol names to test (i.e. no path or extension)")
    options = TestPythonReproducibility.options = parser.parse_args()
    if not options.models:
        options.models = Defaults.models
    if not options.protocols:
        options.protocols = list(Defaults.protocolOutputs.keys())
    # Add test cases to the suite
    for model in options.models:
        for protocol in options.protocols:
            suite.addTest(TestPythonReproducibility('TestExperimentReproducibility'))
    return suite
