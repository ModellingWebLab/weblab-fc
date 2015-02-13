"""Copyright (c) 2005-2015, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import argparse
import multiprocessing
import os
import sys
import traceback
from cStringIO import StringIO

import fc
import fc.utility.test_support as TestSupport

class RedirectStdStreams(object):
    """Context manager to redirect standard streams, from http://stackoverflow.com/a/6796752/2639299.
    
    TODO: Consider if we can redirect for child processes (model compilation) too.  It's trickier as they write to os-level file handles.
    """
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

def RunExperiment(modelName, protoName, expectedOutputs):
    """Worker function to run a single experiment, i.e. application of a protocol to a model.

    :param modelName: name of model to run, i.e. no path or extension
    :param protoName: name of protocol to run, i.e. no path or extension
    :param expectedOutputs: dictionary of outputs to check against reference data, mapping output name to number of dimensions
    :returns: a tuple (modelName, protoName, boolean result, output text, error/warning messages)
    """
    messages = []
    result = True
    output = StringIO()
    with RedirectStdStreams(output, output):
        try:
            print "Applying", protoName, "to", modelName, "on process", TestSupport.GetProcessNumber(), "of", CHASTE_NUM_PROCS
            proto = fc.Protocol('projects/FunctionalCuration/protocols/%s.txt' % protoName)
            proto.SetOutputFolder(os.path.join(CHASTE_TEST_OUTPUT, 'Py_FunctionalCuration', modelName, protoName))
            proto.SetModel('projects/FunctionalCuration/cellml/%s.cellml' % modelName)
            for input in ['num_paces', 'max_steady_state_beats']:
                try:
                    proto.SetInput(input, fc.language.values.Simple(1000))
                except:
                    pass # Input doesn't exist
            proto.Run()
        except:
            result = False
            messages.append(traceback.format_exc())
        if expectedOutputs:
            outputs_match = TestSupport.CheckResults(proto, expectedOutputs,
                                                     'projects/FunctionalCuration/test/data/historic/%s/%s' % (modelName, protoName),
                                                     rtol=0.005, atol=1e-4, messages=messages)
            if outputs_match is None:
                if result:
                    messages.append("Experiment succeeded but produced no results, and this was expected!")
                else:
                    messages.append("Note: experiment was expected to fail - no reference results stored.")
                    result = True
            else:
                result = result and outputs_match
    return (modelName, protoName, result, output.getvalue(), messages)

class Defaults(object):
    """Simple encapsulation of default settings for the test suite below."""
    # List of model names
    models = ["aslanidi_atrial_model_2009",
              "aslanidi_Purkinje_model_2009",
              "beeler_reuter_model_1977",
              #"benson_epicardial_2008",
              #"bernus_wilders_zemlin_verschelde_panfilov_2002",
              "bondarenko_szigeti_bett_kim_rasmusson_2004_apical",
              "carro_2011_epi",
              #"clancy_rudy_2002",
              #"courtemanche_ramirez_nattel_1998",
              "decker_2009",
              "difrancesco_noble_model_1985",
              #"earm_noble_model_1990",
              "fink_noble_giles_model_2008",
              "grandi_pasqualini_bers_2010_ss",
              "iyer_2004",
              #"iyer_model_2007",
              "li_mouse_2010",
              "luo_rudy_1991",
              "mahajan_shiferaw_2008",
              #"maleckar_model_2009",
              #"matsuoka_model_2003",
              #"noble_model_1991",
              #"noble_model_1998",
              "ohara_rudy_2011_epi",
              "priebe_beuckelmann_1998",
              #"sachse_moreno_abildskov_2008_b",
              "shannon_wang_puglisi_weber_bers_2004",
              #"ten_tusscher_model_2004_endo",
              "ten_tusscher_model_2006_epi",
              #"winslow_model_1999"
              ]

    # Map from protocol name to expected outputs (names & dimensions)
    protocolOutputs = {"ExtracellularPotassiumVariation": {"scaled_APD90": 1, "scaled_resting_potential": 1, "detailed_voltage": 2},
                       "GraphState": {"state": 2},
                       "ICaL": {"min_LCC": 2, "final_membrane_voltage": 1},
                       #"ICaL_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       #"ICaL_IV_curve": {"normalised_peak_currents": 1},
                       #"IK1_block": {"scaled_resting_potential": 1, "scaled_APD90": 1, "detailed_voltage": 2},
                       "IK1_IV_curve": {"normalised_low_K1": 1, "normalised_high_K1": 1},
                       "IKr_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       "IKr_IV_curve": {"normalised_peak_Kr_tail": 1},
                       #"IKs_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       #"IKs_IV_curve": {"normalised_peak_Ks_tail": 1},
                       "INa_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       "INa_IV_curves": {"normalised_peak_currents": 1, "current_activation": 2},
                       #"Ito_block": {"scaled_resting_potential": 1, "scaled_APD90": 1, "detailed_voltage": 2},
                       "NCX_block": {"scaled_resting_potential": 1, "scaled_APD90": 1, "detailed_voltage": 2},
                       "RyR_block": {"scaled_APD90": 1, "detailed_voltage": 2},
                       "S1S2": {"APD90": 1},
                       "SteadyStateRestitution": {"APD": 2, "restitution_slope": 1},
                       "SteadyStateRunner": {"num_paces": 0, "detailed_voltage": 1},
                       "SteadyStateRunner0_5Hz": {"num_paces": 0, "detailed_voltage": 1},
                       #"SteadyStateRunner2Hz": {"num_paces": 0, "detailed_voltage": 1},
                       #"SteadyStateRunner3Hz": {"num_paces": 0, "detailed_voltage": 1},
                       "SteadyStateRunner4Hz": {"num_paces": 0, "detailed_voltage": 1},
                       }

class TestPythonReproducibility(unittest.TestCase):
    """Test reproducibility of the Python implementation of Functional Curation.

    We compare the results of many protocols applied to many models against stored results produced with
    the original (C++) implementation.

    This module also demonstrates how to handle optional run-time flags passed to a Python test,
    and test execution in parallel.
    """
    def TestExperimentReproducibility(self):
        # Get the first result, when available
        model, protocol, resultCode, output, messages = self.results.pop().get()
        print output,
        print "Applied", protocol, "to", model
        for i, message in enumerate(messages):
            print "%d) %s" % (i+1, message)
        self.assert_(resultCode, "Experiment %s / %s failed" % (model, protocol))

    @classmethod
    def setUpClass(cls):
        """Set up a pool of workers for executing experiments, and submit all experiments to the pool."""
        # Set up parallel execution if available (if not, we have just one worker)
        cls.pool = multiprocessing.Pool(processes=CHASTE_NUM_PROCS)
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
                cls.results.append(cls.pool.apply_async(RunExperiment, args=(model, protocol, Defaults.protocolOutputs[protocol])))

        # Disallow further job submission to the pool
        cls.pool.close()
    
    @classmethod
    def tearDownClass(cls):
        """Wait for all workers to exit."""
        cls.pool.join()

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
        options.protocols = Defaults.protocolOutputs.keys()
    # Add test cases to the suite
    for model in options.models:
        for protocol in options.protocols:
            suite.addTest(TestPythonReproducibility('TestExperimentReproducibility'))
    return suite
