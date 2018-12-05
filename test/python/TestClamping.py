
import os
import unittest

import fc


class TestClamping(unittest.TestCase):
    """Test implementation of voltage clamps in the Python code.

    Note that the first two cases in TestClamping.hpp are covered in TestAlgebraicModels.py!
    """

    def TestClampingWithVariableUnits(self):
        proto = fc.Protocol('projects/FunctionalCuration/test/protocols/test_clamping3.txt')
        proto.SetOutputFolder('Py_TestClamping_TestClampingWithVariableUnits')
        proto.SetModel('projects/FunctionalCuration/cellml/beeler_reuter_model_1977.cellml')
        proto.Run()
        # Test assertions are within the protocol itself
        self.assertTrue(os.path.exists(os.path.join(proto.outputFolder.path, 'output.h5')))

    def TestClampToDataFile(self):
        proto_file = 'projects/FunctionalCuration/protocols/timecourse_voltage_clamp.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('Py_TestClamping_TestClampToDataFile')
        proto.SetModel('projects/FunctionalCuration/cellml/ten_tusscher_model_2004_epi.cellml')
        proto.Run()
        # Test assertions are within the protocol itself
        self.assertTrue(os.path.exists(os.path.join(proto.outputFolder.path, 'output.h5')))

    def TestInterpolationClamp(self):
        proto_file = 'projects/FunctionalCuration/test/protocols/test_model_interpolation.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('Py_TestClamping_TestInterpolationClamp')
        proto.SetModel('projects/FunctionalCuration/test/data/simple_ode.cellml')
        proto.Run()
        # Test assertions are within the protocol itself
        self.assertTrue(os.path.exists(os.path.join(proto.outputFolder.path, 'output.h5')))
