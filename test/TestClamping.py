
import os
import pytest
import unittest

import fc


class TestClamping(unittest.TestCase):
    """Test implementation of voltage clamps in the Python code.

    Note that the first two cases in TestClamping.hpp are covered in TestAlgebraicModels.py!
    """

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def testClampingWithVariableUnits(self):
        proto = fc.Protocol('test/protocols/test_clamping3.txt')
        proto.SetOutputFolder('Py_TestClamping_TestClampingWithVariableUnits')
        proto.set_model('cellml/beeler_reuter_model_1977.cellml')
        proto.Run()
        # Test assertions are within the protocol itself
        self.assertTrue(os.path.exists(os.path.join(proto.outputFolder.path, 'output.h5')))

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def testClampToDataFile(self):
        proto_file = 'protocols/timecourse_voltage_clamp.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('Py_TestClamping_TestClampToDataFile')
        proto.set_model('cellml/ten_tusscher_model_2004_epi.cellml')
        proto.Run()
        # Test assertions are within the protocol itself
        self.assertTrue(os.path.exists(os.path.join(proto.outputFolder.path, 'output.h5')))

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def testInterpolationClamp(self):
        proto_file = 'test/protocols/test_model_interpolation.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('Py_TestClamping_TestInterpolationClamp')
        proto.set_model('test/data/simple_ode.cellml')
        proto.Run()
        # Test assertions are within the protocol itself
        self.assertTrue(os.path.exists(os.path.join(proto.outputFolder.path, 'output.h5')))
