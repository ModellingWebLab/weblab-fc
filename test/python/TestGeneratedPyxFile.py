
import pytest
import unittest

import fc


class TestGeneratedPyx(unittest.TestCase):
    """

    Temporary test for `fccodegen` generated `pyx` model.

    """

    def test_run(self):
        proto = fc.Protocol('test/protocols/test_generated_pyx.txt')
        proto.SetOutputFolder('Py_TestGeneratedPyx_test_run')
        proto.SetModel('test/data/weblab_model.pyx')
        #proto.Run()
        # Test assertions are within the protocol itself

