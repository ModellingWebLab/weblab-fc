
import pytest
import unittest

import fc
import fc.simulations.model as Model
import fc.language.expressions as E


class TestVariousProtocols(unittest.TestCase):
    """Test that various test protocols are executed correctly."""

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def testNestedProtocols(self):
        proto_file = 'test/protocols/test_nested_protocol.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestNestedProtocols')
        proto.SetModel('cellml/luo_rudy_1991.cellml')
        proto.Run()
        self.assertNotIn('always_missing', proto.outputEnv)
        self.assertNotIn('first_missing', proto.outputEnv)
        self.assertNotIn('some_missing', proto.outputEnv)
        self.assertNotIn('first_present', proto.outputEnv)

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def testAnnotatingWithOtherOntologies(self):
        proto_file = 'test/protocols/test_other_ontologies.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestAnnotatingWithOtherOntologies')
        proto.SetModel('test/data/test_lr91.cellml')
        proto.Run()

    def testParallelNestedTxt(self):
        # NB: In the current Python implementation this doesn't actually parallelise!
        proto_file = 'test/protocols/test_parallel_nested.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestParallelNestedTxt')
        proto.SetModel(Model.TestOdeModel(1))
        proto.Run()
        proto.model.ResetState()
        proto.Run()

    def testSimEnvTxt(self):
        proto_file = 'test/protocols/test_sim_environments.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestSimEnv')
        proto.SetModel(Model.TestOdeModel(1))
        proto.Run()

    def testWhileLoopTxt(self):
        proto_file = 'test/protocols/test_while_loop.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestWhileLoopTxt')
        proto.SetModel(Model.TestOdeModel(1))
        proto.SetInput('num_iters', E.N(10))
        proto.Run()
