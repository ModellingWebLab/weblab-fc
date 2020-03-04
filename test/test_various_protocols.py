
import pytest
import unittest

import fc
import fc.language.expressions as E
from fc.simulations.model import TestOdeModel


class TestVariousProtocols(unittest.TestCase):
    """Test that various test protocols are executed correctly."""

    def test_nested_protocols(self):
        proto_file = 'test/protocols/test_nested_protocol.txt'
        proto = fc.Protocol(proto_file)
        proto.set_output_folder('TestVariousProtocols_TestNestedProtocols')
        proto.set_model('test/models/luo_rudy_1991.cellml')
        proto.run()
        self.assertNotIn('always_missing', proto.output_env)
        self.assertNotIn('first_missing', proto.output_env)
        self.assertNotIn('some_missing', proto.output_env)
        self.assertNotIn('first_present', proto.output_env)

    @pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
    def test_annotating_with_other_ontologies(self):
        proto_file = 'test/protocols/test_other_ontologies.txt'
        proto = fc.Protocol(proto_file)
        proto.set_output_folder('TestVariousProtocols_TestAnnotatingWithOtherOntologies')
        proto.set_model('test/data/test_lr91.cellml')
        proto.run()

    def test_parallel_nested_txt(self):
        # NB: In the current Python implementation this doesn't actually parallelise!
        proto_file = 'test/protocols/test_parallel_nested.txt'
        proto = fc.Protocol(proto_file)
        proto.set_output_folder('TestVariousProtocols_TestParallelNestedTxt')
        proto.set_model(TestOdeModel(1))
        proto.run()
        proto.model.reset_state()
        proto.run()

    def test_sim_env_txt(self):
        proto_file = 'test/protocols/test_sim_environments.txt'
        proto = fc.Protocol(proto_file)
        proto.set_output_folder('TestVariousProtocols_TestSimEnv')
        proto.set_model(TestOdeModel(1))
        proto.run()

    def test_while_loop_txt(self):
        proto_file = 'test/protocols/test_while_loop.txt'
        proto = fc.Protocol(proto_file)
        proto.set_output_folder('TestVariousProtocols_TestWhileLoopTxt')
        proto.set_model(TestOdeModel(1))
        proto.set_input('num_iters', E.N(10))
        proto.run()
