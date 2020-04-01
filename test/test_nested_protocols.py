"""Tests protocols that use nesting."""
import fc
from fc.simulations.model import TestOdeModel


def test_nested_protocols():
    proto_file = 'test/protocols/test_nested_protocol.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('TestVariousProtocols_TestNestedProtocols')
    proto.set_model('test/models/luo_rudy_1991.cellml')
    proto.run()
    assert 'always_missing' not in proto.output_env
    assert 'first_missing' not in proto.output_env
    assert 'some_missing' not in proto.output_env
    assert 'first_present' not in proto.output_env


def test_parallel_nested_txt():
    # NB: In the current Python implementation this doesn't actually parallelise!
    proto_file = 'test/protocols/test_parallel_nested.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('TestVariousProtocols_TestParallelNestedTxt')
    proto.set_model(TestOdeModel(1))
    proto.run()
    proto.model.reset_state()
    proto.run()

