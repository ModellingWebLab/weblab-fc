"""Tests protocols that use nesting."""
import os

import fc
import fc.language.values as V
from fc.simulations.model import TestOdeModel


def test_nested_protocols():
    proto_file = 'test/protocols/test_nested_protocol.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_nested_protocols')
    proto.set_model('test/models/luo_rudy_1991.cellml')
    proto.run()
    assert 'always_missing' not in proto.output_env
    assert 'first_missing' not in proto.output_env
    assert 'some_missing' not in proto.output_env
    assert 'first_present' not in proto.output_env


def test_merging_interfaces():
    # Checks that merging model interface sections from nested protocols works
    proto_file = 'protocols/IK1_block.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_merging_interfaces')
    proto.set_model('test/models/luo_rudy_1991.cellml')

    # Make the run shorter for testing
    proto.set_input('block_levels', V.Array([0.0, 0.5]))

    proto.run()
    # Just check output exists
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))


def test_parallel_nested_protocol():
    # NB: In the current Python implementation this doesn't actually parallelise!
    proto_file = 'test/protocols/test_parallel_nested.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_parallel_nested_protocol')
    proto.set_model(TestOdeModel(1))
    proto.run()
    proto.model.reset_state()
    proto.run()

