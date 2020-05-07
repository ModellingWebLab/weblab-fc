"""
Test parsing and running a few of the simplest protocols
"""

import fc
from fc.simulations import model


def test_find_index():
    proto_file = 'test/protocols/test_find_index.txt'
    proto = fc.Protocol(proto_file)
    proto.run()


def test_core_postproc():
    proto_file = 'test/protocols/test_core_postproc.txt'
    proto = fc.Protocol(proto_file)
    proto.run()


def test_graph_state():
    proto_file = 'test/protocols/real/GraphState.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_graph_txt')
    proto.set_model(model.TestOdeModel(1))
    proto.run()
