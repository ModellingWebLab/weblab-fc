"""Test while loop support."""
import os
import fc
import fc.language.expressions as E
from fc.simulations.model import TestOdeModel


def test_while_loop_txt():
    proto_file = 'test/protocols/test_while_loop.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_while_loop_txt')
    proto.set_model(TestOdeModel(1))
    proto.set_input('num_iters', E.N(10))
    proto.run()


def test_while_loop_on_hh_model():
    # Create protocol
    proto = fc.Protocol(os.path.join(
        'test', 'protocols', 'test_while_loop.txt'))

    # Set model (generates & compiles model)
    model_name = 'hodgkin_huxley_squid_axon_model_1952_modified'
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))

    # Set protocol input
    proto.set_input('num_iters', E.N(10))

    # Run protocol
    # Test assertions are within the protocol itself
    proto.set_output_folder('test_while_loop_on_hh_model')
    proto.run()

