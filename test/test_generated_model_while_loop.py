"""

Run the `test_while_loop` protocol on a generated model.

"""
import fc
import fc.language.expressions as E
import os


def test_generated_model_while_loop():

    # Create protocol
    proto = fc.Protocol(os.path.join(
        'test', 'protocols', 'test_while_loop_TEMP.txt'))

    # Set model (generates & compiles model)
    model_name = 'hodgkin_huxley_squid_axon_model_1952_modified'
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))

    # Set protocol input
    proto.set_input('num_iters', E.N(10))

    # Run protocol
    # Test assertions are within the protocol itself
    proto.set_output_folder('test_generated_model_while_loop')
    proto.run()

