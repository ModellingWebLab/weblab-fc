"""

Run the `test_while_loop` protocol on a generated model.

"""
import fc
import fc.language.expressions as E
import os


def test_while_loop_on_simple_ode_model():
    # This model has the same ODE that the protocol defines, so is useful for testing
    # the input and time units constructs before we've implemented 'define'.

    # Create protocol
    proto = fc.Protocol(os.path.join(
        'test', 'protocols', 'test_while_loop.txt'))

    # Set model (generates & compiles model)
    model_name = 'single_ode'
    proto.set_model(os.path.join('test', 'models', model_name + '.cellml'))

    # Set protocol input
    proto.set_input('num_iters', E.N(10))

    # Run protocol
    # Test assertions are within the protocol itself
    proto.set_output_folder('test_generated_model_while_loop')
    proto.run()

