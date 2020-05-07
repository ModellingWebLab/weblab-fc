"""
Tests that empty (but present) sections are handled without errors.
"""
import fc


def test_empty_sections():

    proto = fc.Protocol('test/protocols/test_empty_sections.txt')
    proto.set_output_folder('test_empty_sections')
    proto.set_model('test/models/simple_ode_model.cellml')
    proto.run()

