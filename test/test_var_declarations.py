"""
Tests creating variables with the ``var`` construct.
"""
import pytest

import fc
from fc.error_handling import ProtocolError
from fc.simulations.model import TestOdeModel


def test_var_declarations():
    # Tests the ``var`` construct in a couple of different legal applications

    proto = fc.Protocol('test/protocols/test_var_declaration.txt')
    proto.set_output_folder('test_var_declarations')
    proto.set_model(TestOdeModel(1))
    proto.run()

    # Test assertions are within the protocol itself
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))


def test_var_declaration_underdefined():
    # Tests an error is raised if a ``var`` variable isn't given a value

    proto = fc.Protocol('test/protocols/test_var_declaration_underdefined.txt')
    proto.set_output_folder('test_var_declaration_underdefined')
    with pytest.raises(ProtocolError, 'Lalalala'):
        proto.set_model(TestOdeModel(1))


def test_var_declaration_overdefined():
    # Tests an error is raised if a ``var`` variable is given multiple values

    proto = fc.Protocol('test/protocols/test_var_declaration_overdefined.txt')
    proto.set_output_folder('test_var_declaration_overdefined')
    with pytest.raises(ProtocolError, 'Lalalala'):
        proto.set_model(TestOdeModel(1))

