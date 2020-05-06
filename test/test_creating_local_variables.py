"""
Tests creating variables with the ``var`` construct.
"""
import os
import pytest

import fc
from fc.error_handling import ProtocolError


def test_var_declarations():
    # Tests the ``var`` construct in a couple of different legal applications

    proto = fc.Protocol('test/protocols/test_var_declaration.txt')
    proto.set_output_folder('test_var_declarations')
    proto.set_model('test/real/models/beeler_reuter_model_1977.cellml')
    proto.run()

    # Test assertions are within the protocol itself
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))


def test_var_declaration_redefined():
    # Tests an error is raised if multiple ``var`` statements declare the same name

    proto = fc.Protocol('test/protocols/test_var_declaration_redefined.txt')
    proto.set_output_folder('test_var_declaration_redefined')
    with pytest.raises(ProtocolError, match='more than one var statement'):
        proto.set_model('test/real/models/beeler_reuter_model_1977.cellml')


def test_var_declaration_underdefined():
    # Tests an error is raised if a ``var`` variable isn't given a value

    proto = fc.Protocol('test/protocols/test_var_declaration_underdefined.txt')
    proto.set_output_folder('test_var_declaration_underdefined')
    with pytest.raises(ProtocolError, match='No definition given for local variable'):
        proto.set_model('test/real/models/beeler_reuter_model_1977.cellml')


def test_var_declaration_overdefined():
    # Tests an error is raised if a ``var`` variable is given an initial value and a non-ODE equation in a define

    proto = fc.Protocol('test/protocols/test_var_declaration_overdefined.txt')
    proto.set_output_folder('test_var_declaration_overdefined')
    with pytest.raises(ProtocolError, match='which is not a state variable'):
        proto.set_model('test/real/models/beeler_reuter_model_1977.cellml')

