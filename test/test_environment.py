"""Test environment and delegations and associated functions."""
import numpy as np
import pytest

import fc
import fc.environment as Env
import fc.language.values as V
from fc.error_handling import ProtocolError
from fc.simulations.model import TestOdeModel


def test_sim_env_txt():
    proto_file = 'test/protocols/test_sim_environments.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_sim_env_txt')
    proto.set_model(TestOdeModel(1))
    proto.run()


def test_defining_names():
    env = Env.Environment()
    one = V.Simple(1)
    env.define_name("one", one)
    assert env.look_up("one") == one
    assert len(env) == 1
    assert env.defined_names() == ["one"]

    # value must be a value type
    with pytest.raises(ProtocolError):
        env.define_name("one", 1)

    # already defined
    two = V.Simple(2)
    with pytest.raises(ProtocolError):
        env.define_name("one", two)

    # don't have permission to overwrite
    with pytest.raises(ProtocolError):
        env.overwrite_definition("one", two)

    names = ["two", "three", "four"]
    values = [V.Simple(2), V.Simple(3), V.Simple(4)]
    env.define_names(names, values)

    # already defined
    with pytest.raises(ProtocolError):
        env.define_names(names, values)

    assert len(env) == 4
    for i, name in enumerate(names):
        assert env.look_up(names[i]) == values[i]
    env2 = Env.Environment()
    env2.merge(env)
    fresh1 = env.fresh_ident()
    fresh2 = env2.fresh_ident()
    assert fresh1 != fresh2
    assert sorted(env.defined_names()) == sorted(env2.defined_names())

    env.clear()
    assert len(env) == 0

    env.define_name("one", one)
    assert env.look_up("one") == one


def test_overwriting_env():
    env = Env.Environment()
    one = V.Simple(1)
    env.define_name("one", one)
    assert env.look_up("one") == one
    two = V.Simple(2)
    with pytest.raises(ProtocolError):
        env.define_name("one", two)  # already defined
    with pytest.raises(ProtocolError):
        env.overwrite_definition("one", two)  # don't have permission to overwrite
    with pytest.raises(ProtocolError):
        env.remove("one")  # don't have permission to overwrite
    env.allow_overwrite = True
    env.overwrite_definition("one", two)
    assert env.look_up("one") == two
    env.remove("one")
    with pytest.raises(KeyError):
        env.look_up("one")  # item was removed
    env.define_name("one", one)
    assert env.look_up("one") == one
    with pytest.raises(ProtocolError):
        env.remove("three")  # never added


def test_delegation():
    root_env = Env.Environment()
    middle_env = Env.Environment(delegatee=root_env)
    top_env = Env.Environment()
    top_env.set_delegatee_env(middle_env)

    name = "name"
    value = V.Simple(123.4)
    root_env.define_name(name, value)
    assert top_env.look_up(name) == value

    value2 = V.Simple(432.1)
    middle_env.define_name(name, value2)
    assert top_env.look_up(name) == value2

    value3 = V.Simple(6.5)
    top_env.define_name(name, value3)
    assert top_env.look_up(name) == value3


def test_prefixed_delegation():
    root_env = Env.Environment()
    env_a = Env.Environment()
    env_b = Env.Environment()
    env_a.define_name("A", V.Simple(1))
    env_b.define_name("B", V.Simple(2))
    root_env.set_delegatee_env(env_a, 'a')
    root_env.set_delegatee_env(env_b, 'b')

    with pytest.raises(ProtocolError):
        env_a.define_name('a:b', V.Simple(1))
    assert root_env.look_up('a:A').value == 1
    assert root_env.look_up('b:B').value == 2
    with pytest.raises(KeyError):
        root_env.look_up('a:n')
    with pytest.raises(KeyError):
        root_env.look_up('c:c')

    env_aa = Env.Environment(delegatee=env_a)
    env_aa.define_name('A', V.Simple(3))
    env_a.set_delegatee_env(env_aa, 'a')
    assert env_a.look_up('a:A').value == 3
    assert root_env.look_up('a:a:A').value == 3


def test_evaluate_expr_and_stmt():
    # basic mathml function
    env = Env.Environment()
    expr_str = 'MathML:max(100, 115, 98)'
    assert env.evaluate_expr(expr_str, env).value == 115

    # array creation
    arr = V.Array(np.arange(10))
    env.define_name('arr', arr)
    expr_str = 'arr[1:2:10]'
    predicted = np.array([1, 3, 5, 7, 9])
    np.testing.assert_array_almost_equal(predicted, env.evaluate_expr(expr_str, env).array)

    # view of an array
    view_arr = V.Array(np.arange(10))
    env.define_name('view_arr', view_arr)
    expr_str = 'view_arr[4]'
    predicted = np.array(4)
    np.testing.assert_array_almost_equal(env.evaluate_expr(expr_str, env).array, predicted)

    # statement list
    stmt_str = 'z = lambda a: a+2\nassert z(2) == 4'
    env.evaluate_statement(stmt_str, env)  # assertion built into list, no extra test needed
