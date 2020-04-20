"""
Test lambda expressions and all function methods.
"""

import pytest

import fc.language.expressions as E
import fc.language.statements as S
import fc.language.values as V
import fc.environment as Env
from fc.error_handling import ProtocolError


def test_func_definitions():
    env = Env.Environment()
    parameters = ["a", "b"]
    body = [S.Return(E.NameLookUp("b"), E.NameLookUp("a"))]
    swap = E.LambdaExpression(parameters, body)
    env.execute_statements([S.Assign(["swap"], swap)])
    args = [E.N(1), E.N(2)]
    swap_call = E.FunctionCall("swap", args)
    result = swap_call.evaluate(env)
    assert isinstance(result, V.Tuple)
    assert result.values[0].value == pytest.approx(2)
    assert result.values[1].value == pytest.approx(1)
    env.execute_statements([S.Assign(parameters, swap_call)])
    defined = env.defined_names()
    assert len(defined) == 3
    assert env.look_up('a').value == 2
    assert env.look_up('b').value == 1

    args = [E.N(3), E.N(5)]
    swap_call = E.FunctionCall("swap", args)
    result = swap_call.evaluate(env)
    assert isinstance(result, V.Tuple)
    assert result.values[0].value == pytest.approx(5)
    assert result.values[1].value == pytest.approx(3)


def test_lambda_expression_wrap():
    env = Env.Environment()
    add = E.LambdaExpression.wrap(E.Plus, 3)
    args = [E.N(1), E.N(2), E.N(3)]
    add_call = E.FunctionCall(add, args)
    result = add_call.evaluate(env)
    assert result.value == 6


def test_nested_function():
    env = Env.Environment()
    nested_body = [S.Return(E.Plus(E.NameLookUp('input'), E.NameLookUp('outer_var')))]
    nested_function = E.LambdaExpression(["input"], nested_body)
    body = [S.Assign(["nested_fn"], nested_function),
            S.Assign(["outer_var"], E.N(1)),
            S.Return(E.Eq(E.FunctionCall("nested_fn", [E.N(1)]), E.N(2)))]
    nested_scope = E.LambdaExpression([], body)
    nested_call = E.FunctionCall(nested_scope, [])
    result = nested_call.evaluate(env)
    assert result.value == 1


def test_functions_with_defaults_used():
    # Function has default which is used
    env = Env.Environment()
    nested_body = [S.Return(E.Plus(E.NameLookUp('input'), E.NameLookUp('outer_var')))]
    nested_function = E.LambdaExpression(["input"], nested_body, default_parameters=[V.Simple(1)])
    body = [S.Assign(["nested_fn"], nested_function),
            S.Assign(["outer_var"], E.N(1)),
            S.Return(E.Eq(E.FunctionCall("nested_fn", [E.Const(V.DefaultParameter())]), E.N(2)))]
    nested_scope = E.LambdaExpression([], body)
    nested_call = E.FunctionCall(nested_scope, [])
    result = nested_call.evaluate(env)
    assert result.value == 1


def test_functions_with_defaults_unused():
    # Function has default, but value is explicitly assigned in this case
    env = Env.Environment()
    nested_body = [S.Return(E.Plus(E.NameLookUp('input'), E.NameLookUp('outer_var')))]
    nested_function = E.LambdaExpression(["input"], nested_body, default_parameters=[V.Simple(0)])
    body = [S.Assign(["nested_fn"], nested_function),
            S.Assign(["outer_var"], E.N(1)),
            S.Return(E.Eq(E.FunctionCall("nested_fn", [E.N(1)]), E.N(2)))]
    nested_scope = E.LambdaExpression([], body)
    nested_call = E.FunctionCall(nested_scope, [])
    result = nested_call.evaluate(env)
    assert result.value == 1


def test_multiple_default_values():
    env = Env.Environment()
    parameters = ['a', 'b', 'c']
    body = [S.Return(E.Plus(E.NameLookUp('a'), E.NameLookUp('b'), E.NameLookUp('c')))]
    add = E.LambdaExpression(parameters, body, default_parameters=[V.Simple(1), V.Simple(2), V.Simple(3)])
    args = [E.Const(V.DefaultParameter()), E.Const(V.DefaultParameter()), E.Const(V.DefaultParameter())]
    add_call = E.FunctionCall(add, args)
    result = add_call.evaluate(env)
    assert result.value == 6

    args = [E.N(3)]
    add_call = E.FunctionCall(add, args)
    result = add_call.evaluate(env)
    assert result.value == 8

    args = [E.Const(V.DefaultParameter()), E.Const(V.DefaultParameter()), E.N(1)]
    add_call = E.FunctionCall(add, args)
    result = add_call.evaluate(env)
    assert result.value == 4

    args = [E.N(4), E.Const(V.DefaultParameter()), E.N(4)]
    add_call = E.FunctionCall(add, args)
    result = add_call.evaluate(env)
    assert result.value == 10


def test_assert_statement():
    env = Env.Environment()
    # evaluates to one, assertion should pass
    env.execute_statements([S.Assert(E.N(1))])

    # evaluates to zero, assertion should fail
    with pytest.raises(ProtocolError):
        env.execute_statements([S.Assert(E.N(0))])

    # evaluates to non-simple value, assertion should fail
    with pytest.raises(ProtocolError):
        env.execute_statements([S.Assert(E.Const(V.DefaultParameter()))])

