"""
Test basic math expressions using simple, null, and default values.
"""
import numpy as np
import pytest

import fc.language.expressions as E
import fc.language.values as V
import fc.environment as Env
from fc.error_handling import ProtocolError


def test_addition():
    env = Env.Environment()
    assert E.Plus(E.Const(V.Simple(1)), E.Const(V.Simple(2))).evaluate(env).value == pytest.approx(3)
    assert E.Plus(
        E.Const(V.Simple(1)), E.Const(V.Simple(2)), E.Const(V.Simple(4))).evaluate(env).value == pytest.approx(7)


def test_with_0d_array():
    env = Env.Environment()
    one = V.Array(np.array(1))
    assert one.value == 1
    two = V.Array(np.array(2))
    four = V.Array(np.array(4))
    assert E.Plus(E.Const(one), E.Const(two)).evaluate(env).value == pytest.approx(3)
    assert E.Plus(E.Const(one), E.Const(two), E.Const(four)).evaluate(env).value == pytest.approx(7)
    assert E.Minus(E.Const(one), E.Const(two)).evaluate(env).value == pytest.approx(-1)
    assert E.Power(E.Const(two), E.Const(four)).evaluate(env).value == pytest.approx(16)
    assert E.Times(E.Const(four), E.Const(two)).evaluate(env).value == pytest.approx(8)
    assert E.Root(E.Const(four)).evaluate(env).value == pytest.approx(2)


def test_minus():
    env = Env.Environment()
    assert E.Minus(E.Const(V.Simple(1)), E.Const(V.Simple(2))).evaluate(env).value == pytest.approx(-1)
    assert E.Minus(E.Const(V.Simple(4))).evaluate(env).value == pytest.approx(-4)


def test_times():
    env = Env.Environment()
    assert E.Times(E.Const(V.Simple(6)), E.Const(V.Simple(2))).evaluate(env).value == pytest.approx(12)
    assert E.Times(
        E.Const(V.Simple(6)), E.Const(V.Simple(2)), E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(36)


def test_divide():
    env = Env.Environment()
    assert E.Divide(E.Const(V.Simple(1)), E.Const(V.Simple(2))).evaluate(env).value == pytest.approx(.5)


def test_max():
    env = Env.Environment()
    assert E.Max(
        E.Const(V.Simple(6)), E.Const(V.Simple(12)), E.Const(V.Simple(2))).evaluate(env).value == pytest.approx(12)

def test_min():
    env = Env.Environment()
    assert E.Min(
        E.Const(V.Simple(6)), E.Const(V.Simple(2)), E.Const(V.Simple(12))).evaluate(env).value == pytest.approx(2)

def test_rem():
    env = Env.Environment()
    assert E.Rem(E.Const(V.Simple(6)), E.Const(V.Simple(4))).evaluate(env).value == pytest.approx(2)


def test_power():
    env = Env.Environment()
    assert E.Power(E.Const(V.Simple(2)), E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(8)


def test_root():
    env = Env.Environment()
    assert E.Root(E.Const(V.Simple(16))).evaluate(env).value == pytest.approx(4)
    assert E.Root(E.Const(V.Simple(3)), E.Const(V.Simple(8))).evaluate(env).value == pytest.approx(2)


def test_abs():
    env = Env.Environment()
    assert E.Abs(E.Const(V.Simple(-4))).evaluate(env).value == pytest.approx(4)
    assert E.Abs(E.Const(V.Simple(4))).evaluate(env).value == pytest.approx(4)


def test_floor():
    env = Env.Environment()
    assert E.Floor(E.Const(V.Simple(1.8))).evaluate(env).value == pytest.approx(1)


def test_ceiling():
    env = Env.Environment()
    assert E.Ceiling(E.Const(V.Simple(1.2))).evaluate(env).value == pytest.approx(2)


def test_exp():
    env = Env.Environment()
    assert E.Exp(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(20.0855369231)


def test_ln():
    env = Env.Environment()
    assert E.Ln(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(1.0986122886)


def test_log():
    env = Env.Environment()
    assert E.Log(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(0.4771212547196)
    assert E.Log(E.Const(V.Simple(4)), E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(0.79248125036)


def test_name_look_up():
    env = Env.Environment()
    one = V.Simple(1)
    env.define_name("one", one)
    # Note that evaluate will try to optimise using numexpr, and so always returns a V.Array (0d in this case)
    assert E.NameLookUp("one").interpret(env).value == 1
    np.testing.assert_array_equal(E.NameLookUp("one").evaluate(env).array, np.array(1))


def test_if():
    # test is true
    env = Env.Environment()
    result = E.If(E.N(1), E.Plus(E.N(1), E.N(2)), E.Minus(E.N(1), E.N(2))).evaluate(env)
    assert 3 == result.value

    # test is false
    result = E.If(E.N(0), E.Plus(E.N(1), E.N(2)), E.Minus(E.N(1), E.N(2))).evaluate(env)
    assert -1 == result.value


def test_accessor():
    env = Env.Environment()

    # test simple value
    simple = E.N(1)
    result = E.Accessor(simple, E.Accessor.IS_SIMPLE_VALUE).interpret(env)
    assert 1 == result.value

    # test array
    array = E.NewArray(E.NewArray(E.N(1), E.N(2)), E.NewArray(E.N(3), E.N(4)))
    result = E.Accessor(array, E.Accessor.IS_ARRAY).interpret(env)
    assert 1 == result.value
    result = E.Accessor(array, E.Accessor.NUM_DIMS).interpret(env)
    assert 2 == result.value
    result = E.Accessor(array, E.Accessor.NUM_ELEMENTS).interpret(env)
    assert 4 == result.value
    result = E.Accessor(array, E.Accessor.SHAPE).interpret(env)
    np.testing.assert_array_almost_equal(result.array, np.array([2, 2]))

    # test string
    string_test = E.Const(V.String("hi"))
    result = E.Accessor(string_test, E.Accessor.IS_STRING).interpret(env)
    assert 1 == result.value
    result = E.Accessor(array, E.Accessor.IS_STRING).interpret(env)
    assert 0 == result.value

    # test function
    function = E.LambdaExpression.wrap(E.Plus, 3)
    result = E.Accessor(function, E.Accessor.IS_FUNCTION).interpret(env)
    assert 1 == result.value
    result = E.Accessor(string_test, E.Accessor.IS_FUNCTION).interpret(env)
    assert 0 == result.value

    # test tuple
    tuple_test = E.TupleExpression(E.N(1), E.N(2))
    result = E.Accessor(tuple_test, E.Accessor.IS_TUPLE).interpret(env)
    assert 1 == result.value

    # test null
    null_test = E.Const(V.Null())
    result = E.Accessor(null_test, E.Accessor.IS_NULL).interpret(env)
    assert 1 == result.value

    # test default
    default_test = E.Const(V.DefaultParameter())
    result = E.Accessor(default_test, E.Accessor.IS_DEFAULT).interpret(env)
    assert 1 == result.value
    result = E.Accessor(null_test, E.Accessor.IS_DEFAULT).interpret(env)
    assert 0 == result.value

    # test if non-array variables have array attributes, should raise errors
    with pytest.raises(ProtocolError):
        E.Accessor(default_test, E.Accessor.NUM_DIMS).interpret(env)
    with pytest.raises(ProtocolError):
        E.Accessor(function, E.Accessor.SHAPE).interpret(env)
    with pytest.raises(ProtocolError):
        E.Accessor(string_test, E.Accessor.NUM_ELEMENTS).interpret(env)

