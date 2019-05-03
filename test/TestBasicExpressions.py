
import numpy as np
import unittest

import fc.language.expressions as E
import fc.language.values as V
import fc.environment as Env
from fc.error_handling import ProtocolError


class TestBasicExpressions(unittest.TestCase):
    """Test basic math expressions using simple, null, and default values."""

    def test_addition(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Plus(E.Const(V.Simple(1)), E.Const(V.Simple(2))).evaluate(env).value, 3)
        self.assertAlmostEqual(E.Plus(E.Const(V.Simple(1)), E.Const(
            V.Simple(2)), E.Const(V.Simple(4))).evaluate(env).value, 7)

    def test_with0d_array(self):
        env = Env.Environment()
        one = V.Array(np.array(1))
        self.assertEqual(one.value, 1)
        two = V.Array(np.array(2))
        four = V.Array(np.array(4))
        self.assertAlmostEqual(E.Plus(E.Const(one), E.Const(two)).evaluate(env).value, 3)
        self.assertAlmostEqual(E.Plus(E.Const(one), E.Const(two), E.Const(four)).evaluate(env).value, 7)
        self.assertAlmostEqual(E.Minus(E.Const(one), E.Const(two)).evaluate(env).value, -1)
        self.assertAlmostEqual(E.Power(E.Const(two), E.Const(four)).evaluate(env).value, 16)
        self.assertAlmostEqual(E.Times(E.Const(four), E.Const(two)).evaluate(env).value, 8)
        self.assertAlmostEqual(E.Root(E.Const(four)).evaluate(env).value, 2)

    def test_minus(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Minus(E.Const(V.Simple(1)), E.Const(V.Simple(2))).evaluate(env).value, -1)
        self.assertAlmostEqual(E.Minus(E.Const(V.Simple(4))).evaluate(env).value, -4)

    def test_times(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Times(E.Const(V.Simple(6)), E.Const(V.Simple(2))).evaluate(env).value, 12)
        self.assertAlmostEqual(E.Times(E.Const(V.Simple(6)), E.Const(
            V.Simple(2)), E.Const(V.Simple(3))).evaluate(env).value, 36)

    def test_divide(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Divide(E.Const(V.Simple(1)), E.Const(V.Simple(2))).evaluate(env).value, .5)

    def test_max(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Max(E.Const(V.Simple(6)), E.Const(V.Simple(12)),
                                     E.Const(V.Simple(2))).evaluate(env).value, 12)

    def test_min(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Min(E.Const(V.Simple(6)), E.Const(V.Simple(2)),
                                     E.Const(V.Simple(12))).evaluate(env).value, 2)

    def test_rem(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Rem(E.Const(V.Simple(6)), E.Const(V.Simple(4))).evaluate(env).value, 2)

    def test_power(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Power(E.Const(V.Simple(2)), E.Const(V.Simple(3))).evaluate(env).value, 8)

    def test_root(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Root(E.Const(V.Simple(16))).evaluate(env).value, 4)
        self.assertAlmostEqual(E.Root(E.Const(V.Simple(3)), E.Const(V.Simple(8))).evaluate(env).value, 2)

    def test_abs(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Abs(E.Const(V.Simple(-4))).evaluate(env).value, 4)
        self.assertAlmostEqual(E.Abs(E.Const(V.Simple(4))).evaluate(env).value, 4)

    def test_floor(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Floor(E.Const(V.Simple(1.8))).evaluate(env).value, 1)

    def test_ceiling(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Ceiling(E.Const(V.Simple(1.2))).evaluate(env).value, 2)

    def test_exp(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Exp(E.Const(V.Simple(3))).evaluate(env).value, 20.0855369231)

    def test_ln(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Ln(E.Const(V.Simple(3))).evaluate(env).value, 1.0986122886)

    def test_log(self):
        env = Env.Environment()
        self.assertAlmostEqual(E.Log(E.Const(V.Simple(3))).evaluate(env).value, 0.4771212547196)
        self.assertAlmostEqual(E.Log(E.Const(V.Simple(4)), E.Const(V.Simple(3))).evaluate(env).value, 0.79248125036)

    def test_name_look_up(self):
        env = Env.Environment()
        one = V.Simple(1)
        env.define_name("one", one)
        # Note that evaluate will try to optimise using numexpr, and so always returns a V.Array (0d in this case)
        self.assertEqual(E.NameLookUp("one").interpret(env).value, 1)
        np.testing.assert_array_equal(E.NameLookUp("one").evaluate(env).array, np.array(1))

    def test_if(self):
        # test is true
        env = Env.Environment()
        result = E.If(E.n(1), E.Plus(E.n(1), E.n(2)), E.Minus(E.n(1), E.n(2))).evaluate(env)
        self.assertEqual(3, result.value)

        # test is false
        result = E.If(E.n(0), E.Plus(E.n(1), E.n(2)), E.Minus(E.n(1), E.n(2))).evaluate(env)
        self.assertEqual(-1, result.value)

    def test_accessor(self):
        env = Env.Environment()

        # test simple value
        simple = E.n(1)
        result = E.Accessor(simple, E.Accessor.IS_SIMPLE_VALUE).interpret(env)
        self.assertEqual(1, result.value)

        # test array
        array = E.NewArray(E.NewArray(E.n(1), E.n(2)), E.NewArray(E.n(3), E.n(4)))
        result = E.Accessor(array, E.Accessor.IS_ARRAY).interpret(env)
        self.assertEqual(1, result.value)
        result = E.Accessor(array, E.Accessor.NUM_DIMS).interpret(env)
        self.assertEqual(2, result.value)
        result = E.Accessor(array, E.Accessor.NUM_ELEMENTS).interpret(env)
        self.assertEqual(4, result.value)
        result = E.Accessor(array, E.Accessor.SHAPE).interpret(env)
        np.testing.assert_array_almost_equal(result.array, np.array([2, 2]))

        # test string
        string_test = E.Const(V.String("hi"))
        result = E.Accessor(string_test, E.Accessor.IS_STRING).interpret(env)
        self.assertEqual(1, result.value)
        result = E.Accessor(array, E.Accessor.IS_STRING).interpret(env)
        self.assertEqual(0, result.value)

        # test function
        function = E.LambdaExpression.wrap(E.Plus, 3)
        result = E.Accessor(function, E.Accessor.IS_FUNCTION).interpret(env)
        self.assertEqual(1, result.value)
        result = E.Accessor(string_test, E.Accessor.IS_FUNCTION).interpret(env)
        self.assertEqual(0, result.value)

        # test tuple
        tuple_test = E.TupleExpression(E.n(1), E.n(2))
        result = E.Accessor(tuple_test, E.Accessor.IS_TUPLE).interpret(env)
        self.assertEqual(1, result.value)

        # test null
        null_test = E.Const(V.Null())
        result = E.Accessor(null_test, E.Accessor.IS_NULL).interpret(env)
        self.assertEqual(1, result.value)

        # test default
        default_test = E.Const(V.DefaultParameter())
        result = E.Accessor(default_test, E.Accessor.IS_DEFAULT).interpret(env)
        self.assertEqual(1, result.value)
        result = E.Accessor(null_test, E.Accessor.IS_DEFAULT).interpret(env)
        self.assertEqual(0, result.value)

        # test if non-array variables have array attributes, should raise errors
        self.assertRaises(ProtocolError, E.Accessor(default_test, E.Accessor.NUM_DIMS).interpret, env)
        self.assertRaises(ProtocolError, E.Accessor(function, E.Accessor.SHAPE).interpret, env)
        self.assertRaises(ProtocolError, E.Accessor(string_test, E.Accessor.NUM_ELEMENTS).interpret, env)
