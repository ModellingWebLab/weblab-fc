
import numpy as np
import unittest
import sys

import fc.language.expressions as E
import fc.language.statements as S
import fc.language.values as V
import fc.environment as Env
from fc.error_handling import ProtocolError

N = E.N


class TestArrayExpressions(unittest.TestCase):
    """Test array creation, view, map, fold, index, join, and stretch."""

    def testNewArray(self):
        arr = E.NewArray(N(1), N(2), N(3))  # simple one-dimensional array
        predictedArr = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(arr.Evaluate({}).array, predictedArr)
        arr = E.NewArray(E.NewArray(N(1), N(2), N(3)), E.NewArray(N(3), N(3), N(2)))
        predictedArr = np.array([[1, 2, 3], [3, 3, 2]])
        np.testing.assert_array_almost_equal(arr.Evaluate({}).array, predictedArr)

    def testViews(self):
        arr = E.NewArray(N(1), N(2), N(3), N(4))

        # two parameters: beginning and end, null represents end of original array
        view = E.View(arr, E.TupleExpression(N(1), E.Const(V.Null())))
        predictedArr = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        view = E.View(arr, E.TupleExpression(N(0), N(2), N(3)))  # three parameters: beginning, step, end
        predictedArr = np.array([1, 3])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        view = E.View(arr, E.TupleExpression(N(3), N(-1), N(0)))  # negative step
        predictedArr = np.array([4, 3, 2])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        view = E.View(arr, E.TupleExpression(N(1), N(0), N(1)))  # 0 as step
        predicted = 2
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predicted)
        view = E.View(arr, N(1))  # same as immediately above, but only one number passed instead of a step of 0
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predicted)

        array = E.NewArray(E.NewArray(N(-2), N(-1), N(0)), E.NewArray(N(1), N(2), N(3)),
                           E.NewArray(N(4), N(5), N(6)))  # testing many aspects of views of a 2-d array
        view = E.View(array, E.TupleExpression(N(0), E.Const(V.Null()), N(2)), E.TupleExpression(
            N(2), N(-1), N(0)))  # can slice stepping forward, backward, picking a position...etc
        predictedArr = np.array([[0, -1], [3, 2]])
        self.assertEqual(view.Evaluate({}).array.ndim, 2)
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        array = E.NewArray(E.NewArray(E.NewArray(N(-2), N(-1), N(0)),
                                      E.NewArray(N(1), N(2), N(3))),
                           E.NewArray(E.NewArray(N(4), N(5), N(6)),
                                      E.NewArray(N(1), N(2), N(3))))  # 3-d array
        view = E.View(array, E.TupleExpression(N(0), E.Const(V.Null()), E.Const(V.Null())),
                      E.TupleExpression(E.Const(V.Null()), N(-1), N(0)),
                      E.TupleExpression(N(0), N(2)))
        predictedArr = np.array([[[1, 2]], [[1, 2]]])
        self.assertEqual(view.Evaluate({}).array.ndim, 3)
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        # use four parameters in the tuples to specify dimension explicitly
        view = E.View(array, E.TupleExpression(N(0), N(0), E.Const(V.Null()), N(2)),
                      E.TupleExpression(N(1), N(2), N(-1), N(0)),
                      E.TupleExpression(N(2), N(0), E.Const(V.Null()), N(2)))
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        # use four parameters in the tuples to specify dimension with a mix of implicit and explicit declarations
        view = E.View(array, E.TupleExpression(N(0), E.Const(V.Null()), E.Const(V.Null())),
                      E.TupleExpression(N(1), E.Const(V.Null()), N(-1), N(0)),
                      E.TupleExpression(N(0), E.Const(V.Null()), N(2)))
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        # test leaving some parameters out so they fall to default
        view = E.View(array, E.TupleExpression(N(1), E.Const(V.Null()), N(-1), N(0)),
                      E.TupleExpression(N(0), E.Const(V.Null()), E.Const(V.Null())))
        view2 = E.View(array, E.TupleExpression(N(0), E.Const(V.Null()), E.Const(V.Null())),
                       E.TupleExpression(N(1), E.Const(V.Null()), N(-1), N(0)),
                       E.TupleExpression(E.Const(V.Null()), E.Const(V.Null()), N(1), E.Const(V.Null())))
        predictedArr = np.array([[[1, 2, 3]], [[1, 2, 3]]])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        # test leaving some parameters out so they get set to slice determined by dimension null
        view = E.View(array, E.TupleExpression(E.Const(V.Null()), N(0), N(1), N(2)))
        view2 = E.View(array, E.TupleExpression(N(0), N(0), N(1), N(2)),
                       E.TupleExpression(N(1), N(0), N(1), N(2)),
                       E.TupleExpression(N(2), N(0), N(1), N(2)))
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, view2.Evaluate({}).array)

        # checks to make sure the "default default" is equivalent to a tuple of (Null, Null, 1, Null),
        # also checks to make sure implicitly defined slices go to the first dimension that is not assigned explicitly
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, view2.Evaluate({}).array)
        # only specified dimension is in middle
        view = E.View(array, E.TupleExpression(N(1), E.Const(V.Null()), N(-1), N(0)))
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        view = E.View(array, E.TupleExpression(N(1), E.Const(V.Null()), N(-1), N(0)),
                      E.TupleExpression(N(0), N(0), E.Const(V.Null()), E.Const(V.Null())))
        # tests explicitly assigning dimension one before explicitly defining dimension zero
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        array = E.NewArray(E.NewArray(N(0), N(1), N(2)), E.NewArray(N(3), N(4), N(5)))
        view = E.View(array, E.TupleExpression(N(1)), E.TupleExpression(N(1), N(3)))
        predictedArr = np.array([4, 5])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

        array = E.NewArray(E.NewArray(N(0), N(1), N(2)), E.NewArray(N(3), N(4), N(5)))
        view = E.View(array, N(1), E.TupleExpression(N(1)))
        predictedArr = np.array([4])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)

    def testArrayCreationProtocolErrors(self):
        array = E.NewArray(E.NewArray(E.NewArray(N(-2), N(-1), N(0)),
                                      E.NewArray(N(1), N(2), N(3))),
                           E.NewArray(E.NewArray(N(4), N(5), N(6)),
                                      E.NewArray(N(1), N(2), N(3))))
        view = E.View(array, E.TupleExpression(N(0), N(0), E.Const(V.Null()), E.Const(V.Null())),
                      E.TupleExpression(N(1), E.Const(V.Null()), N(-1), N(0)),
                      E.TupleExpression(N(2), N(0), E.Const(V.Null()), E.Const(V.Null())),
                      E.TupleExpression(N(2), N(0), E.Const(V.Null()), E.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # more tuple slices than dimensions
        view = E.View(array, E.TupleExpression(N(1), E.Const(V.Null()), N(-1), N(0), N(1), N(4)))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # too many arguments for a slice
        view = E.View(N(1))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # first argument must be an array
        view = E.View(array, E.TupleExpression(N(5)))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # index goes out of range
        view = E.View(array, E.TupleExpression(N(3), N(0), E.Const(V.Null()), E.Const(V.Null())))
        # attempts to assign dimension three when array only has dimensions 0, 1, and 2
        self.assertRaises(ProtocolError, view.Evaluate, {})
        view = E.View(array, E.TupleExpression(N(1), N(1), N(0)))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # start is after end with positive step
        view = E.View(array, E.TupleExpression(N(0), N(-1), N(1)))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # start is before end with negative step
        view = E.View(array, E.TupleExpression(N(0), N(0), N(4)))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # start and end aren't equal and there's a step of 0
        view = E.View(array, E.TupleExpression(N(-4), N(1), N(2)))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # start is before beginning of array
        view = E.View(array, E.TupleExpression(N(-2), N(1), N(3)))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # end is after end of array
        view = E.View(array, E.TupleExpression(E.Const(V.Null()), N(1), N(-4)))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # end is before beginning of array
        view = E.View(array, E.TupleExpression(N(2), N(1), E.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {})  # beginning is after end of array

    def testArrayComprehension(self):
        env = Env.Environment()

        # 1-d array
        counting1d = E.NewArray(E.NameLookUp("i"), E.TupleExpression(
            N(0), N(0), N(1), N(10), E.Const(V.String("i"))), comprehension=True)
        predictedArr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_almost_equal(counting1d.Evaluate(env).array, predictedArr)

        # 2-d array, explicitly defined dimensions
        counting2d = E.NewArray(E.Plus(E.Times(E.NameLookUp("i"), N(3)), E.NameLookUp("j")),
                                E.TupleExpression(N(0), N(1), N(1), N(3), E.Const(V.String("i"))),
                                E.TupleExpression(N(1), N(0), N(1), N(3), E.Const(V.String("j"))),
                                comprehension=True)
        predictedArr = np.array([[3, 4, 5], [6, 7, 8]])
        np.testing.assert_array_almost_equal(counting2d.Evaluate(env).array, predictedArr)

        # 2-d array, order of variable definitions opposite of previous test
        counting2d = E.NewArray(E.Plus(E.Times(E.NameLookUp("i"), N(3)), E.NameLookUp("j")),
                                E.TupleExpression(N(1), N(0), N(1), N(3), E.Const(V.String("j"))),
                                E.TupleExpression(N(0), N(1), N(1), N(3), E.Const(V.String("i"))),
                                comprehension=True)
        predictedArr = np.array([[3, 4, 5], [6, 7, 8]])
        np.testing.assert_array_almost_equal(counting2d.Evaluate(env).array, predictedArr)

        # 2-d array with implicitly defined dimension assigned after explicit
        counting2d = E.NewArray(E.Plus(E.Times(E.NameLookUp("i"), N(3)), E.NameLookUp("j")),
                                E.TupleExpression(N(1), N(0), N(1), N(3), E.Const(V.String("j"))),
                                E.TupleExpression(N(1), N(1), N(3), E.Const(V.String("i"))),
                                comprehension=True)
        predictedArr = np.array([[3, 4, 5], [6, 7, 8]])
        np.testing.assert_array_almost_equal(counting2d.Evaluate(env).array, predictedArr)

        # 2-d array with implicitly defined dimension assigned before explicit
        counting2d = E.NewArray(E.Plus(E.Times(E.NameLookUp("i"), N(3)), E.NameLookUp("j")),
                                E.TupleExpression(N(0), N(1), N(3), E.Const(V.String("j"))),
                                E.TupleExpression(N(0), N(1), N(1), N(3), E.Const(V.String("i"))),
                                comprehension=True)
        predictedArr = np.array([[3, 4, 5], [6, 7, 8]])
        np.testing.assert_array_almost_equal(counting2d.Evaluate(env).array, predictedArr)

        # comprehension using arrays in generator expression with one variable
        blocks = E.NewArray(E.NewArray(E.NewArray(E.Plus(N(-10), E.NameLookUp("j")),
                                                  E.NameLookUp("j")),
                                       E.NewArray(E.Plus(N(10), E.NameLookUp("j")),
                                                  E.Plus(N(20), E.NameLookUp("j")))),
                            E.TupleExpression(N(1), N(0), N(1), N(2), E.Const(V.String("j"))),
                            comprehension=True)
        predictedArr = np.array([[[-10, 0], [-9, 1]], [[10, 20], [11, 21]]])
        np.testing.assert_array_almost_equal(blocks.Evaluate(Env.Environment()).array, predictedArr)

        # two gaps between instead of one
        blocks = E.NewArray(E.NewArray(E.NewArray(E.Plus(N(-10), E.NameLookUp("j")),
                                                  E.NameLookUp("j")),
                                       E.NewArray(E.Plus(N(10), E.NameLookUp("j")),
                                                  E.Plus(N(20), E.NameLookUp("j")))),
                            E.TupleExpression(N(2), N(0), N(1), N(2), E.Const(V.String("j"))),
                            comprehension=True)
        predictedArr = np.array([[[-10, -9], [0, 1]], [[10, 11], [20, 21]]])
        np.testing.assert_array_almost_equal(blocks.Evaluate(Env.Environment()).array, predictedArr)

        # comprehension using arrays in generator expression with two variables
        blocks = E.NewArray(E.NewArray(E.NewArray(E.Plus(N(-10), E.NameLookUp("j")),
                                                  E.NameLookUp("j")),
                                       E.NewArray(E.Plus(N(10), E.NameLookUp("i")),
                                                  E.Plus(N(20), E.NameLookUp("i")))),
                            E.TupleExpression(N(0), N(1), N(2), E.Const(V.String("j"))),
                            E.TupleExpression(N(0), N(1), N(2), E.Const(V.String("i"))),
                            comprehension=True)
        predictedArr = np.array([[[[-10, 0], [10, 20]], [[-10, 0], [11, 21]]],
                                 [[[-9, 1], [10, 20]], [[-9, 1], [11, 21]]]])
        np.testing.assert_array_almost_equal(blocks.Evaluate(Env.Environment()).array, predictedArr)

    def testArrayExpressionProtocolErrors(self):
        env = Env.Environment()
        # creates an empty array because the start is greater than the end
        fail = E.NewArray(E.NameLookUp("i"), E.TupleExpression(
            N(0), N(10), N(1), N(0), E.Const(V.String("i"))), comprehension=True)
        self.assertRaises(ProtocolError, fail.Evaluate, env)

        # creates an empty array because the step is negative when it should be positive
        fail = E.NewArray(E.NameLookUp("i"), E.TupleExpression(
            N(0), N(0), N(-1), N(10), E.Const(V.String("i"))), comprehension=True)
        self.assertRaises(ProtocolError, fail.Evaluate, env)

        blocks = E.NewArray(E.NewArray(E.NewArray(E.NameLookUp("j")),
                                       E.NewArray(E.NameLookUp("j"))),
                            E.TupleExpression(N(3), N(0), N(1), N(2), E.Const(V.String("j"))),
                            comprehension=True)
        self.assertRaises(ProtocolError, blocks.Evaluate, env)

        # map(lambda a, b=[0,1,2]: a + b, [1,2,3]) should give an error
        parameters = ['a', 'b']
        body = [S.Return(E.Plus(E.NameLookUp('a'), E.NameLookUp('b')))]
        array_default = E.LambdaExpression(parameters, body,
                                           defaultParameters=[V.DefaultParameter(), V.Array(np.array([0, 1, 2]))])
        a = E.NewArray(N(1), N(2), N(3))
        result = E.Map(array_default, a)
        self.assertRaises(ProtocolError, result.Evaluate, env)

    def testSimpleMap(self):
        env = Env.Environment()
        parameters = ['a', 'b', 'c']
        body = [S.Return(E.Plus(E.NameLookUp('a'), E.NameLookUp('b'), E.NameLookUp('c')))]
        add = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(2))
        b = E.NewArray(N(2), N(4))
        c = E.NewArray(N(3), N(7))
        result = E.Map(add, a, b, c)
        predicted = V.Array(np.array([6, 13]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

    def testMapWithMultiDimensionalArrays(self):
        env = Env.Environment()
        parameters = ['a', 'b', 'c']
        body = [S.Return(E.Plus(E.NameLookUp('a'), E.NameLookUp('b'), E.NameLookUp('c')))]
        add = E.LambdaExpression(parameters, body)
        a = E.NewArray(E.NewArray(N(1), N(2)), E.NewArray(N(2), N(3)))
        b = E.NewArray(E.NewArray(N(4), N(3)), E.NewArray(N(6), N(1)))
        c = E.NewArray(E.NewArray(N(2), N(2)), E.NewArray(N(8), N(0)))
        result = E.Map(add, a, b, c)
        predicted = V.Array(np.array([[7, 7], [16, 4]]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # more complex function and more complex array
        env = Env.Environment()
        parameters = ['a', 'b', 'c']
        body = [S.Return(E.Times(E.Plus(E.NameLookUp('a'), E.NameLookUp('b')), E.NameLookUp('c')))]
        add_times = E.LambdaExpression(parameters, body)
        a = E.NewArray(E.NewArray(E.NewArray(N(1), N(2)), E.NewArray(N(2), N(3))),
                       E.NewArray(E.NewArray(N(1), N(2)), E.NewArray(N(2), N(3))))
        b = E.NewArray(E.NewArray(E.NewArray(N(4), N(3)), E.NewArray(N(6), N(1))),
                       E.NewArray(E.NewArray(N(0), N(6)), E.NewArray(N(5), N(3))))
        c = E.NewArray(E.NewArray(E.NewArray(N(2), N(2)), E.NewArray(N(8), N(0))),
                       E.NewArray(E.NewArray(N(4), N(2)), E.NewArray(N(2), N(1))))
        result = E.Map(add_times, a, b, c)
        predicted = np.array([[[10, 10], [64, 0]], [[4, 16], [14, 6]]])
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)

        # test complicated expression involving views of a default that is an array
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Plus(E.Times(E.Power(E.NameLookUp('a'), N(2)), E.View(E.NameLookUp('b'), N(0))),
                                E.Times(E.NameLookUp('a'), E.View(E.NameLookUp('b'), N(1))),
                                E.View(E.NameLookUp('b'), N(2))))]
        default_array_test = E.LambdaExpression(parameters, body,
                                                defaultParameters=[V.DefaultParameter(), V.Array(np.array([1, 2, 3]))])
        a = E.NewArray(N(1), N(2), N(3))
        result = E.Map(default_array_test, a)
        predicted = np.array([6, 11, 18])
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)

    def testUsingManyOperationsinFunction(self):
        env = Env.Environment()
        parameters = ['a', 'b', 'c']
        body = [S.Return(E.Times(E.Plus(E.NameLookUp('a'), E.Times(
            E.NameLookUp('b'), E.NameLookUp('c'))), E.NameLookUp('a')))]
        add_times = E.LambdaExpression(parameters, body)
        a = E.NewArray(E.NewArray(N(1), N(2)), E.NewArray(N(2), N(3)))
        b = E.NewArray(E.NewArray(N(4), N(3)), E.NewArray(N(6), N(1)))
        c = E.NewArray(E.NewArray(N(2), N(2)), E.NewArray(N(8), N(0)))
        result = E.Map(add_times, a, b, c)
        predicted = np.array([[9, 16], [100, 9]])
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)

    def testMapWithFunctionWithDefaults(self):
        env = Env.Environment()
        body = [S.Return(E.Plus(E.NameLookUp('item'), E.NameLookUp('incr')))]
        add = E.LambdaExpression(['item', 'incr'], body, defaultParameters=[V.DefaultParameter(), V.Simple(3)])
        item = E.NewArray(N(1), N(3), N(5))
        result = E.Map(add, item)
        predicted = np.array([4, 6, 8])
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)

    def testNestedFunction(self):
        env = Env.Environment()
        nested_body = [S.Return(E.Plus(E.NameLookUp('input'), E.NameLookUp('outer_var')))]
        nested_function = E.LambdaExpression(["input"], nested_body)
        body = [S.Assign(["nested_fn"], nested_function),
                S.Assign(["outer_var"], N(1)),
                S.Return(E.Eq(E.FunctionCall("nested_fn", [N(1)]), N(2)))]
        nested_scope = E.LambdaExpression([], body)
        nested_call = E.FunctionCall(nested_scope, [])
        result = nested_call.Evaluate(env)
        predicted = np.array([True])
        self.assertEqual(result.value, predicted)

    def testCompileMethodsForMathExpression(self):
        # Minus
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Minus(E.NameLookUp('a'), E.NameLookUp('b')))]
        minus = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(4), N(2))
        b = E.NewArray(N(2), N(1))
        result = E.Map(minus, a, b)
        predicted = V.Array(np.array([2, 1]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Divide
        body = [S.Return(E.Divide(E.NameLookUp('a'), E.NameLookUp('b')))]
        divide = E.LambdaExpression(parameters, body)
        result = E.Map(divide, a, b)
        predicted = V.Array(np.array([2, 2]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Remainder
        body = [S.Return(E.Rem(E.NameLookUp('a'), E.NameLookUp('b')))]
        rem = E.LambdaExpression(parameters, body)
        result = E.Map(rem, a, b)
        predicted = V.Array(np.array([0, 0]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Power
        body = [S.Return(E.Power(E.NameLookUp('a'), E.NameLookUp('b')))]
        power = E.LambdaExpression(parameters, body)
        result = E.Map(power, a, b)
        predicted = V.Array(np.array([16, 2]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Root with one argument
        body = [S.Return(E.Root(E.NameLookUp('a')))]
        root = E.LambdaExpression('a', body)
        result = E.Map(root, a)
        predicted = V.Array(np.array([2, 1.41421356]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Root with two arguments
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Root(E.NameLookUp('a'), E.NameLookUp('b')))]
        root = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(3))
        b = E.NewArray(N(8))
        result = E.Map(root, b, a)
        predicted = V.Array(np.array([2]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Absolute value
        env = Env.Environment()
        parameters = ['a']
        body = [S.Return(E.Abs(E.NameLookUp('a')))]
        absolute = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(4), N(-2))
        result = E.Map(absolute, a)
        predicted = V.Array(np.array([4, 2]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Exponential
        env = Env.Environment()
        parameters = ['a']
        body = [S.Return(E.Exp(E.NameLookUp('a')))]
        exponential = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(3))
        result = E.Map(exponential, a)
        predicted = V.Array(np.array([20.0855369231]))
        # np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Natural Log
        env = Env.Environment()
        body = [S.Return(E.Ln(E.NameLookUp('a')))]
        ln = E.LambdaExpression(parameters, body)
        result = E.Map(ln, a)
        predicted = V.Array(np.array([1.0986122886]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Log with One Argument (Log base 10)
        env = Env.Environment()
        body = [S.Return(E.Log(E.NameLookUp('a')))]
        log = E.LambdaExpression(parameters, body)
        result = E.Map(log, a)
        predicted = V.Array(np.array([0.4771212547196]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Log with two arguments, second is log base qualifier
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Log(E.NameLookUp('a'), E.NameLookUp('b')))]
        log = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(4))
        b = E.NewArray(N(3))
        result = E.Map(log, a, b)
        predicted = V.Array(np.array([0.79248125036]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Max
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Max(E.NameLookUp('a'), E.NameLookUp('b')))]
        max = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(4), N(5), N(1))
        b = E.NewArray(N(3), N(6), N(0))
        result = E.Map(max, a, b)
        predicted = V.Array(np.array([4, 6, 1]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Floor
        env = Env.Environment()
        parameters = ['a']
        body = [S.Return(E.Floor(E.NameLookUp('a')))]
        floor = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(3.2), N(3.7))
        result = E.Map(floor, a)
        predicted = V.Array(np.array([3, 3]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Ceiling
        env = Env.Environment()
        parameters = ['a']
        body = [S.Return(E.Ceiling(E.NameLookUp('a')))]
        ceiling = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(3.2), N(3.7))
        result = E.Map(ceiling, a)
        predicted = V.Array(np.array([4, 4]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

    def testCompileMethodsForLogicalExpressions(self):
        # And
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.And(E.NameLookUp('a'), E.NameLookUp('b')))]
        test_and = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(1))
        b = E.NewArray(N(0), N(1))
        result = E.Map(test_and, a, b)
        predicted = V.Array(np.array([False, True]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Or
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Or(E.NameLookUp('a'), E.NameLookUp('b')))]
        test_or = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(1), N(0))
        b = E.NewArray(N(0), N(1), N(0))
        result = E.Map(test_or, a, b)
        predicted = V.Array(np.array([True, True, False]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Xor
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Xor(E.NameLookUp('a'), E.NameLookUp('b')))]
        test_xor = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(1), N(0))
        b = E.NewArray(N(0), N(1), N(0))
        result = E.Map(test_xor, a, b)
        predicted = V.Array(np.array([True, False, False]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # Not
        env = Env.Environment()
        parameters = ['a']
        body = [S.Return(E.Not(E.NameLookUp('a')))]
        test_not = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(0))
        result = E.Map(test_not, a)
        predicted = V.Array(np.array([False, True]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # greater than
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Gt(E.NameLookUp('a'), E.NameLookUp('b')))]
        test_greater = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(0), N(0))
        b = E.NewArray(N(0), N(0), N(1))
        result = E.Map(test_greater, a, b)
        predicted = V.Array(np.array([True, False, False]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # less than
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Lt(E.NameLookUp('a'), E.NameLookUp('b')))]
        test_less = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(0), N(0))
        b = E.NewArray(N(0), N(0), N(1))
        result = E.Map(test_less, a, b)
        predicted = V.Array(np.array([False, False, True]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # greater than equal to
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Geq(E.NameLookUp('a'), E.NameLookUp('b')))]
        test_greater_eq = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(0), N(0))
        b = E.NewArray(N(0), N(0), N(1))
        result = E.Map(test_greater_eq, a, b)
        predicted = V.Array(np.array([True, True, False]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # less than equal to
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Leq(E.NameLookUp('a'), E.NameLookUp('b')))]
        test_less_eq = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(0), N(0))
        b = E.NewArray(N(0), N(0), N(1))
        result = E.Map(test_less_eq, a, b)
        predicted = V.Array(np.array([False, True, True]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

        # not equal
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Neq(E.NameLookUp('a'), E.NameLookUp('b')))]
        test_not_eq = E.LambdaExpression(parameters, body)
        a = E.NewArray(N(1), N(0), N(0))
        b = E.NewArray(N(0), N(0), N(1))
        result = E.Map(test_not_eq, a, b)
        predicted = V.Array(np.array([True, False, True]))
        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)

    def testFold(self):
        # 1-d array, add fold
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Plus(E.NameLookUp('a'), E.NameLookUp('b')))]
        add = E.LambdaExpression(parameters, body)
        array = E.NewArray(N(0), N(1), N(2))
        result = E.Fold(add, array, N(0), N(0)).Interpret(env)
        predicted = np.array([3])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d array, add fold over dimension 0
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Plus(E.NameLookUp('a'), E.NameLookUp('b')))]
        add = E.LambdaExpression(parameters, body)
        array = E.NewArray(E.NewArray(N(0), N(1), N(2)), E.NewArray(N(3), N(4), N(5)))
        result = E.Fold(add, array, N(0), N(0)).Interpret(env)
        predicted = np.array([[3, 5, 7]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d array, add fold over dimension 1 and non-zero initial value
        result = E.Fold(add, array, N(5), N(1)).Interpret(env)
        predicted = np.array([[8], [17]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d array, add fold over dimension 1 using implicitly defined dimension
        result = E.Fold(add, array, N(0)).Interpret(env)
        predicted = np.array([[3], [12]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d array, add fold over dimension 1 using implicitly defined dimension and initial
        # can you implicitly define initial and explicitly define dimension?
        array = E.NewArray(E.NewArray(N(1), N(2), N(3)), E.NewArray(N(3), N(4), N(5)))
        result = E.Fold(add, array).Interpret(env)
        predicted = np.array([[6], [12]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 3-d array, times fold over dimension 0
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Times(E.NameLookUp('a'), E.NameLookUp('b')))]
        times = E.LambdaExpression(parameters, body)
        array = E.NewArray(E.NewArray(E.NewArray(N(1), N(2), N(3)), E.NewArray(N(4), N(2), N(1))),
                           E.NewArray(E.NewArray(N(3), N(0), N(5)), E.NewArray(N(2), N(2), N(1))))
        result = E.Fold(times, array, N(1), N(0)).Interpret(env)
        predicted = np.array([[[3, 0, 15], [8, 4, 1]]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 3-d array, times fold over dimension 1
        result = E.Fold(times, array, N(1), N(1)).Interpret(env)
        predicted = np.array([[[4, 4, 3]], [[6, 0, 5]]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 3-d array, times fold over dimension 2
        result = E.Fold(times, array, N(1), N(2)).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 3-d array, times fold over dimension 2 (defined implicitly)
        result = E.Fold(times, array, N(1)).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 3-d array, times fold over dimension 2 (defined explicitly as default)
        result = E.Fold(times, array, N(1), E.Const(V.DefaultParameter())).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 3-d array, times fold over dimension 2 (defined implicitly) with no initial value input
        result = E.Fold(times, array).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 3-d array, times fold over dimension 2 using default parameter for both initial value and dimension
        result = E.Fold(times, array, E.Const(V.DefaultParameter()), E.Const(V.DefaultParameter())).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted)

    def testFoldWithDifferentFunctions(self):
        # fold with max function
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Max(E.NameLookUp('a'), E.NameLookUp('b')))]
        max_function = E.LambdaExpression(parameters, body)
        array = E.NewArray(E.NewArray(N(0), N(1), N(8)), E.NewArray(N(3), N(4), N(5)))
        result = E.Fold(max_function, array, N(0), N(0)).Interpret(env)
        predicted = np.array([[3, 4, 8]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # fold with max function with an initial value that affects output
        array = E.NewArray(E.NewArray(N(0), N(1), N(8)), E.NewArray(N(3), N(4), N(5)))
        result = E.Fold(max_function, array, N(7), N(0)).Interpret(env)
        predicted = np.array([[7, 7, 8]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # fold with min function
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Min(E.NameLookUp('a'), E.NameLookUp('b')))]
        min_function = E.LambdaExpression(parameters, body)
        array = E.NewArray(E.NewArray(N(0), N(1), N(8)), E.NewArray(N(3), N(-4), N(5)))
        result = E.Fold(min_function, array).Interpret(env)
        predicted = np.array([[0], [-4]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # fold with minus function
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Minus(E.NameLookUp('a'), E.NameLookUp('b')))]
        minus_function = E.LambdaExpression(parameters, body)
        array = E.NewArray(E.NewArray(N(0), N(1), N(4)), E.NewArray(N(3), N(1), N(5)))
        result = E.Fold(minus_function, array, N(5), N(1)).Interpret(env)
        predicted = np.array([[0], [-4]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # fold with divide function
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(E.Divide(E.NameLookUp('a'), E.NameLookUp('b')))]
        divide_function = E.LambdaExpression(parameters, body)
        array = E.NewArray(E.NewArray(N(2), N(2), N(4)), E.NewArray(N(16), N(2), N(1)))
        result = E.Fold(divide_function, array, N(32), N(1)).Interpret(env)
        predicted = np.array([[2], [1]])
        np.testing.assert_array_almost_equal(result.array, predicted)

    def testFind(self):
        # Find with 2-d array as input
        env = Env.Environment()
        array = E.NewArray(E.NewArray(N(1), N(0), N(2), N(3)), E.NewArray(N(0), N(0), N(3), N(9)))
        result = E.Find(array).Evaluate(env)
        predicted = np.array(np.array([[0, 0], [0, 2], [0, 3], [1, 2], [1, 3]]))
        np.testing.assert_array_almost_equal(result.array, predicted)

        # Should raise error if non-array passed in
        array = E.Const(V.Simple(1))
        result = E.Find(array)
        self.assertRaises(ProtocolError, result.Evaluate, env)

    def testIndex(self):
        # 2-d pad first dimension to the left
        env = Env.Environment()
        array = E.NewArray(E.NewArray(N(1), N(0), N(2)), E.NewArray(N(0), N(3), N(0)), E.NewArray(N(1), N(1), N(1)))
        # [ [1, 0, 2]
        #   [0, 3, 0]
        #   [1, 1, 1] ]
        find = E.Find(array)
        result = E.Index(array, find, N(1), N(0), N(1), N(45)).Interpret(env)
        predicted = np.array(np.array([[1, 2, 45], [3, 45, 45], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d pad dimension 0 up
        result = E.Index(array, E.Find(array), N(0), N(0), N(1), N(45)).Interpret(env)
        predicted = np.array(np.array([[1, 3, 2], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d pad dimension 0 to the left with defaults for shrink, pad, pad_value
        result = E.Index(array, find, N(0)).Interpret(env)
        predicted = np.array(np.array([[1, 3, 2], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d pad first dimension to the right
        result = E.Index(array, find, N(1), N(0), N(-1), N(45)).Interpret(env)
        predicted = np.array(np.array([[45, 1, 2], [45, 45, 3], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d pad first dimension to the right with default max value for pad_value
        result = E.Index(array, find, N(1), N(0), N(-1)).Interpret(env)
        predicted = np.array(np.array([[sys.float_info.max, 1, 2], [
                             sys.float_info.max, sys.float_info.max, 3], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d shrink first dimension to the left with defaults for pad and pad_value
        result = E.Index(array, find, N(1), N(1)).Interpret(env)
        predicted = np.array(np.array([[1], [3], [1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d shrink first dimension to the right with defaults for pad and pad_value
        result = E.Index(array, find, N(1), N(-1)).Interpret(env)
        predicted = np.array(np.array([[2], [3], [1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 1-d
        env = Env.Environment()
        array = E.NewArray(N(1), N(0), N(2), N(0))
        # [1, 0, 2, 0]
        find = E.Find(array)
        result = E.Index(array, find).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)

        # a few more tests for 1-d array, should all yield the same result
        result = E.Index(array, find, N(0)).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)
        result = E.Index(array, find, N(0), N(1)).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)
        result = E.Index(array, find, N(0), N(0), N(0), N(0)).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)
        result = E.Index(array, find, N(0), N(0), N(-1), N(100)).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)

    def testIndexProtocolErrors(self):
        # index over dimension 2 in 2 dimensional array (out of range)
        env = Env.Environment()
        array = E.NewArray(E.NewArray(N(1), N(0), N(2)), E.NewArray(N(0), N(3), N(0)), E.NewArray(N(1), N(1), N(1)))
        find = E.Find(array)
        result = E.Index(array, find, N(2), N(0), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)

        # index by shrinking and padding at the same time
        result = E.Index(array, find, N(1), N(1), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)

        # shrink and pad are both 0, but output is irregular
        result = E.Index(array, find, N(1), N(0), N(0), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)

        # input an array of indices that is not 2-d
        result = E.Index(array, E.NewArray(N(1), N(2)), N(1), N(1), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)

        # input array for dimension value instead of simple value
        result = E.Index(array, E.NewArray(N(1), N(2)), array, N(1), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)

        # input simple value for array instead of array
        result = E.Index(N(1), E.NewArray(N(1), N(2)), array, N(1), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)

    def testJoinAndStretch(self):
        env = Env.Environment()
        env.DefineName('repeated_arr', V.Array(np.array([1, 2, 3])))
        stretch = E.NewArray(E.NameLookUp("repeated_arr"),
                             E.TupleExpression(N(1), N(0), N(1), N(3), E.Const(V.String("j"))),
                             comprehension=True)
        predictedArr = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_almost_equal(stretch.Evaluate(env).array, predictedArr)

        stretch = E.NewArray(E.NameLookUp("repeated_arr"),
                             E.TupleExpression(N(0), N(0), N(1), N(3), E.Const(V.String("j"))),
                             comprehension=True)
        predictedArr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_almost_equal(stretch.Evaluate(env).array, predictedArr)
