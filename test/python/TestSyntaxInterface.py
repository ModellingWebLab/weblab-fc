
import numpy as np
try:
    import unittest2 as unittest
except ImportError:
    import unittest

import fc
import fc.language.expressions as E
import fc.language.statements as S
import fc.language.values as V
import fc.simulations.model as Model
import fc.simulations.modifiers as Modifiers
import fc.simulations.ranges as Ranges
import fc.simulations.simulations as Simulations
import fc.utility.environment as Env

import CompactSyntaxParser as CSP

csp = CSP.CompactSyntaxParser
N = E.N


class TestSyntaxInterface(unittest.TestCase):
    """Test expr methods of compact syntax parser."""

    def TestParsingNumber(self):
        parse_action = csp.expr.parseString('1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Const)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1.0)

    def TestParsingVariable(self):
        parse_action = csp.expr.parseString('a', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NameLookUp)
        env = Env.Environment()
        env.DefineName('a', V.Simple(3.0))
        self.assertEqual(expr.Evaluate(env).value, 3.0)

    def TestParsingMathOperations(self):
        # plus
        parse_action = csp.expr.parseString('1.0 + 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Plus)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 3.0)

        # minus
        parse_action = csp.expr.parseString('5.0 - 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Minus)
        self.assertEqual(expr.Evaluate(env).value, 3.0)

        # times
        parse_action = csp.expr.parseString('4.0 * 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Times)
        self.assertEqual(expr.Evaluate(env).value, 8.0)

        # division and infinity
        parse_action = csp.expr.parseString('6.0 / 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Divide)
        self.assertEqual(expr.Evaluate(env).value, 3.0)

        parse_action = csp.expr.parseString('1/MathML:infinity', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Divide)
        self.assertEqual(expr.Evaluate(env).value, 0)

        # power
        parse_action = csp.expr.parseString('4.0 ^ 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Power)
        self.assertEqual(expr.Evaluate(env).value, 16.0)

    def TestParsingLogicalOperations(self):
        # greater than
        parse_action = csp.expr.parseString('4.0 > 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Gt)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1)

        parse_action = csp.expr.parseString('2.0 > 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Gt)
        self.assertEqual(expr.Evaluate(env).value, 0)

        # less than
        parse_action = csp.expr.parseString('4.0 < 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Lt)
        self.assertEqual(expr.Evaluate(env).value, 0)

        parse_action = csp.expr.parseString('2.0 < 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Lt)
        self.assertEqual(expr.Evaluate(env).value, 0)

        # less than or equal to
        parse_action = csp.expr.parseString('4.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Leq)
        self.assertEqual(expr.Evaluate(env).value, 0)

        parse_action = csp.expr.parseString('2.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Leq)
        self.assertEqual(expr.Evaluate(env).value, 1)

        parse_action = csp.expr.parseString('1.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Leq)
        self.assertEqual(expr.Evaluate(env).value, 1)

        # equal to
        parse_action = csp.expr.parseString('2.0 == 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Eq)
        self.assertEqual(expr.Evaluate(env).value, 1)

        # not equal to and not a number
        parse_action = csp.expr.parseString('2.0 != 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Neq)
        self.assertEqual(expr.Evaluate(env).value, 0)

        parse_action = csp.expr.parseString('1.0 != 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Neq)
        self.assertEqual(expr.Evaluate(env).value, 1)

        parse_action = csp.expr.parseString('MathML:notanumber != MathML:notanumber', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Neq)
        self.assertEqual(expr.Evaluate(env).value, 1)

        # and
        parse_action = csp.expr.parseString('1.0 && 1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.And)
        self.assertEqual(expr.Evaluate(env).value, 1)

        parse_action = csp.expr.parseString('0.0 && 1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.And)
        self.assertEqual(expr.Evaluate(env).value, 0)

        # or and true or false
        parse_action = csp.expr.parseString('MathML:true || MathML:true', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Or)
        self.assertEqual(expr.Evaluate(env).value, 1)

        parse_action = csp.expr.parseString('MathML:false || MathML:true', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Or)
        self.assertEqual(expr.Evaluate(env).value, 1)

        parse_action = csp.expr.parseString('MathML:false || MathML:false', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Or)
        self.assertEqual(expr.Evaluate(env).value, 0)

        # not
        parse_action = csp.expr.parseString('not 1', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Not)
        self.assertEqual(expr.Evaluate(env).value, 0)

        parse_action = csp.expr.parseString('not 0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Not)
        self.assertEqual(expr.Evaluate(env).value, 1)

    def TestParsingComplicatedMath(self):
        parse_action = csp.expr.parseString('1.0 + (4.0 * 2.0)', parseAll=True)
        expr = parse_action[0].expr()
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 9.0)

        parse_action = csp.expr.parseString('(2.0 ^ 3.0) + (5.0 * 2.0) - 10.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr.Evaluate(env).value, 8.0)

        parse_action = csp.expr.parseString('2.0 ^ 3.0 == 5.0 + 3.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr.Evaluate(env).value, 1)

    def TestParsingMathMLFuncs(self):
        # ceiling
        parse_action = csp.expr.parseString('MathML:ceiling(1.2)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Ceiling)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 2.0)

        # floor
        parse_action = csp.expr.parseString('MathML:floor(1.8)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Floor)
        self.assertEqual(expr.Evaluate(env).value, 1.0)

        # ln and exponentiale value
        parse_action = csp.expr.parseString('MathML:ln(MathML:exponentiale)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Ln)
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)

        # log
        parse_action = csp.expr.parseString('MathML:log(10)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Log)
        self.assertEqual(expr.Evaluate(env).value, 1)

        # exp and infinity value
        parse_action = csp.expr.parseString('MathML:exp(MathML:ln(10))', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Exp)
        self.assertAlmostEqual(expr.Evaluate(env).value, 10)

        # abs
        parse_action = csp.expr.parseString('MathML:abs(-10)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Abs)
        self.assertAlmostEqual(expr.Evaluate(env).value, 10)

        # root
        parse_action = csp.expr.parseString('MathML:root(100)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Root)
        self.assertAlmostEqual(expr.Evaluate(env).value, 10)

        # rem
        parse_action = csp.expr.parseString('MathML:rem(100, 97)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Rem)
        self.assertAlmostEqual(expr.Evaluate(env).value, 3.0)

        # max
        parse_action = csp.expr.parseString('MathML:max(100, 97, 105)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Max)
        self.assertAlmostEqual(expr.Evaluate(env).value, 105)

        # min and pi value
        parse_action = csp.expr.parseString('MathML:min(100, 97, 105, MathML:pi)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Min)
        self.assertAlmostEqual(expr.Evaluate(env).value, 3.1415926535)

        # null
        parse_action = csp.expr.parseString('null', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Const)
        self.assertIsInstance(expr.value, V.Null)

    def TestParsingIf(self):
        parse_action = csp.expr.parseString('if 1 then 2 + 3 else 4-2', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.If)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).value, 5)

        parse_action = csp.expr.parseString('if MathML:false then 2 + 3 else 4-2', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.If)
        self.assertAlmostEqual(expr.Evaluate(env).value, 2)

    def TestParsingTupleExpression(self):
        parse_action = csp.expr.parseString('(1, 2)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.TupleExpression)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).values[0].value, 1)
        self.assertAlmostEqual(expr.Evaluate(env).values[1].value, 2)

    def TestParsingArray(self):
        parse_action = csp.expr.parseString('[1, 2, 3]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NewArray)
        env = Env.Environment()
        np.testing.assert_array_almost_equal(expr.Evaluate(env).array, np.array([1, 2, 3]))

    def TestParsingAccessor(self):
        # is simple true
        parse_action = csp.expr.parseString('1.IS_SIMPLE_VALUE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)

        # is array true
        parse_action = csp.expr.parseString('[1, 2].IS_ARRAY', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)

        # is tuple true
        parse_action = csp.expr.parseString('(1, 2).IS_TUPLE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)

        # is tuple false
        parse_action = csp.expr.parseString('[1, 2].IS_TUPLE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.Evaluate(env).value, 0)

        # multiple accessors- .shape.is_array
        parse_action = csp.expr.parseString('[1, 2, 3].SHAPE.IS_ARRAY', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)

        # shape
        parse_action = csp.expr.parseString('[1, 2, 3].SHAPE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        np.testing.assert_array_equal(expr.Evaluate(env).array, np.array([3]))

        # number of dimensions
        parse_action = csp.expr.parseString('[1, 2, 3].NUM_DIMS', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        np.testing.assert_array_equal(expr.Evaluate(env).array, np.array([1]))

        # number of elements
        parse_action = csp.expr.parseString('[1, 2, 3].NUM_ELEMENTS', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        np.testing.assert_array_equal(expr.Evaluate(env).array, np.array([3]))

    def TestStatements(self):
        # test assertion
        parse_action = csp.assertStmt.parseString("assert 1", parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Assert)
        env = Env.Environment()
        expr.Evaluate(env)  # checked simply by not raising protocol error

        # test assign

        # assign one variable to an expression
        env = Env.Environment()
        parse_action = csp.assignStmt.parseString('a = 1.0 + 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Assign)
        expr.Evaluate(env)
        self.assertEqual(env.LookUp('a').value, 3)

        # assign two variables at once to numbers
        parse_action = csp.assignStmt.parseString('b, c = 1.0, 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Assign)
        expr.Evaluate(env)
        self.assertEqual(env.LookUp('b').value, 1)
        self.assertEqual(env.LookUp('c').value, 2)

        # assign three variables at once to expressions
        parse_action = csp.assignStmt.parseString('d, e, f = 1.0, 2 + 2.0, (3*4)-2', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Assign)
        expr.Evaluate(env)
        self.assertEqual(env.LookUp('d').value, 1)
        self.assertEqual(env.LookUp('e').value, 4)
        self.assertEqual(env.LookUp('f').value, 10)

        # test return

        # return one number
        parse_action = csp.returnStmt.parseString('return 1', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Return)
        results = expr.Evaluate(env)
        self.assertEqual(results.value, 1)

        # return one expression involving variables
        parse_action = csp.returnStmt.parseString('return d + e', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Return)
        results = expr.Evaluate(env)
        self.assertEqual(results.value, 5)

        # return two numbers
        parse_action = csp.returnStmt.parseString('return 1, 3', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Return)
        result1, result2 = expr.Evaluate(env).values
        self.assertEqual(result1.value, 1)
        self.assertEqual(result2.value, 3)

        # test statement list
        parse_action = csp.stmtList.parseString('z = lambda a: a+2\nassert z(2) == 4', parseAll=True)
        result = parse_action[0].expr()
        env.ExecuteStatements(result)

    def TestParsingLambda(self):
        # no default, one variable
        env = Env.Environment()
        parse_action = csp.lambdaExpr.parseString('lambda a: a + 1', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.LambdaExpression)
        result = E.FunctionCall(expr, [N(3)]).Evaluate(env)
        self.assertEqual(result.value, 4)

        # no default, two variables
        env = Env.Environment()
        parse_action = csp.lambdaExpr.parseString('lambda a, b: a * b', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.LambdaExpression)
        result = E.FunctionCall(expr, [N(4), N(2)]).Evaluate(env)
        self.assertEqual(result.value, 8)

        # test lambda with defaults unused
        env = Env.Environment()
        parse_action = csp.lambdaExpr.parseString('lambda a=2, b=3: a + b', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.LambdaExpression)
        result = E.FunctionCall(expr, [N(2), N(6)]).Evaluate(env)
        self.assertEqual(result.value, 8)

        # test lambda with defaults used
        env = Env.Environment()
        parse_action = csp.lambdaExpr.parseString('lambda a=2, b=3: a + b', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.LambdaExpression)
        result = E.FunctionCall(expr, [E.Const(V.DefaultParameter())]).Evaluate(env)
        self.assertEqual(result.value, 5)

    def TestArrayComprehensions(self):
        env = Env.Environment()
        parse_action = csp.array.parseString('[i for i in 0:10]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NewArray)
        result = expr.Evaluate(env)
        predicted = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_almost_equal(predicted, result.array)

        parse_action = csp.array.parseString('[i*2 for i in 0:2:4]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NewArray)
        result = expr.Evaluate(env)
        predicted = np.array([0, 4])
        np.testing.assert_array_almost_equal(predicted, result.array)

        parse_action = csp.array.parseString('[i+j*5 for i in 1:3 for j in 2:4]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NewArray)
        result = expr.Evaluate(env)
        predicted = np.array([[11, 16], [12, 17]])
        np.testing.assert_array_almost_equal(predicted, result.array)

        env = Env.Environment()
        arr = V.Array(np.arange(10))
        env.DefineName('arr', arr)
        parse_action = csp.expr.parseString('arr[1:2:10]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.Evaluate(env)
        predicted = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_almost_equal(predicted, result.array)

    def TestParsingViews(self):
        env = Env.Environment()
        view_arr = V.Array(np.arange(10))
        env.DefineName('view_arr', view_arr)
        view_parse_action = csp.expr.parseString('view_arr[4]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.Evaluate(env)
        predicted = np.array(4)
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = csp.expr.parseString('view_arr[2:5]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.Evaluate(env)
        predicted = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = csp.expr.parseString('view_arr[1:2:10]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.Evaluate(env)
        predicted = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_almost_equal(predicted, result.array)

        env = Env.Environment()
        view_arr = V.Array(np.array([[0, 1, 2, 3, 4], [7, 8, 12, 3, 9]]))
        env.DefineName('view_arr', view_arr)
        view_parse_action = csp.expr.parseString('view_arr[1$2]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.Evaluate(env)
        predicted = np.array([2, 12])
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = csp.expr.parseString('view_arr[1$(3):]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.Evaluate(env)
        predicted = np.array([[3, 4], [3, 9]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = csp.expr.parseString('view_arr[*$1]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.Evaluate(env)
        predicted = np.array(8)
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = csp.expr.parseString('view_arr[*$1][0]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.Evaluate(env)
        predicted = np.array(1)
        np.testing.assert_array_almost_equal(result.array, predicted)

    def ParsingFindAndIndexArray(self):
        env = Env.Environment()
        arr = V.Array(np.arange(4))
        env.DefineName('arr', arr)
        find_parse_action = csp.expr.parseString('find(arr)', parseAll=True)
        expr = find_parse_action[0].expr()
        self.assertIsInstance(expr, E.Find)
        find_result = expr.Evaluate(env)
        predicted = np.array([[1], [2], [3]])
        np.testing.assert_array_almost_equal(find_result.array, predicted)

        find_arr = V.Array(np.array([[1, 0, 3, 0, 0], [0, 7, 0, 0, 10], [0, 0, 13, 14, 0],
                                     [0, 0, 0, 19, 20], [0, 0, 0, 0, 25]]))
        index_arr = V.Array(np.arange(1, 26).reshape(5, 5))
        env.DefineName('find_arr', find_arr)
        env.DefineName('index_arr', index_arr)
        find_parse_action = csp.expr.parseString('find(find_arr)', parseAll=True)
        expr = find_parse_action[0].expr()
        indices_from_find = expr.Evaluate(env)
        env.DefineName('indices_from_find', indices_from_find)
        index_parse_action = csp.expr.parseString('index_arr{indices_from_find, 1, pad:1=0}')
        expr = index_parse_action[0].expr()
        index_result = expr.Interpret(env)
        predicted = np.array([[1, 3], [7, 10], [13, 14], [19, 20], [25, 0]])
        np.testing.assert_array_almost_equal(index_result.array, predicted)

        env.DefineName('find_result', find_result)
        index_parse_action = csp.expr.parseString('arr{find_result}', parseAll=True)
        expr = index_parse_action[0].expr()
        self.assertIsInstance(expr, E.Index)
        result = expr.Interpret(env)
        predicted = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(predicted, result.array)

        arr1 = V.Array(np.array([[1, 0, 2], [0, 3, 0], [1, 1, 1]]))
        env.DefineName('arr1', arr1)
        find_parse_action = csp.expr.parseString('find(arr1)', parseAll=True)
        expr = find_parse_action[0].expr()
        indices = expr.Evaluate(env)
        env.DefineName('indices', indices)
        index_parse_action = csp.expr.parseString('arr1{indices, 1, pad:1=45}', parseAll=True)
        expr = index_parse_action[0].expr()
        result = expr.Interpret(env)
        predicted = np.array(np.array([[1, 2, 45], [3, 45, 45], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(predicted, result.array)

        index_parse_action = csp.expr.parseString('arr1{indices, 1, shrink: 1}', parseAll=True)
        expr = index_parse_action[0].expr()
        result = expr.Interpret(env)
        predicted = np.array(np.array([[1], [3], [1]]))
        np.testing.assert_array_almost_equal(predicted, result.array)

    def TestParsingMapAndFold(self):
        # test map
        env = Env.Environment()
        arr = V.Array(np.arange(4))
        arr2 = V.Array(np.array([4, 5, 6, 7]))
        env.DefineNames(['arr', 'arr2'], [arr, arr2])
        lambda_parse_action = csp.lambdaExpr.parseString('lambda a, b: a + b', parseAll=True)
        add_function = lambda_parse_action[0].expr()
        env.DefineName('add_function', add_function.Interpret(env))
        map_parse_action = csp.expr.parseString('map(add_function, arr, arr2)', parseAll=True)
        expr = map_parse_action[0].expr()
        self.assertIsInstance(expr, E.Map)
        result = expr.Evaluate(env)
        predicted = np.array([4, 6, 8, 10])
        np.testing.assert_array_almost_equal(predicted, result.array)

        # test fold
        fold_parse_action = csp.expr.parseString('fold(add_function, arr, 2, 0)', parseAll=True)
        expr = fold_parse_action[0].expr()
        self.assertIsInstance(expr, E.Fold)
        result = expr.Evaluate(env)
        predicted = np.array([8])
        np.testing.assert_array_almost_equal(predicted, result.array)

        arr3 = V.Array(np.array([[1, 2, 3], [3, 4, 5]]))
        env.DefineName('arr3', arr3)
        fold_parse_action = csp.expr.parseString('fold(add_function, arr3, 4, 1)')
        expr = fold_parse_action[0].expr()
        result = expr.Evaluate(env)
        predicted = np.array([[10], [16]])
        np.testing.assert_array_almost_equal(result.array, predicted)

    def TestProtocolandPostProcessing(self):
        env = Env.Environment()
        parse_action = csp.postProcessing.parseString('post-processing{a=2}')
        expr = parse_action[0].expr()
        result = env.ExecuteStatements(expr)
        self.assertEqual(env.LookUp('a').value, 2)

        parse_action = csp.lambdaExpr.parseString('lambda t: 0*t')
        expr = parse_action[0].expr()
        result = E.FunctionCall(expr, [E.NewArray(E.NewArray(N(1), N(2), N(3)),
                                                  E.NewArray(N(3), N(4), N(5)))]).Evaluate(env)
        predicted = np.array([[0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_almost_equal(predicted, result.array)

    def TestGetUsedVars(self):
        env = Env.Environment()
        parse_action = csp.expr.parseString('[i for i in 0:10]', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.GetUsedVariables()
        self.assertEqual(used_vars, set())

        parse_action = csp.array.parseString('[i*2 for j in 0:2:4]', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.GetUsedVariables()
        self.assertEqual(used_vars, set(['i']))

        parse_action = csp.array.parseString('[i+j*5 for j in 1:3 for l in 2:4]', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.GetUsedVariables()
        self.assertEqual(used_vars, set(['i']))

        env.DefineName('a', V.Simple(1))
        env.DefineName('b', V.Simple(2))
        parse_action = csp.expr.parseString('a + b', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.GetUsedVariables()
        self.assertEqual(used_vars, set(['a', 'b']))

        parse_action = csp.expr.parseString('if a then b else 0', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.GetUsedVariables()
        self.assertEqual(used_vars, set(['a', 'b']))

    def TestFindIndexTxt(self):
        proto_file = 'projects/FunctionalCuration/test/protocols/test_find_index.txt'
        proto = fc.Protocol(proto_file)
        proto.Run()

    def TestCorePostProcTxt(self):
        proto_file = 'projects/FunctionalCuration/test/protocols/test_core_postproc.txt'
        proto = fc.Protocol(proto_file)
        proto.Run()

    def TestGraphTxt(self):
        proto_file = 'projects/FunctionalCuration/protocols/GraphState.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestSyntaxInterface_TestGraphTxt')
        proto.SetModel(Model.TestOdeModel(1))
        proto.Run()

    def TestParsingInputs(self):
        parse_action = csp.inputs.parseString('inputs{X=1}', parseAll=True)
        expr = parse_action[0].expr()
        for each in expr:
            self.assertIsInstance(each, S.Assign)

        # test below is just to test that we get the correct output for a protocol error
        # TODO - it's commented out because it causes a protocol error every time
#     def TestProtocolError(self):
#         proto_file = 'projects/FunctionalCuration/test/protocols/test_error_msg.txt'
#         proto = fc.Protocol(proto_file)
#         proto.Run()

    def TestParsingRanges(self):
        # test parsing uniform range
        parse_action = csp.range.parseString('range t units s uniform 0:10', parseAll=True)
        expr = parse_action[0].expr()
        expr.Initialise(Env.Environment())
        self.assertIsInstance(expr, Ranges.UniformRange)
        r = list(range(11))
        for i, num in enumerate(expr):
            self.assertAlmostEqual(r[i], num)

        # test parsing vector range
        parse_action = csp.range.parseString('range run units dimensionless vector [1, 3, 4]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Ranges.VectorRange)
        expr.Initialise(Env.Environment())
        r = [1, 3, 4]
        for i, num in enumerate(expr):
            self.assertAlmostEqual(r[i], num)

        # test parsing while range
        parse_action = csp.range.parseString('range rpt units dimensionless while rpt < 5', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Ranges.While)
        expr.Initialise(Env.Environment())
        r = list(range(5))
        for i, num in enumerate(expr):
            self.assertAlmostEqual(r[i], num)

    def TestParsingSimulations(self):
        # test parsing timecourse simulation
        parse_action = csp.simulation.parseString(
            'simulation sim = timecourse { range time units ms uniform 0:10 }', parseAll=True)
        expr = parse_action[0].expr()
        expr.Initialise()
        a = 5
        expr.SetModel(Model.TestOdeModel(a))
        run_sim = expr.Run()
        np.testing.assert_array_almost_equal(run_sim.LookUp('a').array, np.array([5] * 11))
        np.testing.assert_array_almost_equal(run_sim.LookUp('y').array, np.array([t * 5 for t in range(11)]))

        # test parsing tasks
        parse_action = csp.tasks.parseString("""tasks {
                                simulation timecourse { range time units second uniform 1:10 }
                                simulation timecourse { range time units second uniform 10:20 }
                                 }""", parseAll=True)
        expr = parse_action[0].expr()
        for sim in expr:
            self.assertIsInstance(sim, Simulations.AbstractSimulation)

    def TestParsingModifiers(self):
        parse_action = csp.modifierWhen.parseString('at start', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr, Modifiers.AbstractModifier.START_ONLY)

        parse_action = csp.modifierWhen.parseString('at each loop', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr, Modifiers.AbstractModifier.EACH_LOOP)

        parse_action = csp.modifierWhen.parseString('at end', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr, Modifiers.AbstractModifier.END_ONLY)

        parse_action = csp.modifier.parseString('at start set model:a = 5.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Modifiers.SetVariable)
        self.assertEqual(expr.variableName, 'model:a')
        self.assertEqual(expr.valueExpr.value.value, 5)

        parse_action = csp.modifier.parseString('at start set model:t = 10.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Modifiers.SetVariable)
        self.assertEqual(expr.variableName, 'model:t')
        self.assertEqual(expr.valueExpr.value.value, 10)

        parse_action = csp.modifier.parseString('at start save as state_name', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Modifiers.SaveState)
        self.assertEqual(expr.stateName, 'state_name')

        parse_action = csp.modifier.parseString('at start reset to state_name', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Modifiers.ResetState)
        self.assertEqual(expr.stateName, 'state_name')
