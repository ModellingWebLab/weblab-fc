
import numpy as np
import unittest

import fc
import fc.language.expressions as E
import fc.language.statements as S
import fc.language.values as V
import fc.simulations.model as Model
import fc.simulations.modifiers as Modifiers
import fc.simulations.ranges as Ranges
import fc.simulations.simulations as Simulations
from fc.environment import Environment

from fc.parsing.CompactSyntaxParser import CompactSyntaxParser as CSP


class TestSyntaxInterface(unittest.TestCase):
    """Test expr methods of compact syntax parser."""

    def test_parsing_number(self):
        parse_action = CSP.expr.parseString('1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Const)
        env = Environment()
        self.assertEqual(expr.evaluate(env).value, 1.0)

    def test_parsing_variable(self):
        parse_action = CSP.expr.parseString('a', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NameLookUp)
        env = Environment()
        env.define_name('a', V.Simple(3.0))
        self.assertEqual(expr.evaluate(env).value, 3.0)

    def test_parsing_math_operations(self):
        # plus
        parse_action = CSP.expr.parseString('1.0 + 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Plus)
        env = Environment()
        self.assertEqual(expr.evaluate(env).value, 3.0)

        # minus
        parse_action = CSP.expr.parseString('5.0 - 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Minus)
        self.assertEqual(expr.evaluate(env).value, 3.0)

        # times
        parse_action = CSP.expr.parseString('4.0 * 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Times)
        self.assertEqual(expr.evaluate(env).value, 8.0)

        # division and infinity
        parse_action = CSP.expr.parseString('6.0 / 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Divide)
        self.assertEqual(expr.evaluate(env).value, 3.0)

        parse_action = CSP.expr.parseString('1/MathML:infinity', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Divide)
        self.assertEqual(expr.evaluate(env).value, 0)

        # power
        parse_action = CSP.expr.parseString('4.0 ^ 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Power)
        self.assertEqual(expr.evaluate(env).value, 16.0)

    def test_parsing_logical_operations(self):
        # greater than
        parse_action = CSP.expr.parseString('4.0 > 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Gt)
        env = Environment()
        self.assertEqual(expr.evaluate(env).value, 1)

        parse_action = CSP.expr.parseString('2.0 > 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Gt)
        self.assertEqual(expr.evaluate(env).value, 0)

        # less than
        parse_action = CSP.expr.parseString('4.0 < 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Lt)
        self.assertEqual(expr.evaluate(env).value, 0)

        parse_action = CSP.expr.parseString('2.0 < 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Lt)
        self.assertEqual(expr.evaluate(env).value, 0)

        # less than or equal to
        parse_action = CSP.expr.parseString('4.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Leq)
        self.assertEqual(expr.evaluate(env).value, 0)

        parse_action = CSP.expr.parseString('2.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Leq)
        self.assertEqual(expr.evaluate(env).value, 1)

        parse_action = CSP.expr.parseString('1.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Leq)
        self.assertEqual(expr.evaluate(env).value, 1)

        # equal to
        parse_action = CSP.expr.parseString('2.0 == 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Eq)
        self.assertEqual(expr.evaluate(env).value, 1)

        # not equal to and not a number
        parse_action = CSP.expr.parseString('2.0 != 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Neq)
        self.assertEqual(expr.evaluate(env).value, 0)

        parse_action = CSP.expr.parseString('1.0 != 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Neq)
        self.assertEqual(expr.evaluate(env).value, 1)

        parse_action = CSP.expr.parseString('MathML:notanumber != MathML:notanumber', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Neq)
        self.assertEqual(expr.evaluate(env).value, 1)

        # and
        parse_action = CSP.expr.parseString('1.0 && 1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.And)
        self.assertEqual(expr.evaluate(env).value, 1)

        parse_action = CSP.expr.parseString('0.0 && 1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.And)
        self.assertEqual(expr.evaluate(env).value, 0)

        # or and true or false
        parse_action = CSP.expr.parseString('MathML:true || MathML:true', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Or)
        self.assertEqual(expr.evaluate(env).value, 1)

        parse_action = CSP.expr.parseString('MathML:false || MathML:true', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Or)
        self.assertEqual(expr.evaluate(env).value, 1)

        parse_action = CSP.expr.parseString('MathML:false || MathML:false', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Or)
        self.assertEqual(expr.evaluate(env).value, 0)

        # not
        parse_action = CSP.expr.parseString('not 1', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Not)
        self.assertEqual(expr.evaluate(env).value, 0)

        parse_action = CSP.expr.parseString('not 0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Not)
        self.assertEqual(expr.evaluate(env).value, 1)

    def test_parsing_complicated_math(self):
        parse_action = CSP.expr.parseString('1.0 + (4.0 * 2.0)', parseAll=True)
        expr = parse_action[0].expr()
        env = Environment()
        self.assertEqual(expr.evaluate(env).value, 9.0)

        parse_action = CSP.expr.parseString('(2.0 ^ 3.0) + (5.0 * 2.0) - 10.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr.evaluate(env).value, 8.0)

        parse_action = CSP.expr.parseString('2.0 ^ 3.0 == 5.0 + 3.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr.evaluate(env).value, 1)

    def test_parsing_MathML_funcs(self):
        # ceiling
        parse_action = CSP.expr.parseString('MathML:ceiling(1.2)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Ceiling)
        env = Environment()
        self.assertEqual(expr.evaluate(env).value, 2.0)

        # floor
        parse_action = CSP.expr.parseString('MathML:floor(1.8)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Floor)
        self.assertEqual(expr.evaluate(env).value, 1.0)

        # ln and exponentiale value
        parse_action = CSP.expr.parseString('MathML:ln(MathML:exponentiale)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Ln)
        self.assertAlmostEqual(expr.evaluate(env).value, 1)

        # log
        parse_action = CSP.expr.parseString('MathML:log(10)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Log)
        self.assertEqual(expr.evaluate(env).value, 1)

        # exp and infinity value
        parse_action = CSP.expr.parseString('MathML:exp(MathML:ln(10))', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Exp)
        self.assertAlmostEqual(expr.evaluate(env).value, 10)

        # abs
        parse_action = CSP.expr.parseString('MathML:abs(-10)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Abs)
        self.assertAlmostEqual(expr.evaluate(env).value, 10)

        # root
        parse_action = CSP.expr.parseString('MathML:root(100)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Root)
        self.assertAlmostEqual(expr.evaluate(env).value, 10)

        # rem
        parse_action = CSP.expr.parseString('MathML:rem(100, 97)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Rem)
        self.assertAlmostEqual(expr.evaluate(env).value, 3.0)

        # max
        parse_action = CSP.expr.parseString('MathML:max(100, 97, 105)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Max)
        self.assertAlmostEqual(expr.evaluate(env).value, 105)

        # min and pi value
        parse_action = CSP.expr.parseString('MathML:min(100, 97, 105, MathML:pi)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Min)
        self.assertAlmostEqual(expr.evaluate(env).value, 3.1415926535)

        # null
        parse_action = CSP.expr.parseString('null', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Const)
        self.assertIsInstance(expr.value, V.Null)

    def test_parsing_if(self):
        parse_action = CSP.expr.parseString('if 1 then 2 + 3 else 4-2', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.If)
        env = Environment()
        self.assertAlmostEqual(expr.evaluate(env).value, 5)

        parse_action = CSP.expr.parseString('if MathML:false then 2 + 3 else 4-2', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.If)
        self.assertAlmostEqual(expr.evaluate(env).value, 2)

    def test_parsing_tuple_expression(self):
        parse_action = CSP.expr.parseString('(1, 2)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.TupleExpression)
        env = Environment()
        self.assertAlmostEqual(expr.evaluate(env).values[0].value, 1)
        self.assertAlmostEqual(expr.evaluate(env).values[1].value, 2)

    def test_parsing_array(self):
        parse_action = CSP.expr.parseString('[1, 2, 3]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NewArray)
        env = Environment()
        np.testing.assert_array_almost_equal(expr.evaluate(env).array, np.array([1, 2, 3]))

    def test_parsing_accessor(self):
        # is simple true
        parse_action = CSP.expr.parseString('1.IS_SIMPLE_VALUE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        env = Environment()
        self.assertAlmostEqual(expr.evaluate(env).value, 1)

        # is array true
        parse_action = CSP.expr.parseString('[1, 2].IS_ARRAY', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.evaluate(env).value, 1)

        # is tuple true
        parse_action = CSP.expr.parseString('(1, 2).IS_TUPLE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.evaluate(env).value, 1)

        # is tuple false
        parse_action = CSP.expr.parseString('[1, 2].IS_TUPLE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.evaluate(env).value, 0)

        # multiple accessors- .shape.is_array
        parse_action = CSP.expr.parseString('[1, 2, 3].SHAPE.IS_ARRAY', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.evaluate(env).value, 1)

        # shape
        parse_action = CSP.expr.parseString('[1, 2, 3].SHAPE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        np.testing.assert_array_equal(expr.evaluate(env).array, np.array([3]))

        # number of dimensions
        parse_action = CSP.expr.parseString('[1, 2, 3].NUM_DIMS', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        np.testing.assert_array_equal(expr.evaluate(env).array, np.array([1]))

        # number of elements
        parse_action = CSP.expr.parseString('[1, 2, 3].NUM_ELEMENTS', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        np.testing.assert_array_equal(expr.evaluate(env).array, np.array([3]))

    def test_statements(self):
        # test assertion
        parse_action = CSP.assert_stmt.parseString("assert 1", parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Assert)
        env = Environment()
        expr.evaluate(env)  # checked simply by not raising protocol error

        # test assign

        # assign one variable to an expression
        env = Environment()
        parse_action = CSP.assign_stmt.parseString('a = 1.0 + 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Assign)
        expr.evaluate(env)
        self.assertEqual(env.look_up('a').value, 3)

        # assign two variables at once to numbers
        parse_action = CSP.assign_stmt.parseString('b, c = 1.0, 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Assign)
        expr.evaluate(env)
        self.assertEqual(env.look_up('b').value, 1)
        self.assertEqual(env.look_up('c').value, 2)

        # assign three variables at once to expressions
        parse_action = CSP.assign_stmt.parseString('d, e, f = 1.0, 2 + 2.0, (3*4)-2', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Assign)
        expr.evaluate(env)
        self.assertEqual(env.look_up('d').value, 1)
        self.assertEqual(env.look_up('e').value, 4)
        self.assertEqual(env.look_up('f').value, 10)

        # test return

        # return one number
        parse_action = CSP.return_stmt.parseString('return 1', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Return)
        results = expr.evaluate(env)
        self.assertEqual(results.value, 1)

        # return one expression involving variables
        parse_action = CSP.return_stmt.parseString('return d + e', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Return)
        results = expr.evaluate(env)
        self.assertEqual(results.value, 5)

        # return two numbers
        parse_action = CSP.return_stmt.parseString('return 1, 3', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, S.Return)
        result1, result2 = expr.evaluate(env).values
        self.assertEqual(result1.value, 1)
        self.assertEqual(result2.value, 3)

        # test statement list
        parse_action = CSP.stmt_list.parseString('z = lambda a: a+2\nassert z(2) == 4', parseAll=True)
        result = parse_action[0].expr()
        env.execute_statements(result)

    def test_parsing_lambda(self):
        # no default, one variable
        env = Environment()
        parse_action = CSP.lambda_expr.parseString('lambda a: a + 1', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.LambdaExpression)
        result = E.FunctionCall(expr, [E.N(3)]).evaluate(env)
        self.assertEqual(result.value, 4)

        # no default, two variables
        env = Environment()
        parse_action = CSP.lambda_expr.parseString('lambda a, b: a * b', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.LambdaExpression)
        result = E.FunctionCall(expr, [E.N(4), E.N(2)]).evaluate(env)
        self.assertEqual(result.value, 8)

        # test lambda with defaults unused
        env = Environment()
        parse_action = CSP.lambda_expr.parseString('lambda a=2, b=3: a + b', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.LambdaExpression)
        result = E.FunctionCall(expr, [E.N(2), E.N(6)]).evaluate(env)
        self.assertEqual(result.value, 8)

        # test lambda with defaults used
        env = Environment()
        parse_action = CSP.lambda_expr.parseString('lambda a=2, b=3: a + b', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.LambdaExpression)
        result = E.FunctionCall(expr, [E.Const(V.DefaultParameter())]).evaluate(env)
        self.assertEqual(result.value, 5)

    def test_array_comprehensions(self):
        env = Environment()
        parse_action = CSP.array.parseString('[i for i in 0:10]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NewArray)
        result = expr.evaluate(env)
        predicted = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_almost_equal(predicted, result.array)

        parse_action = CSP.array.parseString('[i*2 for i in 0:2:4]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NewArray)
        result = expr.evaluate(env)
        predicted = np.array([0, 4])
        np.testing.assert_array_almost_equal(predicted, result.array)

        parse_action = CSP.array.parseString('[i+j*5 for i in 1:3 for j in 2:4]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.NewArray)
        result = expr.evaluate(env)
        predicted = np.array([[11, 16], [12, 17]])
        np.testing.assert_array_almost_equal(predicted, result.array)

        env = Environment()
        arr = V.Array(np.arange(10))
        env.define_name('arr', arr)
        parse_action = CSP.expr.parseString('arr[1:2:10]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.evaluate(env)
        predicted = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_almost_equal(predicted, result.array)

    def test_parsing_views(self):
        env = Environment()
        view_arr = V.Array(np.arange(10))
        env.define_name('view_arr', view_arr)
        view_parse_action = CSP.expr.parseString('view_arr[4]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.evaluate(env)
        predicted = np.array(4)
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = CSP.expr.parseString('view_arr[2:5]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.evaluate(env)
        predicted = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = CSP.expr.parseString('view_arr[1:2:10]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.evaluate(env)
        predicted = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_almost_equal(predicted, result.array)

        env = Environment()
        view_arr = V.Array(np.array([[0, 1, 2, 3, 4], [7, 8, 12, 3, 9]]))
        env.define_name('view_arr', view_arr)
        view_parse_action = CSP.expr.parseString('view_arr[1$2]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.evaluate(env)
        predicted = np.array([2, 12])
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = CSP.expr.parseString('view_arr[1$(3):]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.evaluate(env)
        predicted = np.array([[3, 4], [3, 9]])
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = CSP.expr.parseString('view_arr[*$1]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.evaluate(env)
        predicted = np.array(8)
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = CSP.expr.parseString('view_arr[*$1][0]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, E.View)
        result = expr.evaluate(env)
        predicted = np.array(1)
        np.testing.assert_array_almost_equal(result.array, predicted)

    def parsing_find_and_index_array(self):
        env = Environment()
        arr = V.Array(np.arange(4))
        env.define_name('arr', arr)
        find_parse_action = CSP.expr.parseString('find(arr)', parseAll=True)
        expr = find_parse_action[0].expr()
        self.assertIsInstance(expr, E.Find)
        find_result = expr.evaluate(env)
        predicted = np.array([[1], [2], [3]])
        np.testing.assert_array_almost_equal(find_result.array, predicted)

        find_arr = V.Array(np.array([[1, 0, 3, 0, 0], [0, 7, 0, 0, 10], [0, 0, 13, 14, 0],
                                     [0, 0, 0, 19, 20], [0, 0, 0, 0, 25]]))
        index_arr = V.Array(np.arange(1, 26).reshape(5, 5))
        env.define_name('find_arr', find_arr)
        env.define_name('index_arr', index_arr)
        find_parse_action = CSP.expr.parseString('find(find_arr)', parseAll=True)
        expr = find_parse_action[0].expr()
        indices_from_find = expr.evaluate(env)
        env.define_name('indices_from_find', indices_from_find)
        index_parse_action = CSP.expr.parseString('index_arr{indices_from_find, 1, pad:1=0}')
        expr = index_parse_action[0].expr()
        index_result = expr.interpret(env)
        predicted = np.array([[1, 3], [7, 10], [13, 14], [19, 20], [25, 0]])
        np.testing.assert_array_almost_equal(index_result.array, predicted)

        env.define_name('find_result', find_result)
        index_parse_action = CSP.expr.parseString('arr{find_result}', parseAll=True)
        expr = index_parse_action[0].expr()
        self.assertIsInstance(expr, E.Index)
        result = expr.interpret(env)
        predicted = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(predicted, result.array)

        arr1 = V.Array(np.array([[1, 0, 2], [0, 3, 0], [1, 1, 1]]))
        env.define_name('arr1', arr1)
        find_parse_action = CSP.expr.parseString('find(arr1)', parseAll=True)
        expr = find_parse_action[0].expr()
        indices = expr.evaluate(env)
        env.define_name('indices', indices)
        index_parse_action = CSP.expr.parseString('arr1{indices, 1, pad:1=45}', parseAll=True)
        expr = index_parse_action[0].expr()
        result = expr.interpret(env)
        predicted = np.array(np.array([[1, 2, 45], [3, 45, 45], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(predicted, result.array)

        index_parse_action = CSP.expr.parseString('arr1{indices, 1, shrink: 1}', parseAll=True)
        expr = index_parse_action[0].expr()
        result = expr.interpret(env)
        predicted = np.array(np.array([[1], [3], [1]]))
        np.testing.assert_array_almost_equal(predicted, result.array)

    def test_parsing_map_and_fold(self):
        # test map
        env = Environment()
        arr = V.Array(np.arange(4))
        arr2 = V.Array(np.array([4, 5, 6, 7]))
        env.define_names(['arr', 'arr2'], [arr, arr2])
        lambda_parse_action = CSP.lambda_expr.parseString('lambda a, b: a + b', parseAll=True)
        add_function = lambda_parse_action[0].expr()
        env.define_name('add_function', add_function.interpret(env))
        map_parse_action = CSP.expr.parseString('map(add_function, arr, arr2)', parseAll=True)
        expr = map_parse_action[0].expr()
        self.assertIsInstance(expr, E.Map)
        result = expr.evaluate(env)
        predicted = np.array([4, 6, 8, 10])
        np.testing.assert_array_almost_equal(predicted, result.array)

        # test fold
        fold_parse_action = CSP.expr.parseString('fold(add_function, arr, 2, 0)', parseAll=True)
        expr = fold_parse_action[0].expr()
        self.assertIsInstance(expr, E.Fold)
        result = expr.evaluate(env)
        predicted = np.array([8])
        np.testing.assert_array_almost_equal(predicted, result.array)

        arr3 = V.Array(np.array([[1, 2, 3], [3, 4, 5]]))
        env.define_name('arr3', arr3)
        fold_parse_action = CSP.expr.parseString('fold(add_function, arr3, 4, 1)')
        expr = fold_parse_action[0].expr()
        result = expr.evaluate(env)
        predicted = np.array([[10], [16]])
        np.testing.assert_array_almost_equal(result.array, predicted)

    def test_protocol_and_post_processing(self):
        env = Environment()
        parse_action = CSP.post_processing.parseString('post-processing{a=2}')
        expr = parse_action[0].expr()
        result = env.execute_statements(expr)
        self.assertEqual(env.look_up('a').value, 2)

        parse_action = CSP.lambda_expr.parseString('lambda t: 0*t')
        expr = parse_action[0].expr()
        result = E.FunctionCall(expr, [E.NewArray(E.NewArray(E.N(1), E.N(2), E.N(3)),
                                                  E.NewArray(E.N(3), E.N(4), E.N(5)))]).evaluate(env)
        predicted = np.array([[0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_almost_equal(predicted, result.array)

    def test_get_used_vars(self):
        env = Environment()
        parse_action = CSP.expr.parseString('[i for i in 0:10]', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.get_used_variables()
        self.assertEqual(used_vars, set())

        parse_action = CSP.array.parseString('[i*2 for j in 0:2:4]', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.get_used_variables()
        self.assertEqual(used_vars, set(['i']))

        parse_action = CSP.array.parseString('[i+j*5 for j in 1:3 for l in 2:4]', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.get_used_variables()
        self.assertEqual(used_vars, set(['i']))

        env.define_name('a', V.Simple(1))
        env.define_name('b', V.Simple(2))
        parse_action = CSP.expr.parseString('a + b', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.get_used_variables()
        self.assertEqual(used_vars, set(['a', 'b']))

        parse_action = CSP.expr.parseString('if a then b else 0', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.get_used_variables()
        self.assertEqual(used_vars, set(['a', 'b']))

    def test_find_index_txt(self):
        pass
        # TODO fix failing test
        # proto_file = 'test/protocols/test_find_index.txt'
        # proto = fc.Protocol(proto_file)
        # proto.run()

    def test_core_post_proc_txt(self):
        pass
        # TODO fix failing test
        # ModelInterface not iterable
        # proto_file = 'test/protocols/test_core_postproc.txt'
        # proto = fc.Protocol(proto_file)
        # proto.run()

    def test_graph_txt(self):
        proto_file = 'protocols/GraphState.txt'
        proto = fc.Protocol(proto_file)
        proto.set_output_folder('TestSyntaxInterface_TestGraphTxt')
        proto.set_model(Model.TestOdeModel(1))
        proto.run()

    def test_parsing_inputs(self):
        parse_action = CSP.inputs.parseString('inputs{X=1}', parseAll=True)
        expr = parse_action[0].expr()
        for each in expr:
            self.assertIsInstance(each, S.Assign)

        # test below is just to test that we get the correct output for a protocol error
        # TODO - it's commented out because it causes a protocol error every time
#     def test_protocol_error(self):
#         proto_file = 'test/protocols/test_error_msg.txt'
#         proto = fc.Protocol(proto_file)
#         proto.run()

    def test_parsing_ranges(self):
        # test parsing uniform range
        parse_action = CSP.range.parseString('range t units s uniform 0:10', parseAll=True)
        expr = parse_action[0].expr()
        expr.initialise(Environment())
        self.assertIsInstance(expr, Ranges.UniformRange)
        r = list(range(11))
        for i, num in enumerate(expr):
            self.assertAlmostEqual(r[i], num)

        # test parsing vector range
        parse_action = CSP.range.parseString('range run units dimensionless vector [1, 3, 4]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Ranges.VectorRange)
        expr.initialise(Environment())
        r = [1, 3, 4]
        for i, num in enumerate(expr):
            self.assertAlmostEqual(r[i], num)

        # test parsing while range
        parse_action = CSP.range.parseString('range rpt units dimensionless while rpt < 5', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Ranges.While)
        expr.initialise(Environment())
        r = list(range(5))
        for i, num in enumerate(expr):
            self.assertAlmostEqual(r[i], num)

    def test_parsing_simulations(self):
        # test parsing timecourse simulation
        parse_action = CSP.simulation.parseString(
            'simulation sim = timecourse { range time units ms uniform 0:10 }', parseAll=True)
        expr = parse_action[0].expr()
        expr.initialise()
        a = 5
        expr.set_model(Model.TestOdeModel(a))
        run_sim = expr.run()
        np.testing.assert_array_almost_equal(run_sim.look_up('a').array, np.array([5] * 11))
        np.testing.assert_array_almost_equal(run_sim.look_up('y').array, np.array([t * 5 for t in range(11)]))

        # test parsing tasks
        parse_action = CSP.tasks.parseString("""tasks {
                                simulation timecourse { range time units second uniform 1:10 }
                                simulation timecourse { range time units second uniform 10:20 }
                                 }""", parseAll=True)
        expr = parse_action[0].expr()
        for sim in expr:
            self.assertIsInstance(sim, Simulations.AbstractSimulation)

    def test_parsing_modifiers(self):
        parse_action = CSP.modifier_when.parseString('at start', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr, Modifiers.AbstractModifier.START_ONLY)

        parse_action = CSP.modifier_when.parseString('at each loop', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr, Modifiers.AbstractModifier.EACH_LOOP)

        parse_action = CSP.modifier_when.parseString('at end', parseAll=True)
        expr = parse_action[0].expr()
        self.assertEqual(expr, Modifiers.AbstractModifier.END_ONLY)

        parse_action = CSP.modifier.parseString('at start set model:a = 5.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Modifiers.SetVariable)
        self.assertEqual(expr.variable_name, 'model:a')
        self.assertEqual(expr.value_expr.value.value, 5)

        parse_action = CSP.modifier.parseString('at start set model:t = 10.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Modifiers.SetVariable)
        self.assertEqual(expr.variable_name, 'model:t')
        self.assertEqual(expr.value_expr.value.value, 10)

        parse_action = CSP.modifier.parseString('at start save as state_name', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Modifiers.SaveState)
        self.assertEqual(expr.state_name, 'state_name')

        parse_action = CSP.modifier.parseString('at start reset to state_name', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, Modifiers.ResetState)
        self.assertEqual(expr.state_name, 'state_name')

