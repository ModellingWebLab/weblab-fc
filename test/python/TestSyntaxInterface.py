
"""Copyright (c) 2005-2013, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

# Import the modules to test
import CompactSyntaxParser as CSP
CSP.ImportPythonImplementation()
csp = CSP.CompactSyntaxParser

import ArrayExpressions as A
import Values as V
import Environment as Env
import Expressions as E
import Statements as S
import numpy as np
import MathExpressions as M
import Protocol
from ErrorHandling import ProtocolError

def N(number):
    return M.Const(V.Simple(number))

class TestSyntaxInterface(unittest.TestCase):
    def TestParsingNumber(self):
        # number
        parse_action = csp.expr.parseString('1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Const)
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
        self.assertIsInstance(expr, M.Plus)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 3.0)
  
        # minus
        parse_action = csp.expr.parseString('5.0 - 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Minus)
        self.assertEqual(expr.Evaluate(env).value, 3.0)
   
        # times
        parse_action = csp.expr.parseString('4.0 * 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Times)
        self.assertEqual(expr.Evaluate(env).value, 8.0)
        
        # division and infinity
        parse_action = csp.expr.parseString('6.0 / 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Divide)
        self.assertEqual(expr.Evaluate(env).value, 3.0)
        
        parse_action = csp.expr.parseString('1/MathML:infinity', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Divide)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        # power
        parse_action = csp.expr.parseString('4.0 ^ 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Power)
        self.assertEqual(expr.Evaluate(env).value, 16.0)
        
    def TestParsingLogicalOperations(self):
        # greater than
        parse_action = csp.expr.parseString('4.0 > 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Gt)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('2.0 > 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Gt)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        # less than
        parse_action = csp.expr.parseString('4.0 < 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Lt)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        parse_action = csp.expr.parseString('2.0 < 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Lt)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        # less than or equal to
        parse_action = csp.expr.parseString('4.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Leq)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        parse_action = csp.expr.parseString('2.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Leq)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('1.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Leq)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        # equal to
        parse_action = csp.expr.parseString('2.0 == 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Eq)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        
        # not equal to and not a number
        parse_action = csp.expr.parseString('2.0 != 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Neq)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        parse_action = csp.expr.parseString('1.0 != 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Neq)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('MathML:notanumber != MathML:notanumber', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Neq)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        # and
        parse_action = csp.expr.parseString('1.0 && 1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.And)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('0.0 && 1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.And)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        # or and true or false
        parse_action = csp.expr.parseString('MathML:true || MathML:true', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Or)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('MathML:false || MathML:true', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Or)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('MathML:false || MathML:false', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Or)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        # not
        parse_action = csp.expr.parseString('not 1', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Not)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        parse_action = csp.expr.parseString('not 0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Not)
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
        self.assertIsInstance(expr, M.Ceiling)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 2.0)
        
        # floor
        parse_action = csp.expr.parseString('MathML:floor(1.8)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Floor)
        self.assertEqual(expr.Evaluate(env).value, 1.0)
        
        # ln and exponentiale value
        parse_action = csp.expr.parseString('MathML:ln(MathML:exponentiale)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Ln)
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)
        
        # evaluate expression in environment will parse and evaluate so do the parse action line
        # the expr line, and the expr.Evaluate(env) line return result of evaluating 
        
        # log
        parse_action = csp.expr.parseString('MathML:log(10)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Log)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        # exp and infinity value
        parse_action = csp.expr.parseString('MathML:exp(MathML:ln(10))', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Exp)
        self.assertAlmostEqual(expr.Evaluate(env).value, 10)
        
        # abs
        parse_action = csp.expr.parseString('MathML:abs(-10)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Abs)
        self.assertAlmostEqual(expr.Evaluate(env).value, 10)
        
        # root
        parse_action = csp.expr.parseString('MathML:root(100)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Root)
        self.assertAlmostEqual(expr.Evaluate(env).value, 10)
        
        # rem
        parse_action = csp.expr.parseString('MathML:rem(100, 97)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Rem)
        self.assertAlmostEqual(expr.Evaluate(env).value, 3.0)
        
        # max
        parse_action = csp.expr.parseString('MathML:max(100, 97, 105)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Max)
        self.assertAlmostEqual(expr.Evaluate(env).value, 105)
        
        # min and pi value
        parse_action = csp.expr.parseString('MathML:min(100, 97, 105, MathML:pi)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Min)
        self.assertAlmostEqual(expr.Evaluate(env).value, 3.1415926535)
        
        # null
        parse_action = csp.expr.parseString('null', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Const)
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
        self.assertIsInstance(expr, A.NewArray)
        env = Env.Environment()
        np.testing.assert_array_almost_equal(expr.Evaluate(env).array, np.array([1, 2, 3]))
        
    def TestParsingAccessor(self):
        parse_action = csp.expr.parseString('1.IS_SIMPLE_VALUE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)

        parse_action = csp.expr.parseString('[1, 2].IS_ARRAY', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('(1, 2).IS_TUPLE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('[1, 2].IS_TUPLE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.Evaluate(env).value, 0)
        
        parse_action = csp.expr.parseString('[1, 2, 3].SHAPE.IS_ARRAY', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        self.assertAlmostEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('[1, 2, 3].SHAPE', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        np.testing.assert_array_equal(expr.Evaluate(env).array, np.array([3]))
        
        parse_action = csp.expr.parseString('[1, 2, 3].NUM_DIMS', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, E.Accessor)
        np.testing.assert_array_equal(expr.Evaluate(env).array, np.array([1]))
        
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
         expr.Evaluate(env) # checked simply by not raising protocol error
         
         # test assign
         env = Env.Environment()
         parse_action = csp.assignStmt.parseString('a = 1.0 + 2.0', parseAll=True)
         expr = parse_action[0].expr()
         self.assertIsInstance(expr, S.Assign)
         expr.Evaluate(env)
         self.assertEqual(env.LookUp('a').value, 3)
         
         parse_action = csp.assignStmt.parseString('b, c = 1.0, 2.0', parseAll=True)
         expr = parse_action[0].expr()
         self.assertIsInstance(expr, S.Assign)
         expr.Evaluate(env)
         self.assertEqual(env.LookUp('b').value, 1)
         self.assertEqual(env.LookUp('c').value, 2)
         
         parse_action = csp.assignStmt.parseString('d, e, f = 1.0, 2 + 2.0, (3*4)-2', parseAll=True)
         expr = parse_action[0].expr()
         self.assertIsInstance(expr, S.Assign)
         expr.Evaluate(env)
         self.assertEqual(env.LookUp('d').value, 1)
         self.assertEqual(env.LookUp('e').value, 4)
         self.assertEqual(env.LookUp('f').value, 10)
         
         # test return
         parse_action = csp.returnStmt.parseString('return 1', parseAll=True)
         expr = parse_action[0].expr()
         self.assertIsInstance(expr, S.Return)
         results = expr.Evaluate(env)
         self.assertEqual(results.value, 1)
         
         parse_action = csp.returnStmt.parseString('return d + e', parseAll=True)
         expr = parse_action[0].expr()
         self.assertIsInstance(expr, S.Return)
         results = expr.Evaluate(env)
         self.assertEqual(results.value, 5)

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
         result = E.FunctionCall(expr, [M.Const(V.DefaultParameter())]).Evaluate(env)
         self.assertEqual(result.value, 5)
         

         #  'def f(a=1): return a\nassert f(default) == 1'
         
    def TestArrayComprehensions(self):
        env = Env.Environment()
        parse_action = csp.array.parseString('[i for i in 0:10]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, A.NewArray)
        result = expr.Evaluate(env)
        predicted = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_almost_equal(predicted, result.array)

        parse_action = csp.array.parseString('[i*2 for i in 0:2:4]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, A.NewArray)
        result = expr.Evaluate(env)
        predicted = np.array([0, 4])
        np.testing.assert_array_almost_equal(predicted, result.array)
        
        parse_action = csp.array.parseString('[i+j*5 for i in 1:3 for j in 2:4]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, A.NewArray)
        result = expr.Evaluate(env)
        predicted = np.array([[11, 16], [12, 17]])
        np.testing.assert_array_almost_equal(predicted, result.array)
        
        env = Env.Environment()
        arr = V.Array(np.arange(10))
        env.DefineName('arr', arr)
        parse_action = csp.expr.parseString('arr[1:2:10]', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, A.View)
        result = expr.Evaluate(env)
        predicted = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_almost_equal(predicted, result.array)

    def TestParsingArrayExpressions(self):
        #view
        env = Env.Environment()
        view_arr = V.Array(np.arange(10))
        env.DefineName('view_arr', view_arr)
        view_parse_action = csp.expr.parseString('view_arr[4]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, A.View)
        result = expr.Evaluate(env)
        predicted = np.array(4)
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        view_parse_action = csp.expr.parseString('view_arr[2:5]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, A.View)
        result = expr.Evaluate(env)
        predicted = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(result.array, predicted)

        view_parse_action = csp.expr.parseString('view_arr[1:2:10]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, A.View)
        result = expr.Evaluate(env)
        predicted = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_almost_equal(predicted, result.array)
        
        env = Env.Environment()
        view_arr = V.Array(np.array([[0, 1, 2, 3, 4], [7, 8, 12, 3, 9]]))
        env.DefineName('view_arr', view_arr)
        view_parse_action = csp.expr.parseString('view_arr[1$2]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, A.View)
        result = expr.Evaluate(env)
        predicted = np.array([2, 12])
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        view_parse_action = csp.expr.parseString('view_arr[1$(3):]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, A.View)
        result = expr.Evaluate(env)
        predicted = np.array([[3, 4], [3, 9]])
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        view_parse_action = csp.expr.parseString('view_arr[*$1]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, A.View)
        result = expr.Evaluate(env)
        predicted = np.array(8)
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        view_parse_action = csp.expr.parseString('view_arr[*$1][0]', parseAll=True)
        expr = view_parse_action[0].expr()
        self.assertIsInstance(expr, A.View)
        result = expr.Evaluate(env)
        predicted = np.array(1)
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # find
        arr = V.Array(np.arange(4))
        env.DefineName('arr', arr)
        find_parse_action = csp.expr.parseString('find(arr)', parseAll=True)
        expr = find_parse_action[0].expr()
        self.assertIsInstance(expr, A.Find)
        find_result = expr.Evaluate(env)
        predicted = np.array([[1], [2], [3]])
        np.testing.assert_array_almost_equal(find_result.array, predicted)
        
        find_arr = V.Array(np.array([[1, 0 , 3, 0, 0], [0, 7, 0, 0, 10], [0, 0, 13, 14, 0],
                                     [0, 0, 0, 19, 20], [0, 0, 0, 0, 25]]))
        index_arr = V.Array(np.arange(1,26).reshape(5,5))
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
        self.assertIsInstance(expr, A.Index)
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
        
        # map
        arr2 = V.Array(np.array([4, 5, 6, 7]))
        env.DefineName('arr2', arr2)
        lambda_parse_action = csp.lambdaExpr.parseString('lambda a, b: a + b', parseAll=True)
        add_function = lambda_parse_action[0].expr()
        env.DefineName('add_function', add_function.Interpret(env))
        map_parse_action = csp.expr.parseString('map(add_function, arr, arr2)', parseAll=True)
        expr = map_parse_action[0].expr()
        self.assertIsInstance(expr, A.Map)
        result = expr.Evaluate(env)
        predicted = np.array([4, 6, 8, 10])
        np.testing.assert_array_almost_equal(predicted, result.array)
        
        # fold 
        fold_parse_action = csp.expr.parseString('fold(add_function, arr, 2, 0)', parseAll=True)
        expr = fold_parse_action[0].expr()
        self.assertIsInstance(expr, A.Fold)
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
        self.assertEquals(env.LookUp('a').value, 2)
        
        parse_action = csp.lambdaExpr.parseString('lambda t: 0*t')
        expr = parse_action[0].expr()
        result = E.FunctionCall(expr, [A.NewArray(A.NewArray(N(1), N(2), N(3)), A.NewArray(N(3), N(4), N(5)))]).Evaluate(env)
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
        
        env.DefineName('a', N(1))
        env.DefineName('b', N(2))
        parse_action = csp.expr.parseString('a + b', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.GetUsedVariables()
        self.assertEqual(used_vars, set(['a', 'b']))
        
        parse_action = csp.expr.parseString('if a then b else 0', parseAll=True)
        expr = parse_action[0].expr()
        used_vars = expr.GetUsedVariables()
        self.assertEqual(used_vars, set(['a', 'b']))
        
    def TestTxtFiles(self):
        # Parse the protocol into a sequence of post-processing statements
        proto_file = 'projects/FunctionalCuration/test/protocols/compact/test_find_index.txt'
        proto = Protocol.Protocol(proto_file)
        proto.Run()
        
        proto_file = 'projects/FunctionalCuration/test/protocols/compact/test_core_postproc.txt'
        proto = Protocol.Protocol(proto_file)
        proto.Run()
        
        # test below is just to test that we get the correct output for a protocol error
#     def TestProtocolError(self):
#         proto_file = 'projects/FunctionalCuration/test/protocols/compact/test_error_msg.txt'
#         proto = Protocol.Protocol(proto_file)
#         proto.Run()
        
        
        
        