
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
        self.assertEqual(expr.Evaluate(env).value, 1)
        
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
        
    def TestParsingArrayExpression(self):
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


        
        # assign- 'a = 1' ; 'a, b = 1, 2'
        # return - return 1+2  ; return 1, 2
        # assert - assert 0
        # lambda - lambda a, b=1: return a+b
    #    parse_action = csp.stmtList.parseString('''f = lambda a: a+2\nassert f(1) == 3''')
    #    statements = parse_action[0].expr()
    #    env.ExecuteStatements(statements)