
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
        
    def TestParsingAddition(self):
        parse_action = csp.expr.parseString('1.0 + 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Plus)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 3.0)
        
    def TestParsingSubtraction(self):
        parse_action = csp.expr.parseString('5.0 - 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Minus)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 3.0)
        
    def TestParsingTimes(self):
        parse_action = csp.expr.parseString('4.0 * 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Times)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 8.0)
        
    def TestParsingDivisionandInfinityValue(self):
        parse_action = csp.expr.parseString('6.0 / 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Divide)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 3.0)
        
        parse_action = csp.expr.parseString('1/MathML:infinity', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Divide)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
    def TestParsingPower(self):
        parse_action = csp.expr.parseString('4.0 ^ 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Power)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 16.0)
        
    def TestParsingGt(self):
        parse_action = csp.expr.parseString('4.0 > 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Gt)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('2.0 > 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Gt)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
    def TestParsingLt(self):
        parse_action = csp.expr.parseString('4.0 < 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Lt)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        parse_action = csp.expr.parseString('2.0 < 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Lt)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
    def TestParsingLeq(self):
        parse_action = csp.expr.parseString('4.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Leq)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        parse_action = csp.expr.parseString('2.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Leq)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('1.0 <= 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Leq)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
    def TestParsingEq(self):
        parse_action = csp.expr.parseString('2.0 == 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Eq)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        
    def TestParsingNeq(self):
        parse_action = csp.expr.parseString('2.0 != 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Neq)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 0)
        
        parse_action = csp.expr.parseString('1.0 != 2.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Neq)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
    def TestParsingAnd(self):
        parse_action = csp.expr.parseString('1.0 && 1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.And)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('0.0 && 1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.And)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
    def TestParsingOrandTrueandFalseValues(self):
        parse_action = csp.expr.parseString('MathML:true || MathML:true', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Or)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('MathML:false || MathML:true', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Or)
        self.assertEqual(expr.Evaluate(env).value, 1)
        
        parse_action = csp.expr.parseString('MathML:false || MathML:false', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Or)
        self.assertEqual(expr.Evaluate(env).value, 0)
        
    def TestParsingNot(self):
        parse_action = csp.expr.parseString('not 1', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Not)
        env = Env.Environment()
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
        
    def TestParsingCeiling(self):
        parse_action = csp.expr.parseString('MathML:ceiling(1.2)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Ceiling)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 2.0)
        
    def TestParsingFloor(self):
        parse_action = csp.expr.parseString('MathML:floor(1.8)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Floor)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1.0)
        
    def TestParsingLnandE(self):
        parse_action = csp.expr.parseString('MathML:ln(MathML:exponentiale)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Ln)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1)
        
    def TestParsingLog(self):
        parse_action = csp.expr.parseString('MathML:log(10)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Log)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1)
        
    def TestParsingExpandInfinityValue(self):
        parse_action = csp.expr.parseString('MathML:exp(MathML:ln(10))', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Exp)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).value, 10)
        
    def TestParsingAbs(self):
        parse_action = csp.expr.parseString('MathML:abs(-10)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Abs)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).value, 10)
        
    def TestParsingRoot(self):
        parse_action = csp.expr.parseString('MathML:root(100)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Root)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).value, 10)
        
    def TestParsingRem(self):
        parse_action = csp.expr.parseString('MathML:rem(100, 97)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Rem)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).value, 3.0)
        
    def TestParsingMax(self):
        parse_action = csp.expr.parseString('MathML:max(100, 97, 105)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Max)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).value, 105)
        
    def TestParsingMinandPiValue(self):
        parse_action = csp.expr.parseString('MathML:min(100, 97, 105, MathML:pi)', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Min)
        env = Env.Environment()
        self.assertAlmostEqual(expr.Evaluate(env).value, 3.1415926535)
    # MathML:exp(MathML:ln(3)) -> 3
    # true, false, exponentiale, infinity, pi, notanumber
    # if a then b else c
        
        