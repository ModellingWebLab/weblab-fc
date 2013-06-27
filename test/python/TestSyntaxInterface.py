
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
    def TestParsingNumbers(self):
        parse_action = csp.expr.parseString('1.0', parseAll=True)
        expr = parse_action[0].expr()
        self.assertIsInstance(expr, M.Const)
        env = Env.Environment()
        self.assertEqual(expr.Evaluate(env).value, 1.0)
        
#         parse_action = csp.expr.parseString('1.0 + 2.0', parseAll=True)
#         expr = parse_action[0].expr()
#         self.assertIsInstance(expr, M.Plus)
#         env = Env.Environment()
#         self.assertEqual(expr.Evaluate(env).value, 3.0)
        
        # a  ->  E.NameLookup('a')
        # 1+2, 1/2, 3+(4*5)-6