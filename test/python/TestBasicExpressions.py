
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

# Import the module to test
import MathExpressions as M
import Values as V

class TestBasicExpressions(unittest.TestCase):

    def TestAddition(self):
        self.assertAlmostEqual(M.Plus(M.Const(V.Simple(1)), M.Const(V.Simple(2))).Evaluate({}).value, 3)
        self.assertAlmostEqual(M.Plus(M.Const(V.Simple(1)), M.Const(V.Simple(2)), M.Const(V.Simple(4))).Evaluate({}).value, 7)

    def TestMinus(self):
        self.assertAlmostEqual(M.Minus(M.Const(V.Simple(1)), M.Const(V.Simple(2))).Evaluate({}).value, -1)
        self.assertAlmostEqual(M.Minus(M.Const(V.Simple(4))).Evaluate({}).value, -4)
        
    def TestTimes(self):
        self.assertAlmostEqual(M.Times(M.Const(V.Simple(6)), M.Const(V.Simple(2))).Evaluate({}).value, 12)
        self.assertAlmostEqual(M.Times(M.Const(V.Simple(6)), M.Const(V.Simple(2)), M.Const(V.Simple(3))).Evaluate({}).value, 36)
        
    def TestDivide(self):
        self.assertAlmostEqual(M.Divide(M.Const(V.Simple(1)), M.Const(V.Simple(2))).Evaluate({}).value, .5)
        
    def TestMax(self):
        self.assertAlmostEqual(M.Max(M.Const(V.Simple(6)), M.Const(V.Simple(12)), M.Const(V.Simple(2))).Evaluate({}).value, 12)

    def TestMin(self):
        self.assertAlmostEqual(M.Min(M.Const(V.Simple(6)), M.Const(V.Simple(2)), M.Const(V.Simple(12))).Evaluate({}).value, 2)
        
    def TestRem(self):
        self.assertAlmostEqual(M.Rem(M.Const(V.Simple(6)), M.Const(V.Simple(4))).Evaluate({}).value, 2)
        
    def TestPower(self):
        self.assertAlmostEqual(M.Power(M.Const(V.Simple(2)), M.Const(V.Simple(3))).Evaluate({}).value, 8)
        
    def TestRoot(self):
        self.assertAlmostEqual(M.Root(M.Const(V.Simple(16))).Evaluate({}).value, 4)
        self.assertAlmostEqual(M.Root(M.Const(V.Simple(3)), M.Const(V.Simple(8))).Evaluate({}).value, 2)
        
    def TestAbs(self):
        self.assertAlmostEqual(M.Abs(M.Const(V.Simple(-4))).Evaluate({}).value, 4)
        self.assertAlmostEqual(M.Abs(M.Const(V.Simple(4))).Evaluate({}).value, 4)
        
    def TestFloor(self):
        self.assertAlmostEqual(M.Floor(M.Const(V.Simple(1.8))).Evaluate({}).value, 1)
        
    def TestCeiling(self):
        self.assertAlmostEqual(M.Ceiling(M.Const(V.Simple(1.2))).Evaluate({}).value, 2)
    
    def TestExp(self):
        self.assertAlmostEqual(M.Exp(M.Const(V.Simple(3))).Evaluate({}).value, 20.0855369231)
    
    def TestLn(self):
        self.assertAlmostEqual(M.Ln(M.Const(V.Simple(3))).Evaluate({}).value, 1.0986122886) 
        
    def TestLog(self):
        self.assertAlmostEqual(M.Log(M.Const(V.Simple(3))).Evaluate({}).value, 0.4771212547196)
        self.assertAlmostEqual(M.Log(M.Const(V.Simple(4)), M.Const(V.Simple(3))).Evaluate({}).value, 0.79248125036)
    