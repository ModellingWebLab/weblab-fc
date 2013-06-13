
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
import Environment as E
import Statement

from ErrorHandling import ProtocolError

def N(number):
    return M.Const(V.Simple(number))

class TestFunctions(unittest.TestCase):
 
    def TestFuncDefinitions(self):
        env = E.Environment()
        parameters = ["a", "b"]
        body = [Statement.Return(M.NameLookUp("b"), M.NameLookUp("a"))]
        swap = M.LambdaExpression(parameters, body)
        env.ExecuteStatements([Statement.Assign(["swap"], swap)])
        args = [N(1), N(2)]
        swap_call = M.FunctionCall("swap", args)
        result = swap_call.Evaluate(env)
        self.assert_(isinstance(result, V.Tuple))
        self.assertAlmostEqual(result.values[0].value, 2)
        self.assertAlmostEqual(result.values[1].value, 1)
        env.ExecuteStatements([Statement.Assign(parameters, swap_call)])
        defined = env.DefinedNames()   
        self.assertEqual(len(defined), 3)  
        self.assertEqual(env.LookUp('a').value, 2) 
        self.assertEqual(env.LookUp('b').value, 1)
         
        args = [N(3), N(5)]
        swap_call = M.FunctionCall("swap", args)
        result = swap_call.Evaluate(env)
        self.assert_(isinstance(result, V.Tuple))
        self.assertAlmostEqual(result.values[0].value, 5)
        self.assertAlmostEqual(result.values[1].value, 3)  

    def TestNestedFunction(self):
        env = E.Environment()
        nested_body = [Statement.Return(M.Plus(M.NameLookUp('input'), M.NameLookUp('outer_var')))]
        nested_function = M.LambdaExpression(["input"], nested_body)
        body = [Statement.Assign(["nested_fn"], nested_function),
                Statement.Assign(["outer_var"], N(1)),
                Statement.Return(M.Eq(M.FunctionCall("nested_fn", [N(1)]), N(2)))]
        nested_scope = M.LambdaExpression([], body)
        nested_call = M.FunctionCall(nested_scope, [])
        result = nested_call.Evaluate(env)
        self.assertEqual(result.value, 1)
        
    def TestFunctionsWithDefaultsUsed(self):
        # Function has default which is used
        env = E.Environment()
        nested_body = [Statement.Return(M.Plus(M.NameLookUp('input'), M.NameLookUp('outer_var')))]
        nested_function = M.LambdaExpression(["input"], nested_body, defaultParameters=[V.Simple(1)])
        body = [Statement.Assign(["nested_fn"], nested_function),
                Statement.Assign(["outer_var"], N(1)),
                Statement.Return(M.Eq(M.FunctionCall("nested_fn", [M.Const(V.DefaultParameter())]), N(2)))]
        nested_scope = M.LambdaExpression([], body)
        nested_call = M.FunctionCall(nested_scope, [])
        result = nested_call.Evaluate(env)
        self.assertEqual(result.value, 1)    
    
    def TestFunctionsWithDefaultsUnused(self):
        # Function has default, but value is explicitly assigned in this case   
        env = E.Environment()
        nested_body = [Statement.Return(M.Plus(M.NameLookUp('input'), M.NameLookUp('outer_var')))]
        nested_function = M.LambdaExpression(["input"], nested_body, defaultParameters=[V.Simple(0)])
        body = [Statement.Assign(["nested_fn"], nested_function),
                Statement.Assign(["outer_var"], N(1)),
                Statement.Return(M.Eq(M.FunctionCall("nested_fn", [N(1)]), N(2)))]
        nested_scope = M.LambdaExpression([], body)
        nested_call = M.FunctionCall(nested_scope, [])
        result = nested_call.Evaluate(env)
        self.assertEqual(result.value, 1)            
        
    def TestMultipleDefaultValues(self):
        env = E.Environment()
        parameters = ['a', 'b', 'c']
        body = [Statement.Return(M.Plus(M.NameLookUp('a'), M.NameLookUp('b'), M.NameLookUp('c')))]
        add = M.LambdaExpression(parameters, body, defaultParameters=[V.Simple(1), V.Simple(2), V.Simple(3)])
        args = [M.Const(V.DefaultParameter()), M.Const(V.DefaultParameter()), M.Const(V.DefaultParameter())]
        add_call = M.FunctionCall(add, args)
        result = add_call.Evaluate(env)
        self.assertEqual(result.value, 6)

        args = [N(3)]
        add_call = M.FunctionCall(add, args)
        result = add_call.Evaluate(env)
        self.assertEqual(result.value, 8)
        
        args = [M.Const(V.DefaultParameter()), M.Const(V.DefaultParameter()), N(1)]
        add_call = M.FunctionCall(add, args)
        result = add_call.Evaluate(env)
        self.assertEqual(result.value, 4)
        
        args = [N(4), M.Const(V.DefaultParameter()), N(4)]
        add_call = M.FunctionCall(add, args)
        result = add_call.Evaluate(env)
        self.assertEqual(result.value, 10)