
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
import ArrayExpressions as A
import Values as V
import Environment as Env
import Expressions as E
import Statements as S
import numpy as np
import MathExpressions as M
import math
import sys

from ErrorHandling import ProtocolError

def N(number):
    return M.Const(V.Simple(number))

class TestArrayExpressions(unittest.TestCase):
    def TestNewArray(self):
        arr = A.NewArray(N(1), N(2), N(3)) # simple one-dimensional array
        predictedArr = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(arr.Evaluate({}).array, predictedArr)
        arr = A.NewArray(A.NewArray(N(1), N(2), N(3)), A.NewArray(N(3), N(3), N(2)))
        predictedArr = np.array([[1, 2, 3], [3, 3, 2]])
        np.testing.assert_array_almost_equal(arr.Evaluate({}).array, predictedArr)
         
    def TestViews(self):
        arr = A.NewArray(N(1), N(2), N(3), N(4))
         
        view = A.View(arr, E.TupleExpression(N(1), M.Const(V.Null()))) # two parameters: beginning and end, null represents end of original array
        predictedArr = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        view = A.View(arr, E.TupleExpression(N(0), N(2), N(3))) # three parameters: beginning, step, end
        predictedArr = np.array([1, 3])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        view = A.View(arr, E.TupleExpression(N(3), N(-1), N(0))) # negative step
        predictedArr = np.array([4, 3, 2])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
         
        view = A.View(arr, E.TupleExpression(N(1), N(0), N(1))) # 0 as step
        predicted = 2
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predicted)
        view = A.View(arr, N(1)) # same as immediately above, but only one number passed instead of a step of 0
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predicted)
         
        array = A.NewArray(A.NewArray(N(-2), N(-1), N(0)), A.NewArray(N(1), N(2), N(3)), A.NewArray(N(4), N(5), N(6))) # testing many aspects of views of a 2-d array
        view = A.View(array, E.TupleExpression(N(0), M.Const(V.Null()), N(2)), E.TupleExpression(N(2), N(-1), N(0))) # can slice stepping forward, backward, picking a position...etc
        predictedArr = np.array([[0, -1], [3, 2]])
        self.assertEqual(view.Evaluate({}).array.ndim, 2) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
         
        array = A.NewArray(A.NewArray(A.NewArray(N(-2), N(-1), N(0)), 
                                      A.NewArray(N(1), N(2), N(3))), 
                           A.NewArray(A.NewArray(N(4), N(5), N(6)), 
                                      A.NewArray(N(1), N(2), N(3)))) # 3-d array
        view = A.View(array, E.TupleExpression(N(0), M.Const(V.Null()), M.Const(V.Null())), 
                      E.TupleExpression(M.Const(V.Null()), N(-1), N(0)), 
                      E.TupleExpression(N(0), N(2))) 
        predictedArr = np.array([[[1, 2]], [[1, 2]]])
        self.assertEqual(view.Evaluate({}).array.ndim, 3) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        # use four parameters in the tuples to specify dimension explicitly
        view = A.View(array, E.TupleExpression(N(0), N(0), M.Const(V.Null()), N(2)), 
                      E.TupleExpression(N(1), N(2), N(-1), N(0)), 
                      E.TupleExpression(N(2), N(0), M.Const(V.Null()), N(2))) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        # use four parameters in the tuples to specify dimension with a mix of implicit and explicit declarations
        view = A.View(array, E.TupleExpression(N(0), M.Const(V.Null()), M.Const(V.Null())), 
                      E.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                      E.TupleExpression(N(0), M.Const(V.Null()), N(2))) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        # test leaving some parameters out so they fall to default
        view = A.View(array, E.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                      E.TupleExpression(N(0), M.Const(V.Null()), M.Const(V.Null()))) 
        view2 = A.View(array, E.TupleExpression(N(0), M.Const(V.Null()), M.Const(V.Null())), 
                       E.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                       E.TupleExpression(M.Const(V.Null()), M.Const(V.Null()), N(1), M.Const(V.Null())))
        predictedArr = np.array([[[1, 2, 3]], [[1, 2, 3]]])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        # test leaving some parameters out so they get set to slice determined by dimension null
        view = A.View(array, E.TupleExpression(M.Const(V.Null()), N(0), N(1), N(2))) 
        view2 = A.View(array, E.TupleExpression(N(0), N(0), N(1), N(2)), 
                      E.TupleExpression(N(1), N(0), N(1), N(2)),
                      E.TupleExpression(N(2), N(0), N(1), N(2))) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, view2.Evaluate({}).array)
        
        # checks to make sure the "default default" is equivalent to a tuple of (Null, Null, 1, Null), also checks to make sure implicitly defined slices go to the first dimension that is not assigned explicitly
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, view2.Evaluate({}).array)
        view = A.View(array, E.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0))) # only specified dimension is in middle
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        view = A.View(array, E.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                      E.TupleExpression(N(0), N(0), M.Const(V.Null()), M.Const(V.Null())))
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr) # tests explicitly assigning dimension one before explicitly defining dimension zero
        
        #View([[0,1,2],[3,4,5]], M.Const(1), E.Tuple(M.Const(1), M.Const(3))) == [4,5]
        
        
        array = A.NewArray(A.NewArray(N(0), N(1), N(2)), A.NewArray(N(3), N(4), N(5)))
        view = A.View(array, E.TupleExpression(N(1)), E.TupleExpression(N(1), N(3)))
        predictedArr = np.array([4, 5])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        #View([[0,1,2],[3,4,5]], M.Const(1), E.Tuple(M.Const(1))) == 4 
        array = A.NewArray(A.NewArray(N(0), N(1), N(2)), A.NewArray(N(3), N(4), N(5)))
        view = A.View(array, N(1), E.TupleExpression(N(1)))
        predictedArr = np.array([4])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
    def TestArrayCreationProtocolErrors(self):
        array = A.NewArray(A.NewArray(A.NewArray(N(-2), N(-1), N(0)), 
                                      A.NewArray(N(1), N(2), N(3))), 
                           A.NewArray(A.NewArray(N(4), N(5), N(6)), 
                                      A.NewArray(N(1), N(2), N(3))))
        view = A.View(array, E.TupleExpression(N(0), N(0), M.Const(V.Null()), M.Const(V.Null())), 
                      E.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                      E.TupleExpression(N(2), N(0), M.Const(V.Null()), M.Const(V.Null())), 
                      E.TupleExpression(N(2), N(0), M.Const(V.Null()), M.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # more tuple slices than dimensions
        view = A.View(array, E.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0), N(1), N(4)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # too many arguments for a slice
        view = A.View(N(1))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # first argument must be an array
        view = A.View(array, E.TupleExpression(N(5)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # index goes out of range
        view = A.View(array, E.TupleExpression(N(3), N(0), M.Const(V.Null()), M.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # attempts to assign dimension three when array only has dimensions 0, 1, and 2
        view = A.View(array, E.TupleExpression(N(1), N(1), N(0)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start is after end with positive step
        view = A.View(array, E.TupleExpression(N(0), N(-1), N(1)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start is before end with negative step
        view = A.View(array, E.TupleExpression(N(0), N(0), N(4))) 
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start and end aren't equal and there's a step of 0
        view = A.View(array, E.TupleExpression(N(-4), N(1), N(2)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start is before beginning of array
        view = A.View(array, E.TupleExpression(N(-2), N(1), N(3)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # end is after end of array
        view = A.View(array, E.TupleExpression(M.Const(V.Null()), N(1), N(-4)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # end is before beginning of array
        view = A.View(array, E.TupleExpression(N(2), N(1), M.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # beginning is after end of array
           
    def TestArrayComprehension(self):
       env = Env.Environment()
        
       # 1-d array
       counting1d = A.NewArray(E.NameLookUp("i"), E.TupleExpression(N(0), N(0), N(1), N(10), M.Const(V.String("i"))), comprehension=True)
       predictedArr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
       np.testing.assert_array_almost_equal(counting1d.Evaluate(env).array, predictedArr)
        
       # 2-d array, explicitly defined dimensions
       counting2d = A.NewArray(M.Plus(M.Times(E.NameLookUp("i"), N(3)), E.NameLookUp("j")),
                               E.TupleExpression(N(0), N(1), N(1), N(3), M.Const(V.String("i"))),
                               E.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))), 
                               comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate(env).array, predictedArr)
 
       # 2-d array, order of variable definitions opposite of previous test
       counting2d = A.NewArray(M.Plus(M.Times(E.NameLookUp("i"), N(3)), E.NameLookUp("j")),
                                E.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))),
                                E.TupleExpression(N(0), N(1), N(1), N(3), M.Const(V.String("i"))), 
                                comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate(env).array, predictedArr)
        
       # 2-d array with implicitly defined dimension assigned after explicit
       counting2d = A.NewArray(M.Plus(M.Times(E.NameLookUp("i"), N(3)), E.NameLookUp("j")),
                             E.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))),
                             E.TupleExpression(N(1), N(1), N(3), M.Const(V.String("i"))), 
                              comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate(env).array, predictedArr)
        
       # 2-d array with implicitly defined dimension assigned before explicit
       counting2d = A.NewArray(M.Plus(M.Times(E.NameLookUp("i"), N(3)), E.NameLookUp("j")),
                             E.TupleExpression(N(0), N(1), N(3), M.Const(V.String("j"))),
                             E.TupleExpression(N(0), N(1), N(1), N(3), M.Const(V.String("i"))), 
                              comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate(env).array, predictedArr)
        
       # comprehension using arrays in generator expression with one variable
       blocks = A.NewArray(A.NewArray(A.NewArray(M.Plus(N(-10), E.NameLookUp("j")), 
                                                 E.NameLookUp("j")), 
                                      A.NewArray(M.Plus(N(10), E.NameLookUp("j")), 
                                                 M.Plus(N(20), E.NameLookUp("j")))),
                           E.TupleExpression(N(1), N(0), N(1), N(2), M.Const(V.String("j"))),
                           comprehension=True)
       predictedArr = np.array([ [[-10,0], [-9,1]] , [[10,20], [11,21]] ])
       np.testing.assert_array_almost_equal(blocks.Evaluate(Env.Environment()).array, predictedArr)
        
       # two gaps between instead of one
       blocks = A.NewArray(A.NewArray(A.NewArray(M.Plus(N(-10), E.NameLookUp("j")), 
                                                 E.NameLookUp("j")), 
                                      A.NewArray(M.Plus(N(10), E.NameLookUp("j")), 
                                                 M.Plus(N(20), E.NameLookUp("j")))),
                           E.TupleExpression(N(2), N(0), N(1), N(2), M.Const(V.String("j"))),
                           comprehension=True)
       predictedArr = np.array([[[-10, -9], [0, 1]], [[10, 11], [20, 21]]])
       np.testing.assert_array_almost_equal(blocks.Evaluate(Env.Environment()).array, predictedArr)
        
        # comprehension using arrays in generator expression with two variables
       blocks = A.NewArray(A.NewArray(A.NewArray(M.Plus(N(-10), E.NameLookUp("j")), 
                                              E.NameLookUp("j")), 
                                   A.NewArray(M.Plus(N(10), E.NameLookUp("i")), 
                                              M.Plus(N(20), E.NameLookUp("i")))),
                        E.TupleExpression(N(0), N(1), N(2), M.Const(V.String("j"))),
                        E.TupleExpression(N(0), N(1), N(2), M.Const(V.String("i"))),
                        comprehension=True)
       predictedArr = np.array([[ [[-10, 0],[10, 20]] ,[[-10,0],[11,21]] ],[[ [-9, 1],[10, 20] ], [[-9, 1], [11, 21]]]])
       np.testing.assert_array_almost_equal(blocks.Evaluate(Env.Environment()).array, predictedArr)
        
    def TestArrayExpressionProtocolErrors(self):
       env = Env.Environment()
       # creates an empty array because the start is greater than the end
       fail = A.NewArray(E.NameLookUp("i"), E.TupleExpression(N(0), N(10), N(1), N(0), M.Const(V.String("i"))), comprehension=True)
       self.assertRaises(ProtocolError, fail.Evaluate, env)
    
       # creates an empty array because the step is negative when it should be positive
       fail = A.NewArray(E.NameLookUp("i"), E.TupleExpression(N(0), N(0), N(-1), N(10), M.Const(V.String("i"))), comprehension=True)
       self.assertRaises(ProtocolError, fail.Evaluate, env)
        
       blocks = A.NewArray(A.NewArray(A.NewArray(E.NameLookUp("j")), 
                                      A.NewArray(E.NameLookUp("j"))),
                                      E.TupleExpression(N(3), N(0), N(1), N(2), M.Const(V.String("j"))),
                                      comprehension=True)
       self.assertRaises(ProtocolError, blocks.Evaluate, env)
       
       #map(lambda a, b=[0,1,2]: a + b, [1,2,3]) should give an error
       parameters = ['a', 'b']
       body = [S.Return(M.Plus(E.NameLookUp('a'), E.NameLookUp('b')))]
       array_default = E.LambdaExpression(parameters, body,
                                          defaultParameters=[V.DefaultParameter(), V.Array(np.array([0, 1, 2]))])
       a = A.NewArray(N(1), N(2), N(3))
       result = A.Map(array_default, a)
       self.assertRaises(ProtocolError, result.Evaluate, env)
        
    def TestSimpleMap(self):
       env = Env.Environment()
       parameters = ['a', 'b', 'c']
       body = [S.Return(M.Plus(E.NameLookUp('a'), E.NameLookUp('b'), E.NameLookUp('c')))]
       add = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(2))
       b = A.NewArray(N(2), N(4))
       c = A.NewArray(N(3), N(7))
       result = A.Map(add, a, b, c)
       predicted = V.Array(np.array([6, 13]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
    def TestMapWithMultiDimensionalArrays(self):
       env = Env.Environment()
       parameters = ['a', 'b', 'c']
       body = [S.Return(M.Plus(E.NameLookUp('a'), E.NameLookUp('b'), E.NameLookUp('c')))]
       add = E.LambdaExpression(parameters, body)
       a = A.NewArray(A.NewArray(N(1), N(2)), A.NewArray(N(2),N(3)))
       b = A.NewArray(A.NewArray(N(4), N(3)), A.NewArray(N(6),N(1)))
       c = A.NewArray(A.NewArray(N(2), N(2)), A.NewArray(N(8),N(0)))
       result = A.Map(add, a, b, c)
       predicted = V.Array(np.array([[7, 7], [16, 4]]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)   
          
         # more complex function and more complex array
       env = Env.Environment()
       parameters = ['a', 'b', 'c']
       body = [S.Return(M.Times(M.Plus(E.NameLookUp('a'), E.NameLookUp('b')), E.NameLookUp('c')))]
       add_times = E.LambdaExpression(parameters, body)
       a = A.NewArray(A.NewArray(A.NewArray(N(1), N(2)), A.NewArray(N(2),N(3))), A.NewArray(A.NewArray(N(1), N(2)), A.NewArray(N(2),N(3))))
       b = A.NewArray(A.NewArray(A.NewArray(N(4), N(3)), A.NewArray(N(6),N(1))), A.NewArray(A.NewArray(N(0), N(6)), A.NewArray(N(5),N(3))))
       c = A.NewArray(A.NewArray(A.NewArray(N(2), N(2)), A.NewArray(N(8),N(0))), A.NewArray(A.NewArray(N(4), N(2)), A.NewArray(N(2),N(1))))
       result = A.Map(add_times, a, b, c)
       predicted = np.array([[[10, 10], [64, 0]], [[4, 16], [14, 6]]])
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)
       
       # test complicated expression involving views of a default that is an array
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Plus(M.Times(M.Power(E.NameLookUp('a'), N(2)), A.View(E.NameLookUp('b'), N(0))), 
                               M.Times(E.NameLookUp('a'), A.View(E.NameLookUp('b'), N(1))),
                               A.View(E.NameLookUp('b'), N(2))))]
       default_array_test = E.LambdaExpression(parameters, body,
                                               defaultParameters=[V.DefaultParameter(), V.Array(np.array([1, 2, 3]))])
       a = A.NewArray(N(1), N(2), N(3))
       result = A.Map(default_array_test, a)
       predicted = np.array([6, 11, 18])
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)
        
       #map(lambda a, b=[0,1,2]: (a^2)*b[0] + a*b[1] + b[2], [1,2,3]) == [6,11,18]
       
    def TestUsingManyOperationsinFunction(self):
       env = Env.Environment()
       parameters = ['a', 'b', 'c']
       body = [S.Return(M.Times(M.Plus(E.NameLookUp('a'), M.Times(E.NameLookUp('b'), E.NameLookUp('c'))), E.NameLookUp('a')))]
       add_times = E.LambdaExpression(parameters, body)
       a = A.NewArray(A.NewArray(N(1), N(2)), A.NewArray(N(2),N(3)))
       b = A.NewArray(A.NewArray(N(4), N(3)), A.NewArray(N(6),N(1)))
       c = A.NewArray(A.NewArray(N(2), N(2)), A.NewArray(N(8),N(0)))
       result = A.Map(add_times, a, b, c)
       predicted = np.array([[9, 16], [100, 9]])
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)
          
    def TestMapWithFunctionWithDefaults(self):
       env = Env.Environment()
       body = [S.Return(M.Plus(E.NameLookUp('item'), E.NameLookUp('incr')))]
       add = E.LambdaExpression(['item', 'incr'], body, defaultParameters = [V.DefaultParameter(), V.Simple(3)])
       item = A.NewArray(N(1), N(3), N(5))
       result = A.Map(add, item)
       predicted = np.array([4, 6, 8])
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)
       
    def TestNestedFunction(self):
        env = Env.Environment()
        nested_body = [S.Return(M.Plus(E.NameLookUp('input'), E.NameLookUp('outer_var')))]
        nested_function = E.LambdaExpression(["input"], nested_body)
        body = [S.Assign(["nested_fn"], nested_function),
                S.Assign(["outer_var"], N(1)),
                S.Return(M.Eq(E.FunctionCall("nested_fn", [N(1)]), N(2)))]
        nested_scope = E.LambdaExpression([], body)
        nested_call = E.FunctionCall(nested_scope, [])
        result = nested_call.Evaluate(env)
        predicted = np.array([True])
        self.assertEqual(result.value, predicted)
         
    def TestCompileMethodsForMathExpression(self):
       # Minus
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Minus(E.NameLookUp('a'), E.NameLookUp('b')))]
       minus = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(4), N(2))
       b = A.NewArray(N(2), N(1))
       result = A.Map(minus, a, b)
       predicted = V.Array(np.array([2, 1])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Divide
       body = [S.Return(M.Divide(E.NameLookUp('a'), E.NameLookUp('b')))]
       divide = E.LambdaExpression(parameters, body)
       result = A.Map(divide, a, b)
       predicted = V.Array(np.array([2, 2]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
 
       # Remainder
       body = [S.Return(M.Rem(E.NameLookUp('a'), E.NameLookUp('b')))]
       rem = E.LambdaExpression(parameters, body)
       result = A.Map(rem, a, b)
       predicted = V.Array(np.array([0, 0]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
        
       # Power
       body = [S.Return(M.Power(E.NameLookUp('a'), E.NameLookUp('b')))]
       power = E.LambdaExpression(parameters, body)
       result = A.Map(power, a, b)
       predicted = V.Array(np.array([16, 2]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
 
       # Root with one argument
       body = [S.Return(M.Root(E.NameLookUp('a')))]
       root = E.LambdaExpression('a', body)
       result = A.Map(root, a)
       predicted = V.Array(np.array([2, 1.41421356]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
        
       #Root with two arguments
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Root(E.NameLookUp('a'), E.NameLookUp('b')))]
       root = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(3))
       b = A.NewArray(N(8))
       result = A.Map(root, b, a)
       predicted = V.Array(np.array([2])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
 
       # Absolute value
       env = Env.Environment()
       parameters = ['a']
       body = [S.Return(M.Abs(E.NameLookUp('a')))]
       absolute = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(4), N(-2))
       result = A.Map(absolute, a)
       predicted = V.Array(np.array([4, 2]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Exponential 
       env = Env.Environment()
       parameters = ['a']
       body = [S.Return(M.Exp(E.NameLookUp('a')))]
       exponential = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(3))
       result = A.Map(exponential, a)
       predicted = V.Array(np.array([20.0855369231])) 
       #np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Natural Log
       env = Env.Environment()
       body = [S.Return(M.Ln(E.NameLookUp('a')))]
       ln = E.LambdaExpression(parameters, body)
       result = A.Map(ln, a)
       predicted = V.Array(np.array([1.0986122886])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Log with One Argument (Log base 10)
       env = Env.Environment()
       body = [S.Return(M.Log(E.NameLookUp('a')))]
       log = E.LambdaExpression(parameters, body)
       result = A.Map(log, a)
       predicted = V.Array(np.array([0.4771212547196])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Log with two arguments, second is log base qualifier
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Log(E.NameLookUp('a'), E.NameLookUp('b')))]
       log = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(4))
       b = A.NewArray(N(3))
       result = A.Map(log, a, b)
       predicted = V.Array(np.array([0.79248125036])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Max
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Max(E.NameLookUp('a'), E.NameLookUp('b')))]
       max = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(4), N(5), N(1))
       b = A.NewArray(N(3), N(6), N(0))
       result = A.Map(max, a, b)
       predicted = V.Array(np.array([4, 6, 1])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Floor 
       env = Env.Environment()
       parameters = ['a']
       body = [S.Return(M.Floor(E.NameLookUp('a')))]
       floor = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(3.2), N(3.7))
       result = A.Map(floor, a)
       predicted = V.Array(np.array([3, 3])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Ceiling
       env = Env.Environment()
       parameters = ['a']
       body = [S.Return(M.Ceiling(E.NameLookUp('a')))]
       ceiling = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(3.2), N(3.7))
       result = A.Map(ceiling, a)
       predicted = V.Array(np.array([4, 4])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
    
        
    def TestCompileMethodsForLogicalExpressions(self):
       # And
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.And(E.NameLookUp('a'), E.NameLookUp('b')))]
       test_and = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(1))
       b = A.NewArray(N(0), N(1))
       result = A.Map(test_and, a, b)
       predicted = V.Array(np.array([False, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Or
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Or(E.NameLookUp('a'), E.NameLookUp('b')))]
       test_or = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(1), N(0))
       b = A.NewArray(N(0), N(1), N(0))
       result = A.Map(test_or, a, b)
       predicted = V.Array(np.array([True, True, False])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Xor
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Xor(E.NameLookUp('a'), E.NameLookUp('b')))]
       test_xor = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(1), N(0))
       b = A.NewArray(N(0), N(1), N(0))
       result = A.Map(test_xor, a, b)
       predicted = V.Array(np.array([True, False, False])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Not
       env = Env.Environment()
       parameters = ['a']
       body = [S.Return(M.Not(E.NameLookUp('a')))]
       test_not = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0))
       result = A.Map(test_not, a)
       predicted = V.Array(np.array([False, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
  
        # greater than
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Gt(E.NameLookUp('a'), E.NameLookUp('b')))]
       test_greater = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_greater, a, b)
       predicted = V.Array(np.array([True, False, False])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)     
 
        # less than
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Lt(E.NameLookUp('a'), E.NameLookUp('b')))]
       test_less = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_less, a, b)
       predicted = V.Array(np.array([False, False, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)   
  
        # greater than equal to
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Geq(E.NameLookUp('a'), E.NameLookUp('b')))]
       test_greater_eq = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_greater_eq, a, b)
       predicted = V.Array(np.array([True, True, False])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)  
 
        # less than equal to
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Leq(E.NameLookUp('a'), E.NameLookUp('b')))]
       test_less_eq = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_less_eq, a, b)
       predicted = V.Array(np.array([False, True, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
 
        # not equal
       env = Env.Environment()
       parameters = ['a', 'b']
       body = [S.Return(M.Neq(E.NameLookUp('a'), E.NameLookUp('b')))]
       test_not_eq = E.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_not_eq, a, b)
       predicted = V.Array(np.array([True, False, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)   
       
    def TestFold(self):
        # 1-d array, add fold
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(M.Plus(E.NameLookUp('a'), E.NameLookUp('b')))]
        add = E.LambdaExpression(parameters, body)
        array = A.NewArray(N(0), N(1), N(2))
        result = A.Fold(add, array, N(0), N(0)).Interpret(env)
        predicted = np.array([3])
        np.testing.assert_array_almost_equal(result.array, predicted)   
        
        # 2-d array, add fold over dimension 0
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(M.Plus(E.NameLookUp('a'), E.NameLookUp('b')))]
        add = E.LambdaExpression(parameters, body)
        array = A.NewArray(A.NewArray(N(0), N(1), N(2)), A.NewArray(N(3), N(4), N(5)))
        result = A.Fold(add, array, N(0), N(0)).Interpret(env)
        predicted = np.array([[3, 5, 7]])
        np.testing.assert_array_almost_equal(result.array, predicted)   
         
        # 2-d array, add fold over dimension 1 and non-zero initial value
        result = A.Fold(add, array, N(5), N(1)).Interpret(env)
        predicted = np.array([[8],[17]])
        np.testing.assert_array_almost_equal(result.array, predicted)   
        
        # 2-d array, add fold over dimension 1 using implicitly defined dimension
        result = A.Fold(add, array, N(0)).Interpret(env)
        predicted = np.array([[3],[12]])
        np.testing.assert_array_almost_equal(result.array, predicted) 
        
        # 2-d array, add fold over dimension 1 using implicitly defined dimension and initial
        # can you implicitly define initial and explicitly define dimension?
        array = A.NewArray(A.NewArray(N(1), N(2), N(3)), A.NewArray(N(3), N(4), N(5)))
        result = A.Fold(add, array).Interpret(env)
        predicted = np.array([[6],[12]])
        np.testing.assert_array_almost_equal(result.array, predicted)      
        
        # 3-d array, times fold over dimension 0
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(M.Times(E.NameLookUp('a'), E.NameLookUp('b')))]
        times = E.LambdaExpression(parameters, body)
        array = A.NewArray(A.NewArray(A.NewArray(N(1), N(2), N(3)), A.NewArray(N(4), N(2), N(1))),
                           A.NewArray(A.NewArray(N(3), N(0), N(5)), A.NewArray(N(2), N(2), N(1))))
        result = A.Fold(times, array, N(1), N(0)).Interpret(env)
        predicted = np.array([[[3, 0, 15], [8, 4, 1]]])
        np.testing.assert_array_almost_equal(result.array, predicted) 
        
        # 3-d array, times fold over dimension 1
        result = A.Fold(times, array, N(1), N(1)).Interpret(env)
        predicted = np.array([[[4, 4, 3]], [[6, 0, 5]]])
        np.testing.assert_array_almost_equal(result.array, predicted)
         
        # 3-d array, times fold over dimension 2
        result = A.Fold(times, array, N(1), N(2)).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted) 
        
        # 3-d array, times fold over dimension 2 (defined implicitly)
        result = A.Fold(times, array, N(1)).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # 3-d array, times fold over dimension 2 (defined explicitly as default)
        result = A.Fold(times, array, N(1), M.Const(V.DefaultParameter())).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted)
         
        # 3-d array, times fold over dimension 2 (defined implicitly) with no initial value input
        result = A.Fold(times, array).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # 3-d array, times fold over dimension 2 using default parameter for both initial value and dimension
        result = A.Fold(times, array, M.Const(V.DefaultParameter()), M.Const(V.DefaultParameter())).Interpret(env)
        predicted = np.array([[[6], [8]], [[0], [4]]])
        np.testing.assert_array_almost_equal(result.array, predicted)
        
    def TestFoldWithDifferentFunctions(self):
        # fold with max function
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(M.Max(E.NameLookUp('a'), E.NameLookUp('b')))]
        max_function = E.LambdaExpression(parameters, body)
        array = A.NewArray(A.NewArray(N(0), N(1), N(8)), A.NewArray(N(3), N(4), N(5)))
        result = A.Fold(max_function, array, N(0), N(0)).Interpret(env)
        predicted = np.array([[3, 4, 8]])
        np.testing.assert_array_almost_equal(result.array, predicted) 
        
        # fold with max function with an initial value that affects output
        array = A.NewArray(A.NewArray(N(0), N(1), N(8)), A.NewArray(N(3), N(4), N(5)))
        result = A.Fold(max_function, array, N(7), N(0)).Interpret(env)
        predicted = np.array([[7, 7, 8]])
        np.testing.assert_array_almost_equal(result.array, predicted) 
        
        # fold with min function
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(M.Min(E.NameLookUp('a'), E.NameLookUp('b')))]
        min_function = E.LambdaExpression(parameters, body)
        array = A.NewArray(A.NewArray(N(0), N(1), N(8)), A.NewArray(N(3), N(-4), N(5)))
        result = A.Fold(min_function, array).Interpret(env)
        predicted = np.array([[0],[-4]])
        np.testing.assert_array_almost_equal(result.array, predicted) 
        
        # fold with minus function
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(M.Minus(E.NameLookUp('a'), E.NameLookUp('b')))]
        minus_function = E.LambdaExpression(parameters, body)
        array = A.NewArray(A.NewArray(N(0), N(1), N(4)), A.NewArray(N(3), N(1), N(5)))
        result = A.Fold(minus_function, array, N(5), N(1)).Interpret(env)
        predicted = np.array([[0], [-4]])
        np.testing.assert_array_almost_equal(result.array, predicted) 
        
        # fold with divide function
        env = Env.Environment()
        parameters = ['a', 'b']
        body = [S.Return(M.Divide(E.NameLookUp('a'), E.NameLookUp('b')))]
        divide_function = E.LambdaExpression(parameters, body)
        array = A.NewArray(A.NewArray(N(2), N(2), N(4)), A.NewArray(N(16), N(2), N(1)))
        result = A.Fold(divide_function, array, N(32), N(1)).Interpret(env)
        predicted = np.array([[2], [1]])
        np.testing.assert_array_almost_equal(result.array, predicted)
        
    def TestFind(self): 
        # Find with 2-d array as input
        env = Env.Environment()
        array = A.NewArray(A.NewArray(N(1), N(0), N(2), N(3)), A.NewArray(N(0), N(0), N(3), N(9)))
        result = A.Find(array).Evaluate(env)
        predicted = np.array(np.array([[0, 0], [0, 2], [0, 3], [1,2], [1,3]]))
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # Should raise error if non-array passed in
        array = M.Const(V.Simple(1))
        result = A.Find(array)
        self.assertRaises(ProtocolError, result.Evaluate, env)
        
    def TestIndex(self):
        # 2-d pad first dimension to the left
        env = Env.Environment()
        array = A.NewArray(A.NewArray(N(1), N(0), N(2)), A.NewArray(N(0), N(3), N(0)), A.NewArray(N(1), N(1), N(1)))
        # [ [1, 0, 2]
        #   [0, 3, 0]
        #   [1, 1, 1] ]
        find = A.Find(array)
        result = A.Index(array, find, N(1), N(0), N(1), N(45)).Interpret(env)
        predicted = np.array(np.array([[1, 2, 45], [3, 45, 45], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)

        # 2-d pad dimension 0 up
        result = A.Index(array, A.Find(array), N(0), N(0), N(1), N(45)).Interpret(env)
        predicted = np.array(np.array([[1, 3, 2], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # 2-d pad dimension 0 to the left with defaults for shrink, pad, pad_value
        result = A.Index(array, find, N(0)).Interpret(env)
        predicted = np.array(np.array([[1, 3, 2], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # 2-d pad first dimension to the right
        result = A.Index(array, find, N(1), N(0), N(-1), N(45)).Interpret(env)
        predicted = np.array(np.array([[45, 1, 2], [45, 45, 3], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # 2-d pad first dimension to the right with default max value for pad_value
        result = A.Index(array, find, N(1), N(0), N(-1)).Interpret(env)
        predicted = np.array(np.array([[sys.float_info.max, 1, 2], [sys.float_info.max, sys.float_info.max, 3], [1, 1, 1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # 2-d shrink first dimension to the left with defaults for pad and pad_value
        result = A.Index(array, find, N(1), N(1)).Interpret(env)
        predicted = np.array(np.array([[1], [3], [1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # 2-d shrink first dimension to the right with defaults for pad and pad_value
        result = A.Index(array, find, N(1), N(-1)).Interpret(env)
        predicted = np.array(np.array([[2], [3], [1]]))
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # 1-d
        env = Env.Environment()
        array = A.NewArray(N(1), N(0), N(2), N(0))
        # [1, 0, 2, 0]
        find = A.Find(array)
        result = A.Index(array, find).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)
        
        # a few more tests for 1-d array, should all yield the same result
        result = A.Index(array, find, N(0)).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)
        result = A.Index(array, find, N(0), N(1)).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)
        result = A.Index(array, find, N(0), N(0), N(0), N(0)).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)
        result = A.Index(array, find, N(0), N(0), N(-1), N(100)).Interpret(env)
        predicted = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.array, predicted)
        
    def TestIndexProtocolErrors(self):
        # index over dimension 2 in 2 dimensional array (out of range)
        env = Env.Environment()
        array = A.NewArray(A.NewArray(N(1), N(0), N(2)), A.NewArray(N(0), N(3), N(0)), A.NewArray(N(1), N(1), N(1)))
        find = A.Find(array)
        result = A.Index(array, find, N(2), N(0), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)
        
        # index by shrinking and padding at the same time
        result = A.Index(array, find, N(1), N(1), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)
        
        # shrink and pad are both 0, but output is irregular
        result = A.Index(array, find, N(1), N(0), N(0), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)
        
        # input an array of indices that is not 2-d
        result = A.Index(array, A.NewArray(N(1), N(2)), N(1), N(1), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)
        
        # input array for dimension value instead of simple value
        result = A.Index(array, A.NewArray(N(1), N(2)), array, N(1), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)
        
        # input simple value for array instead of array
        result = A.Index(N(1), A.NewArray(N(1), N(2)), array, N(1), N(1), N(45))
        self.assertRaises(ProtocolError, result.Interpret, env)
        
    def TestJoinAndStretch(self):
        env = Env.Environment()
        env.DefineName('repeated_arr', V.Array(np.array([1, 2, 3])))
        stretch = A.NewArray(E.NameLookUp("repeated_arr"),
                             E.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))), 
                             comprehension=True)
        predictedArr = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_almost_equal(stretch.Evaluate(env).array, predictedArr)
        
        stretch = A.NewArray(E.NameLookUp("repeated_arr"),
                             E.TupleExpression(N(0), N(0), N(1), N(3), M.Const(V.String("j"))), 
                             comprehension=True)
        predictedArr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_almost_equal(stretch.Evaluate(env).array, predictedArr)
        
        