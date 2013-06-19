
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
import Environment as E
import Statement
import numpy as np
import MathExpressions as M
import math

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
         
        view = A.View(arr, M.TupleExpression(N(1), M.Const(V.Null()))) # two parameters: beginning and end, null represents end of original array
        predictedArr = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        view = A.View(arr, M.TupleExpression(N(0), N(2), N(3))) # three parameters: beginning, step, end
        predictedArr = np.array([1, 3])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        view = A.View(arr, M.TupleExpression(N(3), N(-1), N(0))) # negative step
        predictedArr = np.array([4, 3, 2])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
         
        view = A.View(arr, M.TupleExpression(N(1), N(0), N(1))) # 0 as step
        predicted = 2
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predicted)
        view = A.View(arr, M.TupleExpression(N(1))) # same as immediately above, but only one number passed instead of a step of 0
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predicted)
         
        array = A.NewArray(A.NewArray(N(-2), N(-1), N(0)), A.NewArray(N(1), N(2), N(3)), A.NewArray(N(4), N(5), N(6))) # testing many aspects of views of a 2-d array
        view = A.View(array, M.TupleExpression(N(0), M.Const(V.Null()), N(2)), M.TupleExpression(N(2), N(-1), N(0))) # can slice stepping forward, backward, picking a position...etc
        predictedArr = np.array([[0, -1], [3, 2]])
        self.assertEqual(view.Evaluate({}).array.ndim, 2) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
         
        array = A.NewArray(A.NewArray(A.NewArray(N(-2), N(-1), N(0)), 
                                      A.NewArray(N(1), N(2), N(3))), 
                           A.NewArray(A.NewArray(N(4), N(5), N(6)), 
                                      A.NewArray(N(1), N(2), N(3)))) # 3-d array
        view = A.View(array, M.TupleExpression(N(0), M.Const(V.Null()), M.Const(V.Null())), 
                      M.TupleExpression(M.Const(V.Null()), N(-1), N(0)), 
                      M.TupleExpression(N(0), N(2))) 
        predictedArr = np.array([[[1, 2]], [[1, 2]]])
        self.assertEqual(view.Evaluate({}).array.ndim, 3) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # use four parameters in the tuples to specify dimension explicitly
        view = A.View(array, M.TupleExpression(N(0), N(0), M.Const(V.Null()), N(2)), 
                      M.TupleExpression(N(1), N(2), N(-1), N(0)), 
                      M.TupleExpression(N(2), N(0), M.Const(V.Null()), N(2))) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # use four parameters in the tuples to specify dimension with a mix of implicit and explicit declarations
        view = A.View(array, M.TupleExpression(N(0), M.Const(V.Null()), M.Const(V.Null())), 
                      M.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                      M.TupleExpression(N(0), M.Const(V.Null()), N(2))) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # test leaving some parameters out so they fall to default
        view = A.View(array, M.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                      M.TupleExpression(N(0), M.Const(V.Null()), M.Const(V.Null()))) 
        view2 = A.View(array, M.TupleExpression(N(0), M.Const(V.Null()), M.Const(V.Null())), 
                       M.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                       M.TupleExpression(M.Const(V.Null()), M.Const(V.Null()), N(1), M.Const(V.Null())))
        predictedArr = np.array([[[1, 2, 3]], [[1, 2, 3]]])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # checks to make sure the "default default" is equivalent to a tuple of (Null, Null, 1, Null), also checks to make sure implicitly defined slices go to the first dimension that is not assigned explicitly
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, view2.Evaluate({}).array)
        view = A.View(array, M.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0))) # only specified dimension is in middle
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        view = A.View(array, M.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                      M.TupleExpression(N(0), N(0), M.Const(V.Null()), M.Const(V.Null())))
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr) # tests explicitly assigning dimension one before explicitly defining dimension zero
         
    def TestArrayCreationProtocolErrors(self):
        array = A.NewArray(A.NewArray(A.NewArray(N(-2), N(-1), N(0)), 
                                      A.NewArray(N(1), N(2), N(3))), 
                           A.NewArray(A.NewArray(N(4), N(5), N(6)), 
                                      A.NewArray(N(1), N(2), N(3))))
        view = A.View(array, M.TupleExpression(N(0), N(0), M.Const(V.Null()), M.Const(V.Null())), 
                      M.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0)), 
                      M.TupleExpression(N(2), N(0), M.Const(V.Null()), M.Const(V.Null())), 
                      M.TupleExpression(N(2), N(0), M.Const(V.Null()), M.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # more tuple slices than dimensions
        view = A.View(array, M.TupleExpression(N(1), M.Const(V.Null()), N(-1), N(0), N(1), N(4)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # too many arguments for a slice
        view = A.View(N(1))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # first argument must be an array
        view = A.View(array, M.TupleExpression(N(5)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # index goes out of range
        view = A.View(array, M.TupleExpression(N(3), N(0), M.Const(V.Null()), M.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # attempts to assign dimension three when array only has dimensions 0, 1, and 2
        view = A.View(array, M.TupleExpression(N(1), N(1), N(0)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start is after end with positive step
        view = A.View(array, M.TupleExpression(N(0), N(-1), N(1)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start is before end with negative step
        view = A.View(array, M.TupleExpression(N(0), N(0), N(4))) 
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start and end aren't equal and there's a step of 0
        view = A.View(array, M.TupleExpression(N(-4), N(1), N(2)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start is before beginning of array
        view = A.View(array, M.TupleExpression(N(-2), N(1), N(2)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # end is after end of array
        view = A.View(array, M.TupleExpression(M.Const(V.Null()), N(1), N(-4)))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # end is before beginning of array
        view = A.View(array, M.TupleExpression(N(2), N(1), M.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # beginning is after end of array
           
    def TestArrayComprehension(self):
        
       # 1-d array
       counting1d = A.NewArray(M.NameLookUp("i"), M.TupleExpression(N(0), N(0), N(1), N(10), M.Const(V.String("i"))), comprehension=True)
       predictedArr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
       np.testing.assert_array_almost_equal(counting1d.Evaluate({}).array, predictedArr)
        
       # 2-d array, explicitly defined dimensions
       counting2d = A.NewArray(M.Plus(M.Times(M.NameLookUp("i"), N(3)), M.NameLookUp("j")),
                               M.TupleExpression(N(0), N(1), N(1), N(3), M.Const(V.String("i"))),
                               M.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))), 
                               comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate({}).array, predictedArr)
 
       # 2-d array, order of variable definitions opposite of previous test
       counting2d = A.NewArray(M.Plus(M.Times(M.NameLookUp("i"), N(3)), M.NameLookUp("j")),
                                M.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))),
                                M.TupleExpression(N(0), N(1), N(1), N(3), M.Const(V.String("i"))), 
                                comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate({}).array, predictedArr)
        
       # 2-d array with implicitly defined dimension assigned after explicit
       counting2d = A.NewArray(M.Plus(M.Times(M.NameLookUp("i"), N(3)), M.NameLookUp("j")),
                             M.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))),
                             M.TupleExpression(N(1), N(1), N(3), M.Const(V.String("i"))), 
                              comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate({}).array, predictedArr)
        
       # 2-d array with implicitly defined dimension assigned before explicit
       counting2d = A.NewArray(M.Plus(M.Times(M.NameLookUp("i"), N(3)), M.NameLookUp("j")),
                             M.TupleExpression(N(0), N(1), N(3), M.Const(V.String("j"))),
                             M.TupleExpression(N(0), N(1), N(1), N(3), M.Const(V.String("i"))), 
                              comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate({}).array, predictedArr)
        
       # comprehension using arrays in generator expression with one variable
       blocks = A.NewArray(A.NewArray(A.NewArray(M.Plus(N(-10), M.NameLookUp("j")), 
                                                 M.NameLookUp("j")), 
                                      A.NewArray(M.Plus(N(10), M.NameLookUp("j")), 
                                                 M.Plus(N(20), M.NameLookUp("j")))),
                           M.TupleExpression(N(1), N(0), N(1), N(2), M.Const(V.String("j"))),
                           comprehension=True)
       predictedArr = np.array([ [[-10,0], [-9,1]] , [[10,20], [11,21]] ])
       np.testing.assert_array_almost_equal(blocks.Evaluate(E.Environment()).array, predictedArr)
        
       # two gaps between instead of one
       blocks = A.NewArray(A.NewArray(A.NewArray(M.Plus(N(-10), M.NameLookUp("j")), 
                                                 M.NameLookUp("j")), 
                                      A.NewArray(M.Plus(N(10), M.NameLookUp("j")), 
                                                 M.Plus(N(20), M.NameLookUp("j")))),
                           M.TupleExpression(N(2), N(0), N(1), N(2), M.Const(V.String("j"))),
                           comprehension=True)
       predictedArr = np.array([[[-10, -9], [0, 1]], [[10, 11], [20, 21]]])
       np.testing.assert_array_almost_equal(blocks.Evaluate(E.Environment()).array, predictedArr)
        
        # comprehension using arrays in generator expression with two variables
       blocks = A.NewArray(A.NewArray(A.NewArray(M.Plus(N(-10), M.NameLookUp("j")), 
                                              M.NameLookUp("j")), 
                                   A.NewArray(M.Plus(N(10), M.NameLookUp("i")), 
                                              M.Plus(N(20), M.NameLookUp("i")))),
                        M.TupleExpression(N(0), N(1), N(2), M.Const(V.String("j"))),
                        M.TupleExpression(N(0), N(1), N(2), M.Const(V.String("i"))),
                        comprehension=True)
       predictedArr = np.array([[ [[-10, 0],[10, 20]] ,[[-10,0],[11,21]] ],[[ [-9, 1],[10, 20] ], [[-9, 1], [11, 21]]]])
       np.testing.assert_array_almost_equal(blocks.Evaluate(E.Environment()).array, predictedArr)
        
    def TestViewProtocolErrors(self):
       # creates an empty array because the start is greater than the end
       fail = A.NewArray(M.NameLookUp("i"), M.TupleExpression(N(0), N(10), N(1), N(0), M.Const(V.String("i"))), comprehension=True)
       self.assertRaises(ProtocolError, fail.Evaluate, {})
    
       # creates an empty array because the step is negative when it should be positive
       fail = A.NewArray(M.NameLookUp("i"), M.TupleExpression(N(0), N(0), N(-1), N(10), M.Const(V.String("i"))), comprehension=True)
       self.assertRaises(ProtocolError, fail.Evaluate, {})
        
       blocks = A.NewArray(A.NewArray(A.NewArray(M.NameLookUp("j")), 
                                      A.NewArray(M.NameLookUp("j"))),
                                      M.TupleExpression(N(3), N(0), N(1), N(2), M.Const(V.String("j"))),
                                      comprehension=True)
       self.assertRaises(ProtocolError, blocks.Evaluate, {})
        
    def TestSimpleMap(self):
       env = E.Environment()
       parameters = ['a', 'b', 'c']
       body = [Statement.Return(M.Plus(M.NameLookUp('a'), M.NameLookUp('b'), M.NameLookUp('c')))]
       add = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(2))
       b = A.NewArray(N(2), N(4))
       c = A.NewArray(N(3), N(7))
       result = A.Map(add, a, b, c)
       predicted = V.Array(np.array([6, 13]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
    def TestMapWithMultiDimensionalArrays(self):
       env = E.Environment()
       parameters = ['a', 'b', 'c']
       body = [Statement.Return(M.Plus(M.NameLookUp('a'), M.NameLookUp('b'), M.NameLookUp('c')))]
       add = M.LambdaExpression(parameters, body)
       a = A.NewArray(A.NewArray(N(1), N(2)), A.NewArray(N(2),N(3)))
       b = A.NewArray(A.NewArray(N(4), N(3)), A.NewArray(N(6),N(1)))
       c = A.NewArray(A.NewArray(N(2), N(2)), A.NewArray(N(8),N(0)))
       result = A.Map(add, a, b, c)
       predicted = V.Array(np.array([[7, 7], [16, 4]]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)   
          
         # more complex function and more complex array
       env = E.Environment()
       parameters = ['a', 'b', 'c']
       body = [Statement.Return(M.Times(M.Plus(M.NameLookUp('a'), M.NameLookUp('b')), M.NameLookUp('c')))]
       add_times = M.LambdaExpression(parameters, body)
       a = A.NewArray(A.NewArray(A.NewArray(N(1), N(2)), A.NewArray(N(2),N(3))), A.NewArray(A.NewArray(N(1), N(2)), A.NewArray(N(2),N(3))))
       b = A.NewArray(A.NewArray(A.NewArray(N(4), N(3)), A.NewArray(N(6),N(1))), A.NewArray(A.NewArray(N(0), N(6)), A.NewArray(N(5),N(3))))
       c = A.NewArray(A.NewArray(A.NewArray(N(2), N(2)), A.NewArray(N(8),N(0))), A.NewArray(A.NewArray(N(4), N(2)), A.NewArray(N(2),N(1))))
       result = A.Map(add_times, a, b, c)
       predicted = np.array([[[10, 10], [64, 0]], [[4, 16], [14, 6]]])
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)
       
#        env = E.Environment()
#        parameters = ['a', 'b']
#        body = [Statement.Return(M.Times(M.Plus(M.NameLookUp('a'), M.NameLookUp('b')), M.NameLookUp('c')))]
#        lambda_test = M.LambdaExpression(parameters, body)
#        a = A.NewArray(N(1), N(2))
#        b = A.NewArray(N(2), N(4))
#        c = A.NewArray(N(3), N(7))
#        result = A.Map(add_times, a, b, c)
#        predicted = np.array([[[10, 10], [64, 0]], [[4, 16], [14, 6]]])
#        np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)
       
       #map(lambda a, b=[0,1,2]: a^2*b[0] + a*b[1] + b[2], [1,2,3]) == [6,11,18]
        
    def TestUsingManyOperationsinFunction(self):
       env = E.Environment()
       parameters = ['a', 'b', 'c']
       body = [Statement.Return(M.Times(M.Plus(M.NameLookUp('a'), M.Times(M.NameLookUp('b'), M.NameLookUp('c'))), M.NameLookUp('a')))]
       add_times = M.LambdaExpression(parameters, body)
       a = A.NewArray(A.NewArray(N(1), N(2)), A.NewArray(N(2),N(3)))
       b = A.NewArray(A.NewArray(N(4), N(3)), A.NewArray(N(6),N(1)))
       c = A.NewArray(A.NewArray(N(2), N(2)), A.NewArray(N(8),N(0)))
       result = A.Map(add_times, a, b, c)
       predicted = np.array([[9, 16], [100, 9]])
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)
          
    def TestMapWithFunctionWithDefaults(self):
       env = E.Environment()
       body = [Statement.Return(M.Plus(M.NameLookUp('item'), M.NameLookUp('incr')))]
       add = M.LambdaExpression(['item', 'incr'], body, defaultParameters = [M.Const(V.DefaultParameter()), V.Simple(3)])
       item = A.NewArray(N(1), N(3), N(5))
       result = A.Map(add, item)
       predicted = np.array([4, 6, 8])
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted)
       
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
        predicted = np.array([True])
        self.assertEqual(result.value, predicted)
         
    def TestCompileMethodsForMathExpression(self):
       # Minus
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Minus(M.NameLookUp('a'), M.NameLookUp('b')))]
       minus = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(4), N(2))
       b = A.NewArray(N(2), N(1))
       result = A.Map(minus, a, b)
       predicted = V.Array(np.array([2, 1])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Divide
       body = [Statement.Return(M.Divide(M.NameLookUp('a'), M.NameLookUp('b')))]
       divide = M.LambdaExpression(parameters, body)
       result = A.Map(divide, a, b)
       predicted = V.Array(np.array([2, 2]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
 
       # Remainder
       body = [Statement.Return(M.Rem(M.NameLookUp('a'), M.NameLookUp('b')))]
       rem = M.LambdaExpression(parameters, body)
       result = A.Map(rem, a, b)
       predicted = V.Array(np.array([0, 0]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
        
       # Power
       body = [Statement.Return(M.Power(M.NameLookUp('a'), M.NameLookUp('b')))]
       power = M.LambdaExpression(parameters, body)
       result = A.Map(power, a, b)
       predicted = V.Array(np.array([16, 2]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
 
       # Root with one argument
       body = [Statement.Return(M.Root(M.NameLookUp('a')))]
       root = M.LambdaExpression('a', body)
       result = A.Map(root, a)
       predicted = V.Array(np.array([2, 1.41421356]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
        
       #Root with two arguments
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Root(M.NameLookUp('a'), M.NameLookUp('b')))]
       root = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(8))
       b = A.NewArray(N(3))
       result = A.Map(root, a, b)
       predicted = V.Array(np.array([2])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
 
       # Absolute value
       env = E.Environment()
       parameters = ['a']
       body = [Statement.Return(M.Abs(M.NameLookUp('a')))]
       absolute = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(4), N(-2))
       result = A.Map(absolute, a)
       predicted = V.Array(np.array([4, 2]))
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Exponential 
       env = E.Environment()
       parameters = ['a']
       body = [Statement.Return(M.Exp(M.NameLookUp('a')))]
       exponential = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(3))
       result = A.Map(exponential, a)
       predicted = V.Array(np.array([20.0855369231])) 
       #np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Natural Log
       env = E.Environment()
       body = [Statement.Return(M.Ln(M.NameLookUp('a')))]
       ln = M.LambdaExpression(parameters, body)
       result = A.Map(ln, a)
       predicted = V.Array(np.array([1.0986122886])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Log with One Argument (Log base 10)
       env = E.Environment()
       body = [Statement.Return(M.Log(M.NameLookUp('a')))]
       log = M.LambdaExpression(parameters, body)
       result = A.Map(log, a)
       predicted = V.Array(np.array([0.4771212547196])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Log with two arguments, second is log base qualifier
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Log(M.NameLookUp('a'), M.NameLookUp('b')))]
       log = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(4))
       b = A.NewArray(N(3))
       result = A.Map(log, a, b)
       predicted = V.Array(np.array([0.79248125036])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Max
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Max(M.NameLookUp('a'), M.NameLookUp('b')))]
       max = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(4), N(5), N(1))
       b = A.NewArray(N(3), N(6), N(0))
       result = A.Map(max, a, b)
       predicted = V.Array(np.array([4, 6, 1])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Floor 
       env = E.Environment()
       parameters = ['a']
       body = [Statement.Return(M.Floor(M.NameLookUp('a')))]
       floor = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(3.2), N(3.7))
       result = A.Map(floor, a)
       predicted = V.Array(np.array([3, 3])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Ceiling
       env = E.Environment()
       parameters = ['a']
       body = [Statement.Return(M.Ceiling(M.NameLookUp('a')))]
       ceiling = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(3.2), N(3.7))
       result = A.Map(ceiling, a)
       predicted = V.Array(np.array([4, 4])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
    
        
    def TestCompileMethodsForLogicalExpressions(self):
       # And
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.And(M.NameLookUp('a'), M.NameLookUp('b')))]
       test_and = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(1))
       b = A.NewArray(N(0), N(1))
       result = A.Map(test_and, a, b)
       predicted = V.Array(np.array([False, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Or
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Or(M.NameLookUp('a'), M.NameLookUp('b')))]
       test_or = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(1), N(0))
       b = A.NewArray(N(0), N(1), N(0))
       result = A.Map(test_or, a, b)
       predicted = V.Array(np.array([True, True, False])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Xor
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Xor(M.NameLookUp('a'), M.NameLookUp('b')))]
       test_xor = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(1), N(0))
       b = A.NewArray(N(0), N(1), N(0))
       result = A.Map(test_xor, a, b)
       predicted = V.Array(np.array([True, False, False])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
        
       # Not
       env = E.Environment()
       parameters = ['a']
       body = [Statement.Return(M.Not(M.NameLookUp('a')))]
       test_not = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0))
       result = A.Map(test_not, a)
       predicted = V.Array(np.array([False, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)
  
        # greater than
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Gt(M.NameLookUp('a'), M.NameLookUp('b')))]
       test_greater = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_greater, a, b)
       predicted = V.Array(np.array([True, False, False])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)     
 
        # less than
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Lt(M.NameLookUp('a'), M.NameLookUp('b')))]
       test_less = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_less, a, b)
       predicted = V.Array(np.array([False, False, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)   
  
        # greater than equal to
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Geq(M.NameLookUp('a'), M.NameLookUp('b')))]
       test_greater_eq = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_greater_eq, a, b)
       predicted = V.Array(np.array([True, True, False])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)  
 
        # less than equal to
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Leq(M.NameLookUp('a'), M.NameLookUp('b')))]
       test_less_eq = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_less_eq, a, b)
       predicted = V.Array(np.array([False, True, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array) 
 
        # not equal
       env = E.Environment()
       parameters = ['a', 'b']
       body = [Statement.Return(M.Neq(M.NameLookUp('a'), M.NameLookUp('b')))]
       test_not_eq = M.LambdaExpression(parameters, body)
       a = A.NewArray(N(1), N(0), N(0))
       b = A.NewArray(N(0), N(0), N(1))
       result = A.Map(test_not_eq, a, b)
       predicted = V.Array(np.array([True, False, True])) 
       np.testing.assert_array_almost_equal(result.Evaluate(env).array, predicted.array)   
       
       
         
         