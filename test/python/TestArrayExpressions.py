
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
import numpy as np
import MathExpressions as M

from ErrorHandling import ProtocolError

def N(number):
    return M.Const(V.Simple(number))

class TestArrayExpressions(unittest.TestCase):
    def TestNewArray(self):
        one = M.Const(V.Simple(1))
        two = M.Const(V.Simple(2))
        three = M.Const(V.Simple(3))
        arr = A.NewArray(one, two, three) # simple one-dimensional array
        predictedArr = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(arr.Evaluate({}).array, predictedArr)
        arr = A.NewArray(A.NewArray(one, two, three), A.NewArray(three, three, two))
        predictedArr = np.array([[1, 2, 3], [3, 3, 2]])
        np.testing.assert_array_almost_equal(arr.Evaluate({}).array, predictedArr)
        
    def TestViews(self):
        minusTwo = M.Const(V.Simple(-2))
        minusOne = M.Const(V.Simple(-1))
        zero = M.Const(V.Simple(0))
        one = M.Const(V.Simple(1))
        two = M.Const(V.Simple(2))
        three = M.Const(V.Simple(3))
        four = M.Const(V.Simple(4))
        five = M.Const(V.Simple(5))
        six = M.Const(V.Simple(6))
        arr = A.NewArray(one, two, three, four)
        
        view = A.View(arr, M.TupleExpression(one, M.Const(V.Null()))) # two parameters: beginning and end, null represents end of original array
        predictedArr = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
       
        view = A.View(arr, M.TupleExpression(zero, two, three)) # three parameters: beginning, step, end
        predictedArr = np.array([1, 3])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
       
        view = A.View(arr, M.TupleExpression(three, minusOne, zero)) # negative step
        predictedArr = np.array([4, 3, 2])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        view = A.View(arr, M.TupleExpression(one, zero, one)) # 0 as step
        predicted = 2
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predicted)
        view = A.View(arr, M.TupleExpression(one)) # same as immediately above, but only one number passed instead of a step of 0
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predicted)
        
        array = A.NewArray(A.NewArray(minusTwo, minusOne, zero), A.NewArray(one, two, three), A.NewArray(four, five, six)) # testing many aspects of views of a 2-d array
        view = A.View(array, M.TupleExpression(zero, M.Const(V.Null()), two), M.TupleExpression(two, minusOne, zero)) # can slice stepping forward, backward, picking a position...etc
        predictedArr = np.array([[0, -1], [3, 2]])
        self.assertEqual(view.Evaluate({}).array.ndim, 2) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        
        array = A.NewArray(A.NewArray(A.NewArray(minusTwo, minusOne, zero), 
                                      A.NewArray(one, two, three)), 
                           A.NewArray(A.NewArray(four, five, six), 
                                      A.NewArray(one, two, three))) # 3-d array
        view = A.View(array, M.TupleExpression(zero, M.Const(V.Null()), M.Const(V.Null())), 
                      M.TupleExpression(M.Const(V.Null()), minusOne, zero), 
                      M.TupleExpression(zero, two)) 
        predictedArr = np.array([[[1, 2]], [[1, 2]]])
        self.assertEqual(view.Evaluate({}).array.ndim, 3) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # use four parameters in the tuples to specify dimension explicitly
        view = A.View(array, M.TupleExpression(zero, zero, M.Const(V.Null()), two), 
                      M.TupleExpression(one, two, minusOne, zero), 
                      M.TupleExpression(two, zero, M.Const(V.Null()), two)) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # use four parameters in the tuples to specify dimension with a mix of implicit and explicit declarations
        view = A.View(array, M.TupleExpression(zero, M.Const(V.Null()), M.Const(V.Null())), 
                      M.TupleExpression(one, M.Const(V.Null()), minusOne, zero), 
                      M.TupleExpression(zero, M.Const(V.Null()), two)) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # test leaving some parameters out so they fall to default
        view = A.View(array, M.TupleExpression(one, M.Const(V.Null()), minusOne, zero), 
                      M.TupleExpression(zero, M.Const(V.Null()), M.Const(V.Null()))) 
        view2 = A.View(array, M.TupleExpression(zero, M.Const(V.Null()), M.Const(V.Null())), 
                       M.TupleExpression(one, M.Const(V.Null()), minusOne, zero), 
                       M.TupleExpression(M.Const(V.Null()), M.Const(V.Null()), one, M.Const(V.Null())))
        predictedArr = np.array([[[1, 2, 3]], [[1, 2, 3]]])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # checks to make sure the "default default" is equivalent to a tuple of (Null, Null, 1, Null), also checks to make sure implicitly defined slices go to the first dimension that is not assigned explicitly
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, view2.Evaluate({}).array)
        view = A.View(array, M.TupleExpression(one, M.Const(V.Null()), minusOne, zero)) # only specified dimension is in middle
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        view = A.View(array, M.TupleExpression(one, M.Const(V.Null()), minusOne, zero), 
                      M.TupleExpression(zero, zero, M.Const(V.Null()), M.Const(V.Null())))
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr) # tests explicitly assigning dimension one before explicitly defining dimension zero
        
        # Testing protocol errors
        view = A.View(array, M.TupleExpression(zero, zero, M.Const(V.Null()), M.Const(V.Null())), 
                      M.TupleExpression(one, M.Const(V.Null()), minusOne, zero), 
                      M.TupleExpression(two, zero, M.Const(V.Null()), M.Const(V.Null())), 
                      M.TupleExpression(two, zero, M.Const(V.Null()), M.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # more tuple slices than dimensions
        view = A.View(array, M.TupleExpression(one, M.Const(V.Null()), minusOne, zero, one, four))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # too many arguments for a slice
        view = A.View(one)
        self.assertRaises(ProtocolError, view.Evaluate, {}) # first argument must be an array
        view = A.View(array, M.TupleExpression(five))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # index goes out of range
        view = A.View(array, M.TupleExpression(three, zero, M.Const(V.Null()), M.Const(V.Null())))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # attempts to assign dimension three when array only has dimensions 0, 1, and 2
        view = A.View(array, M.TupleExpression(one, one, zero))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start is after end with positive step
        view = A.View(array, M.TupleExpression(zero, minusOne, one))
        self.assertRaises(ProtocolError, view.Evaluate, {}) # start is before end with negative step
        view = A.View(array, M.TupleExpression(zero, zero, four)) 
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
       # view = M.TupleExpression(M.Const(V.Null()), N(1), M.Const(V.Null()))
       # rangespec = range(10)
       # arr1d = A.NewArray(True, view, rangespec)
       # 
       counting1d = A.NewArray(M.NameLookUp("i"), M.TupleExpression(N(0), N(0), N(1), N(10), M.Const(V.String("i"))), comprehension=True)
       predictedArr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
       np.testing.assert_array_almost_equal(counting1d.Evaluate({}).array, predictedArr)
       
       counting2d = A.NewArray(M.Plus(M.Times(M.NameLookUp("i"), N(3)), M.NameLookUp("j")),
                               M.TupleExpression(N(0), N(1), N(1), N(3), M.Const(V.String("i"))),
                               M.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))), 
                               comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate({}).array, predictedArr)

       counting2d = A.NewArray(M.Plus(M.Times(M.NameLookUp("i"), N(3)), M.NameLookUp("j")),
                                M.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))),
                                M.TupleExpression(N(0), N(1), N(1), N(3), M.Const(V.String("i"))), 
                                comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate({}).array, predictedArr)
   
       counting2d = A.NewArray(M.Plus(M.Times(M.NameLookUp("i"), N(3)), M.NameLookUp("j")),
                             M.TupleExpression(N(1), N(0), N(1), N(3), M.Const(V.String("j"))),
                             M.TupleExpression(N(1), N(1), N(3), M.Const(V.String("i"))), 
                              comprehension=True)
       predictedArr = np.array([[3, 4, 5],[6, 7, 8]])
       np.testing.assert_array_almost_equal(counting2d.Evaluate({}).array, predictedArr)
"""
34       
35        # Testing array comprehensions (and more accessors)
36       
37        counting1d = [ i for 0$i in 0:10 ]
38        assert counting1d.NUM_DIMS == 1
39        assert counting1d.SHAPE[0] == 10
40        assert ArrayEq(counting1d, [0,1,2,3,4,5,6,7,8,9])
41        assert ArrayEq(counting1d, [ i for i in 0:10 ]) # Implicit dimension number
42       
43        counting2d = [ i*3 + j for 0$i in 1:3 for 1$j in 0:3 ]
44        assert counting2d.NUM_DIMS == 2
45        assert counting2d.NUM_ELEMENTS == 6
46        assert counting2d.SHAPE[0] == 2
47        assert ArrayEq(counting2d, [[3, 4, 5], [6, 7, 8]])
48       
49        counting2d_alt = [ i*3 + j for 1$j in 0:3 for 0$i in 1:3 ]
50        assert ArrayEq(counting2d, counting2d_alt)
51       
52        blocks = [ [[-10+j,j],[10+j,20+j]] for 1$j in 0:2 ]
53        assert blocks.NUM_DIMS == 3
54        assert blocks.NUM_ELEMENTS == 8
55        assert blocks.SHAPE[0] == 2
56        assert blocks.SHAPE[1] == 2
57        assert ArrayEq(blocks, [ [[-10,0], [-9,1]] , [[10,20], [11,21]] ])
58       
59        # Test reversing an array
60        assert ArrayEq(input[:-1:], [ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 ])
61       
62        # Test more complex negative step views
63        # Note this is Python semantics of half-open range: the begin element is included, the end element not.
64        assert ArrayEq(input[:-1:-3], [10, 9])
65        assert ArrayEq(input[3:-1:], [4, 3, 2, 1])
66        assert ArrayEq(input[4:-1:2], [5, 4])
67        assert ArrayEq(input[2:-2:0], [3])
68        assert ArrayEq(input[2:-2:-11], [3, 1])
69        assert ArrayEq(input[-1:-3:], [10, 7, 4, 1])"""
        
        