
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
        #self.assertRaises(ProtocolError, A.View, one) # first argument must be an array
        
        view = A.View(arr, M.TupleExpression(one, M.Const(V.Null()))) # two parameters: beginning and end, null represents end of original array
        predictedArr = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
       
        view = A.View(arr, M.TupleExpression(zero, two, four)) # three parameters: beginning, step, end
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
        
        array = A.NewArray(A.NewArray(A.NewArray(minusTwo, minusOne, zero), A.NewArray(one, two, three)), A.NewArray(A.NewArray(four, five, six), A.NewArray(one, two, three))) # 3-d array
        view = A.View(array, M.TupleExpression(zero, M.Const(V.Null()), two), M.TupleExpression(two, minusOne, zero), M.TupleExpression(zero, two)) 
        predictedArr = np.array([[[1, 2]], [[1, 2]]])
        self.assertEqual(view.Evaluate({}).array.ndim, 3) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # use four parameters in the tuples to specify dimension explicitly
        view = A.View(array, M.TupleExpression(zero, zero, M.Const(V.Null()), two), M.TupleExpression(one, two, minusOne, zero), M.TupleExpression(two, zero, M.Const(V.Null()), two)) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # use four parameters in the tuples to specify dimension with a mix of implicit and explicit declarations
        view = A.View(array, M.TupleExpression(zero, M.Const(V.Null()), two), M.TupleExpression(one, two, minusOne, zero), M.TupleExpression(zero, M.Const(V.Null()), two)) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # test leaving some parameters out so they fall to default
        view = A.View(array, M.TupleExpression(one, two, minusOne, zero), M.TupleExpression(zero, M.Const(V.Null()), two)) 
        view2 = A.View(array, M.TupleExpression(M.Const(V.Null()), M.Const(V.Null()), one, M.Const(V.Null())), M.TupleExpression(one, two, minusOne, zero), M.TupleExpression(zero, M.Const(V.Null()), two)) 
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, predictedArr)
        # checks to make sure the "default default" is equivalent to a tuple of (Null, Null, 1, Null)
        np.testing.assert_array_almost_equal(view.Evaluate({}).array, view2.Evaluate({}).array)
        
        