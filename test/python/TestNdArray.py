"""Copyright (c) 2005-2016, University of Oxford.
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

import numpy as np
import unittest

class TestNdArray(unittest.TestCase):
    """Test functionality of numpy arrays. No protocol arrays tested here."""
    def Test0dArray(self):
        array = np.array(0)
        self.assertEqual(array.ndim, 0) # number of dimensions is 0
        self.assertEqual(array.size, 1) # number of elements is 1
        self.assertEqual(len(array.shape), 0) #size of shape is 0
        array[()] = 1 # note the different syntax for indexing a 0-d array
        self.assertEqual(array[()], 1) # value of array is one
        
    def TestBasicFunctionality(self): 
        array = np.arange(4) # assign array to equal consecutive integers
        self.assertEqual(array.ndim, 1) # number of dimensions is 1
        self.assertEqual(array.size, 4) # number of elements is 4
        self.assertEqual(array[1], 1) # can reference from front
        self.assertEqual(array[-2], 2) # can reference from back
        verify = 0
        for value in array:
            self.assertEqual(array[value], verify)
            verify += 1
        array2d = np.array([[0, 1], [0, 2], [1,1]]) # can make 2-d array
        self.assertEqual(array2d.ndim, 2) # number of dimensions is 2
        self.assertEqual(len(array2d), 3) # count number of subarrays in 2-d array
        self.assertEqual(array2d.size, 6) # count number of total elements in 2-d array
        array2dCopy = array2d.copy() # can create true copy of array, altering array2dCopy does not alter array2d
        array2dCopy[2, 1] = 3
        self.assertNotEqual(array2d[2, 1], array2dCopy[2, 1])
        array2dView = array2d # creates view of array, altering array2dView alters array2d 
        array2dView[2, 1] = 3
        self.assertEqual(array2d[2, 1], array2dView[2, 1])
        array2d = array2d.reshape(1,6)
        self.assertEqual(array2d.shape, (1,6)) # you can reshape the arrays
        
    def TestMoreIterationAndViews(self):
        array = np.ones((3, 4, 2, 7)) # create four dimensional array of ones
        self.assertEqual(array.ndim, 4) # assert that there are four dimensions
        self.assertEqual(array.size, 3*4*2*7) # number of elements
        array = np.arange(168).reshape(3, 4, 2, 7)
        view = array[0::2, :0:-2, -1, 1::-1] # can slice stepping forward, backward, picking a position...etc
        self.assertEqual(view.ndim, 3) # number of dimensions in this view is 3
        np.testing.assert_array_almost_equal(view, np.array([[[50,49], [22,21]], [[162,161], [134,133]]]))
        self.assertEqual(view.shape, (2,2,2))
        self.assertEqual(view.size, 2*2*2)    
        view[0, 0, 0] = 1 # change the first element of the view
        self.assertEqual(view[0, 0, 0], array[0, 3, 1, 1]) # changing view changed the first element of the array
        copy = array.copy()
        self.assertEqual(copy[0, 0, 0, 0], array[0, 0, 0, 0]) # copy is exact copy, first elements are equal
        copy[0, 0, 0, 0] = 10 # change first element of the copy
        self.assertNotEqual(copy[0, 0, 0, 0], array[0, 0, 0, 0]) # changing the copy doesn't affect the original
        view = array[0, 0, 0, 1]
        self.assertEqual(view.ndim, 0) # you can take a 0-d view of an array and treat it as a value
        self.assertEqual(view.size, 1) # one element in the 0-d view
        self.assertEqual(view, 1) # the 0-d array is equal to the value of the number it contains
