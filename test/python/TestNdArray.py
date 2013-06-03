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
import numpy as np

class TestNdArray(unittest.TestCase):
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
        view = array[::, ::-2, -1, ::] # can slice stepping forward, backward, picking a position...etc
        self.assertEqual(view.ndim, 3) # number of dimensions in this view is 3
        #np.testing.assert_array_almost_equal(view, np.array([[0], [2], [2], [4]]))
        #self.assertEqual(view.shape, (2,2,2,4))
        #self.assertEqual(view.size, 2*2*2)
        
        
        
        
        
        
        
"""TestMoreIterationAndViews() throw (Exception)
    {
        Extents extents = boost::assign::list_of(3)(4)(2)(7);

        Array arr(extents);
        TS_ASSERT_EQUALS(arr.GetNumDimensions(), 4u);
        TS_ASSERT_EQUALS(arr.GetNumElements(), 3u*4u*2u*7u);

        // Fill in the array using iterators (and note that it++ works too)
        double value = 0.0;
        for (Iterator it = arr.Begin(); it != arr.End(); it++)
        {
            *it = (value--)/2.0;
        }

        // Check we can take a view missing 'internal' dimensions
        RangeSpec view_indices = boost::assign::list_of(R(0, 2, R::END))  // First & last elements from dim 0
                                                       (R(R::END, -2, 1)) // Dim 1 reversed step 2 (elts 3, 1)
                                                       (R(-1))            // Last element of dim 2
                                                       (R(2, -1, 0));     // First 2 elements of dim 3 reversed
        Array view = arr[view_indices];
        TS_ASSERT_EQUALS(view.GetNumDimensions(), 3u);
        TS_ASSERT_EQUALS(view.GetNumElements(), 2u*2u*2u);
        TS_ASSERT_EQUALS(view.GetShape()[0], 2u);
        TS_ASSERT_EQUALS(view.GetShape()[1], 2u);
        TS_ASSERT_EQUALS(view.GetShape()[2], 2u);
        // Original array multipliers are: 56, 14, 7, 1
        // So offsets into it are 50,49,22,21, 162,161,134,133
        std::vector<double> expected = boost::assign::list_of(-25.0)(-49.0/2)(-11.0)(-21.0/2)
                                                             (-81.0)(-161.0/2)(-67.0)(-133.0/2);
        unsigned i=0;
        for (ConstIterator it=view.Begin(); it != view.End(); ++it)
        {
            TS_ASSERT_EQUALS(*it, expected[i]);
            // Views should ideally alias the original data, not copy it
            TS_ASSERT_EQUALS(it, arr.Begin() + (ptrdiff_t)(-2*expected[i]));
            i++;
        }

        // Check that copying a view gives us a fresh array
        Array view_copy = view.Copy();
        TS_ASSERT_DIFFERS(view_copy.Begin(), view.Begin());
        for (ConstIterator it=view.Begin(), copy_it=view_copy.Begin(); it != view.End(); ++it, ++copy_it)
        {
            TS_ASSERT_EQUALS(*it, *copy_it);
        }

        // And check that a 0d view works
        view_indices = boost::assign::list_of(R(1))(R(2))(R(1))(R(3));
        view = arr[view_indices];
        TS_ASSERT_EQUALS(view.GetNumDimensions(), 0u);
        TS_ASSERT_EQUALS(view.GetNumElements(), 1u);
        TS_ASSERT_EQUALS(*view.Begin(), -0.5*(56+28+7+3));
        TS_ASSERT_EQUALS(++view.Begin(), view.End());
    }
};"""