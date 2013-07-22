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

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import Values as V
import Environment as Env
from Model import TestOdeModel
from ErrorHandling import ProtocolError
import Ranges
import Simulations
import numpy as np

def N(number):
    return M.Const(V.Simple(number))

class TestModelSimulation(unittest.TestCase):
    def TestSimpleODE(self):
#         # using range made in python
#         a = 5
#         model = TestOdeModel(a)
#         for t in range(10):
#             if t > 0:
#                 model.Simulate(t)
#             self.assertEqual(model.GetOutputs().LookUp('a').value, a)
#             self.assertAlmostEqual(model.GetOutputs().LookUp('y').value, t*a)
#         
#         # using UniformRange from Ranges class
#         a = 5
#         model = TestOdeModel(a)
#         range_ = Ranges.UniformRange(V.Simple(0), V.Simple(10), V.Simple(1))
#         time_sim = Simulations.Timecourse(range_)
#         time_sim.SetModel(model)
#         results = time_sim.Run()
#         np.testing.assert_array_almost_equal(results.LookUp('a').array, np.array([5]*11))
#         np.testing.assert_array_almost_equal(results.LookUp('y').array, np.array([t*5 for t in range(11)]))   
        
        # using VectorRange from Ranges class   
        a = 5
        model = TestOdeModel(a)
        range_ = Ranges.VectorRange(V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        # set model method
        time_sim = Simulations.Timecourse(range_)
        time_sim.SetModel(model)
#         results = time_sim.Run()
#         np.testing.assert_array_almost_equal(results.LookUp('a').array, np.array([5]*11))
#         np.testing.assert_array_almost_equal(results.LookUp('y').array, np.array([t*5 for t in range(11)]))   
        
        # test nested simulations
        a = 5
        model = TestOdeModel(a)
        range_ = Ranges.VectorRange(V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(50, 101, 5), np.arange(100, 151, 5), np.arange(150, 201, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)
        # results would be like [[0, 5, 10] [10, 15..
            