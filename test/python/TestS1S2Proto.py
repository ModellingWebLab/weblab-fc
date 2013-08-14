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

import fc
import fc.utility.test_support as TestSupport
import fc.simulations.model as Model


class TestS1S2Proto(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""
    def TestS1S2(self):
        proto = fc.Protocol('projects/FunctionalCuration/test/protocols/compact/S1S2.txt')
        proto.SetOutputFolder('Py_TestS1S2Proto')
        proto.SetModel('projects/FunctionalCuration/cellml/courtemanche_ramirez_nattel_1998.cellml')
        proto.model.SetSolver(Model.PySundialsSolver())
        proto.Run()
        data_folder = 'projects/FunctionalCuration/test/data/TestSpeedRealProto/S1S2'
        TestSupport.CheckResults(proto, {'raw_APD90': 2, 'raw_DI': 2, 'max_S1S2_slope': 1}, data_folder)
