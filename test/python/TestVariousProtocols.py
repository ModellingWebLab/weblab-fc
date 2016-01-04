
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
try:
    import unittest2 as unittest
except ImportError:
    import unittest

import fc
import fc.simulations.model as Model
import fc.language.expressions as E

class TestVariousProtocols(unittest.TestCase):
    """Test that various test protocols are executed correctly."""
    def TestNestedProtocols(self):
        proto_file = 'projects/FunctionalCuration/test/protocols/test_nested_protocol.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestNestedProtocols')
        proto.SetModel('projects/FunctionalCuration/cellml/luo_rudy_1991.cellml')
        proto.Run()
        self.assertNotIn('always_missing', proto.outputEnv)
        self.assertNotIn('first_missing', proto.outputEnv)
        self.assertNotIn('some_missing', proto.outputEnv)
        self.assertNotIn('first_present', proto.outputEnv)
 
    def TestAnnotatingWithOtherOntologies(self):
        proto_file = 'projects/FunctionalCuration/test/protocols/test_other_ontologies.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestAnnotatingWithOtherOntologies')
        proto.SetModel('projects/FunctionalCuration/test/data/test_lr91.cellml')
        proto.Run()
 
    def TestParallelNestedTxt(self):
        # NB: In the current Python implementation this doesn't actually parallelise!
        proto_file = 'projects/FunctionalCuration/test/protocols/test_parallel_nested.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestParallelNestedTxt')
        proto.SetModel(Model.TestOdeModel(1))
        proto.Run()
        proto.model.ResetState()
        proto.Run()
 
    def TestSimEnvTxt(self):
        proto_file = 'projects/FunctionalCuration/test/protocols/test_sim_environments.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestSimEnv')
        proto.SetModel(Model.TestOdeModel(1))
        proto.Run()

    def TestWhileLoopTxt(self):
        proto_file = 'projects/FunctionalCuration/test/protocols/test_while_loop.txt'
        proto = fc.Protocol(proto_file)
        proto.SetOutputFolder('TestVariousProtocols_TestWhileLoopTxt')
        proto.SetModel(Model.TestOdeModel(1))
        proto.SetInput('num_iters', E.N(10))
        proto.Run()
