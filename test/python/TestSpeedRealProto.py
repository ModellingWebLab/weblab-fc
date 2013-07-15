
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
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGEnv.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import CompactSyntaxParser as CSP
CSP.ImportPythonImplementation()
csp = CSP.CompactSyntaxParser

import Environment as Env
import Values as V
import ArrayExpressions as A
import Expressions as E
import MathExpressions as M
import numpy as np
import os
import Protocol

def N(v):
    return M.Const(V.Simple(v))

class TestSpeedRealProto(unittest.TestCase):
    def TestS1S2(self):
        # Parse the protocol into a sequence of post-processing statements
        proto_file = 'projects/FunctionalCuration/test/protocols/compact/S1S2.txt'
        proto = Protocol.Protocol(proto_file)
#         parser = csp()
#         CSP.source_file = proto_file
#         generator = parser._Try(csp.protocol.parseFile, proto_file, parseAll=True)[0]
#         self.assertIsInstance(generator, CSP.Actions.Protocol)
#         statements = generator.expr()[0]
        # Load the raw simulation data from file
#         env = Env.Environment()
        data_folder = 'projects/FunctionalCuration/test/data/TestSpeedRealProto'
        membrane_voltage = self.Load2d(os.path.join(data_folder, 'outputs_membrane_voltage.csv'))
        time_1d = self.Load(os.path.join(data_folder, 'outputs_time_1d.csv'))
        time_2d = A.NewArray(M.Const(time_1d),
                             E.TupleExpression(N(0), N(0), N(1), N(91), M.Const(V.String('_'))),
                             comprehension=True).Evaluate(proto.env)
        proto.env.DefineName('sim:time', time_2d)
        proto.env.DefineName('sim:membrane_voltage', membrane_voltage)
        # Run the protocol
#         env.ExecuteStatements(statements)
        proto.Run()

        for var in ['raw_APD90', 'raw_DI']:
            expected = self.Load2d(os.path.join(data_folder, 'outputs_' + var + '.csv'))
            actual = proto.env.LookUp(var)
            np.testing.assert_allclose(actual.array, expected.array, rtol=0.01)
        for var in ['max_S1S2_slope']:
            expected = self.Load(os.path.join(data_folder, 'outputs_' + var + '.csv'))
            actual = proto.env.LookUp(var)
            np.testing.assert_allclose(actual.array, expected.array, rtol=0.01)

    def Load2d(self, filePath):
        array = np.loadtxt(filePath, dtype=float, delimiter=',', unpack=True) # unpack transposes the array
        if array.ndim == 1:
            array = array[:, np.newaxis]
        return V.Array(array)

    def Load(self, filePath):
        f = open(filePath, 'r')
        f.readline() # Strip comment line
        dims = map(int, f.readline().split(','))[1:]
        array = np.loadtxt(f, dtype=float)
        f.close()
        return V.Array(array.reshape(tuple(dims)))
