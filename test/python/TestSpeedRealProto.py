
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

def N(v):
    return M.Const(V.Simple(v))

class TestSpeedRealProto(unittest.TestCase):
    def TestIndex(self):
        idxs = np.array([[  0.00000000e+00,   1.10000000e+01],
       [  0.00000000e+00,   2.01100000e+03],
       [  1.00000000e+00,   1.10000000e+01],
       [  1.00000000e+00,   1.96100000e+03],
       [  2.00000000e+00,   1.10000000e+01],
       [  2.00000000e+00,   1.91100000e+03],
       [  3.00000000e+00,   1.10000000e+01],
       [  3.00000000e+00,   1.86100000e+03],
       [  4.00000000e+00,   1.10000000e+01],
       [  4.00000000e+00,   1.81100000e+03],
       [  5.00000000e+00,   1.10000000e+01],
       [  5.00000000e+00,   1.76100000e+03],
       [  6.00000000e+00,   1.10000000e+01],
       [  6.00000000e+00,   1.71100000e+03],
       [  7.00000000e+00,   1.10000000e+01],
       [  7.00000000e+00,   1.66100000e+03],
       [  8.00000000e+00,   1.10000000e+01],
       [  8.00000000e+00,   1.61100000e+03],
       [  9.00000000e+00,   1.10000000e+01],
       [  9.00000000e+00,   1.56100000e+03],
       [  1.00000000e+01,   1.10000000e+01],
       [  1.00000000e+01,   1.51100000e+03],
       [  1.10000000e+01,   1.10000000e+01],
       [  1.10000000e+01,   1.46100000e+03],
       [  1.20000000e+01,   1.10000000e+01],
       [  1.20000000e+01,   1.41100000e+03],
       [  1.30000000e+01,   1.10000000e+01],
       [  1.30000000e+01,   1.36100000e+03],
       [  1.40000000e+01,   1.10000000e+01],
       [  1.40000000e+01,   1.31100000e+03],
       [  1.50000000e+01,   1.10000000e+01],
       [  1.50000000e+01,   1.26100000e+03],
       [  1.60000000e+01,   1.10000000e+01],
       [  1.60000000e+01,   1.21100000e+03],
       [  1.70000000e+01,   1.10000000e+01],
       [  1.70000000e+01,   1.16100000e+03],
       [  1.80000000e+01,   1.10000000e+01],
       [  1.80000000e+01,   1.11100000e+03],
       [  1.90000000e+01,   1.10000000e+01],
       [  1.90000000e+01,   1.06100000e+03],
       [  2.00000000e+01,   1.10000000e+01],
       [  2.00000000e+01,   1.01100000e+03],
       [  2.10000000e+01,   1.10000000e+01],
       [  2.10000000e+01,   9.86000000e+02],
       [  2.20000000e+01,   1.10000000e+01],
       [  2.20000000e+01,   9.61000000e+02],
       [  2.30000000e+01,   1.10000000e+01],
       [  2.30000000e+01,   9.36000000e+02],
       [  2.40000000e+01,   1.10000000e+01],
       [  2.40000000e+01,   9.11000000e+02],
       [  2.50000000e+01,   1.10000000e+01],
       [  2.50000000e+01,   8.86000000e+02],
       [  2.60000000e+01,   1.10000000e+01],
       [  2.60000000e+01,   8.61000000e+02],
       [  2.70000000e+01,   1.10000000e+01],
       [  2.70000000e+01,   8.36000000e+02],
       [  2.80000000e+01,   1.10000000e+01],
       [  2.80000000e+01,   8.11000000e+02],
       [  2.90000000e+01,   1.10000000e+01],
       [  2.90000000e+01,   7.86000000e+02],
       [  3.00000000e+01,   1.10000000e+01],
       [  3.00000000e+01,   7.61000000e+02],
       [  3.10000000e+01,   1.10000000e+01],
       [  3.10000000e+01,   7.36000000e+02],
       [  3.20000000e+01,   1.10000000e+01],
       [  3.20000000e+01,   7.11000000e+02],
       [  3.30000000e+01,   1.10000000e+01],
       [  3.30000000e+01,   6.91000000e+02],
       [  3.40000000e+01,   1.10000000e+01],
       [  3.40000000e+01,   6.71000000e+02],
       [  3.50000000e+01,   1.10000000e+01],
       [  3.50000000e+01,   6.51000000e+02],
       [  3.60000000e+01,   1.10000000e+01],
       [  3.60000000e+01,   6.31000000e+02],
       [  3.70000000e+01,   1.10000000e+01],
       [  3.70000000e+01,   6.11000000e+02],
       [  3.80000000e+01,   1.10000000e+01],
       [  3.80000000e+01,   5.91000000e+02],
       [  3.90000000e+01,   1.10000000e+01],
       [  3.90000000e+01,   5.71000000e+02],
       [  4.00000000e+01,   1.10000000e+01],
       [  4.00000000e+01,   5.51000000e+02],
       [  4.10000000e+01,   1.10000000e+01],
       [  4.10000000e+01,   5.31000000e+02],
       [  4.20000000e+01,   1.10000000e+01],
       [  4.20000000e+01,   5.11000000e+02],
       [  4.30000000e+01,   1.10000000e+01],
       [  4.30000000e+01,   5.01000000e+02],
       [  4.40000000e+01,   1.10000000e+01],
       [  4.40000000e+01,   4.91000000e+02],
       [  4.50000000e+01,   1.10000000e+01],
       [  4.50000000e+01,   4.81000000e+02],
       [  4.60000000e+01,   1.10000000e+01],
       [  4.60000000e+01,   4.71000000e+02],
       [  4.70000000e+01,   1.10000000e+01],
       [  4.70000000e+01,   4.61000000e+02],
       [  4.80000000e+01,   1.10000000e+01],
       [  4.80000000e+01,   4.51000000e+02],
       [  4.90000000e+01,   1.10000000e+01],
       [  4.90000000e+01,   4.41000000e+02],
       [  5.00000000e+01,   1.10000000e+01],
       [  5.00000000e+01,   4.31000000e+02],
       [  5.10000000e+01,   1.10000000e+01],
       [  5.10000000e+01,   4.21000000e+02],
       [  5.20000000e+01,   1.10000000e+01],
       [  5.20000000e+01,   4.11000000e+02],
       [  5.30000000e+01,   1.10000000e+01],
       [  5.30000000e+01,   4.01000000e+02],
       [  5.40000000e+01,   1.10000000e+01],
       [  5.40000000e+01,   3.91000000e+02],
       [  5.50000000e+01,   1.10000000e+01],
       [  5.50000000e+01,   3.81000000e+02],
       [  5.60000000e+01,   1.10000000e+01],
       [  5.60000000e+01,   3.71000000e+02],
       [  5.70000000e+01,   1.10000000e+01],
       [  5.70000000e+01,   3.61000000e+02],
       [  5.80000000e+01,   1.10000000e+01],
       [  5.80000000e+01,   3.51000000e+02],
       [  5.90000000e+01,   1.10000000e+01],
       [  5.90000000e+01,   3.41000000e+02],
       [  6.00000000e+01,   1.10000000e+01],
       [  6.00000000e+01,   3.31000000e+02],
       [  6.10000000e+01,   1.10000000e+01],
       [  6.10000000e+01,   3.21000000e+02],
       [  6.20000000e+01,   1.10000000e+01],
       [  6.20000000e+01,   3.11000000e+02],
       [  6.30000000e+01,   1.10000000e+01],
       [  6.30000000e+01,   3.01000000e+02],
       [  6.40000000e+01,   1.10000000e+01],
       [  6.40000000e+01,   2.91000000e+02],
       [  6.50000000e+01,   1.10000000e+01],
       [  6.50000000e+01,   2.81000000e+02],
       [  6.60000000e+01,   1.10000000e+01],
       [  6.60000000e+01,   2.71000000e+02],
       [  6.70000000e+01,   1.10000000e+01],
       [  6.70000000e+01,   2.61000000e+02],
       [  6.80000000e+01,   1.10000000e+01],
       [  6.80000000e+01,   2.51000000e+02],
       [  6.90000000e+01,   1.10000000e+01],
       [  6.90000000e+01,   2.41000000e+02],
       [  7.00000000e+01,   1.10000000e+01],
       [  7.00000000e+01,   2.31000000e+02],
       [  7.10000000e+01,   1.10000000e+01],
       [  7.10000000e+01,   2.21000000e+02],
       [  7.20000000e+01,   1.10000000e+01],
       [  7.20000000e+01,   2.11000000e+02],
       [  7.30000000e+01,   1.10000000e+01],
       [  7.30000000e+01,   2.01000000e+02],
       [  7.40000000e+01,   1.10000000e+01],
       [  7.40000000e+01,   1.90000000e+02],
       [  7.50000000e+01,   1.10000000e+01],
       [  7.50000000e+01,   1.80000000e+02],
       [  7.60000000e+01,   1.10000000e+01],
       [  7.60000000e+01,   1.70000000e+02],
       [  7.70000000e+01,   1.10000000e+01],
       [  7.70000000e+01,   1.60000000e+02],
       [  7.80000000e+01,   1.10000000e+01],
       [  7.80000000e+01,   1.50000000e+02],
       [  7.90000000e+01,   1.10000000e+01],
       [  7.90000000e+01,   1.40000000e+02],
       [  8.00000000e+01,   1.10000000e+01],
       [  8.00000000e+01,   1.30000000e+02],
       [  8.10000000e+01,   1.10000000e+01],
       [  8.10000000e+01,   1.20000000e+02],
       [  8.20000000e+01,   1.10000000e+01],
       [  8.20000000e+01,   1.10000000e+02],
       [  8.30000000e+01,   1.10000000e+01],
       [  8.30000000e+01,   1.00000000e+02],
       [  8.40000000e+01,   1.10000000e+01],
       [  8.40000000e+01,   9.00000000e+01],
       [  8.50000000e+01,   1.10000000e+01],
       [  8.50000000e+01,   8.00000000e+01],
       [  8.60000000e+01,   1.10000000e+01],
       [  8.60000000e+01,   7.00000000e+01],
       [  8.70000000e+01,   1.10000000e+01],
       [  8.80000000e+01,   1.10000000e+01],
       [  8.90000000e+01,   1.10000000e+01],
       [  9.00000000e+01,   1.10000000e+01]])
        env = Env.Environment()
        data_folder = 'projects/FunctionalCuration/test/data/TestSpeedRealProto'
        time_1d = self.Load(os.path.join(data_folder, 'outputs_time_1d.csv'))
        time_2d = A.NewArray(M.Const(time_1d),
                             E.TupleExpression(N(0), N(0), N(1), N(91), M.Const(V.String('_'))),
                             comprehension=True).Evaluate(env)
        env.DefineName('t', time_2d)
        env.DefineName('max_upstroke_idxs', V.Array(idxs))
        expr = 't{max_upstroke_idxs, 1, pad:1=0}'
        index_parse_action = csp.expr.parseString(expr)
        expr_of_index = index_parse_action[0].expr()
        index_result = expr_of_index.Interpret(env)
        print "index result", index_result.array

    
    def TestS1S2(self):
        # Parse the protocol into a sequence of post-processing statements
        proto_file = 'projects/FunctionalCuration/test/protocols/compact/S1S2_postproc.txt'
        parser = csp()
        CSP.source_file = proto_file
        generator = parser._Try(csp.protocol.parseFile, proto_file, parseAll=True)[0]
        self.assertIsInstance(generator, CSP.Actions.Protocol)
        statements = generator.expr()[0]
        # Load the raw simulation data from file
        env = Env.Environment()
        data_folder = 'projects/FunctionalCuration/test/data/TestSpeedRealProto'
        membrane_voltage = self.Load2d(os.path.join(data_folder, 'outputs_membrane_voltage.csv'))
        time_1d = self.Load(os.path.join(data_folder, 'outputs_time_1d.csv'))
        time_2d = A.NewArray(M.Const(time_1d),
                             E.TupleExpression(N(0), N(0), N(1), N(91), M.Const(V.String('_'))),
                             comprehension=True).Evaluate(env)
        env.DefineName('sim:time', time_2d)
        env.DefineName('sim:membrane_voltage', membrane_voltage)
        # Run the protocol
        env.ExecuteStatements(statements)
    
    def Load2d(self, filePath):
        array = np.loadtxt(filePath, dtype=float, delimiter=',', unpack=True) # unpack transposes the array
        return V.Array(array)

    def Load(self, filePath):
        f = open(filePath, 'r')
        f.readline() # Strip comment line
        dims = map(int, f.readline().split(','))[1:]
        array = np.loadtxt(f, dtype=float)
        f.close()
        return V.Array(array.reshape(tuple(dims)))
