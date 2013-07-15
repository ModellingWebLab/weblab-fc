
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
import Protocol

import numpy as np
import os
import time

def N(v):
    return M.Const(V.Simple(v))

class TestSpeedRealProto(unittest.TestCase):
    def TestS1S2(self):
        # Parse the protocol into a sequence of post-processing statements
        proto_file = 'projects/FunctionalCuration/test/protocols/compact/S1S2.txt'
        proto = Protocol.Protocol(proto_file)
        # Load the raw simulation data from file
        data_folder = 'projects/FunctionalCuration/test/data/TestSpeedRealProto/S1S2'
        membrane_voltage = self.Load2d(os.path.join(data_folder, 'outputs_membrane_voltage.csv'))
        time_1d = self.Load(os.path.join(data_folder, 'outputs_time_1d.csv'))
        time_2d = A.NewArray(M.Const(time_1d),
                             E.TupleExpression(N(0), N(0), N(1), N(91), M.Const(V.String('_'))),
                             comprehension=True).Evaluate(proto.env)
        proto.env.DefineName('sim:time', time_2d)
        proto.env.DefineName('sim:membrane_voltage', membrane_voltage)
        # Run the protocol
        self.Time(proto.Run)
        # Check the results
        self.CheckResults(proto, {'raw_APD90': 2, 'raw_DI': 2, 'max_S1S2_slope': 1}, data_folder)

    def TestIcal(self):
        proto = Protocol.Protocol('projects/FunctionalCuration/test/protocols/compact/ICaL.txt')
        data_folder = 'projects/FunctionalCuration/test/data/TestSpeedRealProto/ICaL'
        proto.env.DefineName('sim:membrane_voltage',
                             self.Load(os.path.join(data_folder, 'outputs_membrane_voltage.csv')))
        proto.env.DefineName('sim:membrane_L_type_calcium_current',
                             self.Load(os.path.join(data_folder, 'outputs_membrane_L_type_calcium_current.csv')))
        proto.env.DefineName('sim:extracellular_calcium_concentration',
                             self.Load(os.path.join(data_folder, 'outputs_extracellular_calcium_concentration.csv')))
        proto.env.DefineName('sim:oxmeta:extracellular_calcium_concentration',
                             V.Simple(proto.env.LookUp('sim:extracellular_calcium_concentration').array[1,0,0]))
        shape = list(proto.env.LookUp('sim:membrane_voltage').array.shape)
        shape[-1] = 1
        time = V.Array(np.tile(np.arange(-10.0, 500.01, 0.01), shape))
        proto.env.DefineName('sim:time', time)
        self.Time(proto.Run)
        # Check the results
        self.CheckResults(proto, {'min_LCC': 2, 'final_membrane_voltage': 1}, data_folder)

    def Time(self, func):
        start = time.time()
        func()
        end = time.time()
        print "Protocol execution took", (end - start), "seconds"

    def CheckResults(self, proto, expectedSpec, dataFolder):
        """Check protocol results against saved values.
        expectedSpec is a dictionary mapping result name to number of dimensions, so we can use the correct Load* method.
        """
        for name, ndims in expectedSpec.iteritems():
            data_file = os.path.join(dataFolder, 'outputs_' + name + '.csv')
            if ndims == 2:
                method = self.Load2d
            else:
                method = self.Load
            expected = method(data_file)
            actual = proto.env.LookUp(name)
            np.testing.assert_allclose(actual.array, expected.array, rtol=0.01)
    
    def CheckFileCompression(self, filePath):
        """Return (real_path, is_compressed) if a .gz compressed version of filePath exists."""
        real_path = filePath
        if filePath.endswith('.gz'):
            is_compressed = True
        else:
            if os.path.exists(filePath):
                is_compressed = False
            elif os.path.exists(filePath + '.gz'):
                real_path += '.gz'
                is_compressed = True
        return real_path, is_compressed

    def Load2d(self, filePath):
        real_path, is_compressed = self.CheckFileCompression(filePath)
        array = np.loadtxt(real_path, dtype=float, delimiter=',', unpack=True) # unpack transposes the array
        if array.ndim == 1:
            array = array[:, np.newaxis]
        return V.Array(array)

    def Load(self, filePath):
        real_path, is_compressed = self.CheckFileCompression(filePath)
        if is_compressed:
            import gzip
            f = gzip.GzipFile(real_path, 'rb')
        else:
            f = open(real_path, 'r')
        f.readline() # Strip comment line
        dims = map(int, f.readline().split(','))[1:]
        array = np.loadtxt(f, dtype=float)
        f.close()
        return V.Array(array.reshape(tuple(dims)))
