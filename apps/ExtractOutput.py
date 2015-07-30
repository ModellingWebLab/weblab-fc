#!/usr/bin/env python
"""Copyright (c) 2005-2015, University of Oxford.
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

"""
Export a single protocol output from an HDF5 file to CSV.

Usage: ExtractOutput.py path_to_outputs_h5 output_name [path_to_output_csv]

If the output file is not specified, a file named outputs_{output_name}.csv will be created in the same folder as the h5 file.
"""

import os
import sys

import tables
import numpy as np

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print >> sys.stderr, "Usage:", sys.argv[0], "path_to_output.h5 output_name [path_to_output.csv]"
    sys.exit(1)

in_path = sys.argv[1]
output_name = sys.argv[2]
if len(sys.argv) == 4:
    out_path = sys.argv[3]
else:
    out_path = os.path.join(os.path.dirname(in_path), 'outputs_' + output_name + '.csv')

print "Reading output", output_name, "from", in_path, "and writing CSV to", out_path

f = tables.open_file(in_path, 'r')
a = f.root.output._f_get_child(output_name).read()
np.savetxt(out_path, a.transpose(), delimiter=',')
