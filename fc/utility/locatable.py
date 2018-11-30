
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

import inspect
import os
import sys

class Locatable(object):
    """Base class for constructs in the protocol language.
    
    This class stores the location within a protocol file giving rise to the construct,
    and also whether the construct is being traced, i.e. can have its value written to a trace
    file when evaluated.  The class variable outputFolder records where trace should be written,
    if enabled, and the Trace method can be used to trace a value.
    """
    def __init__(self, location=None):
        """Create a new construct with optional location information.
        
        If no location is given, a string representation of the Python stack of the caller will be used.
        This is not ideal, but better than nothing!
        """
        if location is None:
            location = str(inspect.stack()[2][1:5])
        self.location = location
        self.trace = False
        
    def Trace(self, value, stream=sys.stdout, prefix=''):
        """Trace the given value.
        
        A synopsis will be written to stream, and the full value to a trace file, if self.outputFolder
        is defined.
        """
        from ..language import values as V
        if isinstance(value, V.Array) and value.array.size > 10:
            stream.write(prefix + 'array shape ' + str(value.array.shape) + '\n')
        else:
            stream.write(prefix + str(value) + '\n')

        if self.outputFolder:
            trace_path = os.path.join(self.outputFolder.path, 'trace.txt')
            f = open(trace_path, 'a+')
            f.write(prefix + str(value) + '\n')
            f.close()

    # Global that sets where to write trace, if enabled
    outputFolder = None
