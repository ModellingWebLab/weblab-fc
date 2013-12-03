
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

cimport numpy as np
cimport fc.sundials.sundials as _lib

# Save typing
ctypedef _lib.N_Vector N_Vector
ctypedef np.float64_t realtype

#cdef object NumpyView(N_Vector v)

#cdef int _RhsWrapper(realtype t, N_Vector y, N_Vector ydot, void* user_data)

cdef class CvodeSolver:
    cdef void* cvode_mem # CVODE solver 'object'
    cdef N_Vector _state  # The state vector of the model being simulated
    
    cdef public np.ndarray state # Numpy view of the state vector
    cdef public object model # The model being simulated

    cpdef AssociateWithModel(self, model)
    cpdef ResetSolver(self, np.ndarray[realtype, ndim=1] resetTo)
    cpdef SetFreeVariable(self, realtype t)
    cpdef Simulate(self, realtype endPoint)
    
    cdef ReInit(self)
    cdef CheckFlag(self, int flag, char* called)
