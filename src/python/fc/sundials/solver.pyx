
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
import numpy as np

# NB: Relative cimport isn't yet implemented in Cython (although relative import should be)
cimport fc.sundials.sundials as _lib

# Save typing
ctypedef _lib.N_Vector N_Vector

# Data type for numpy arrays
np_dtype = np.float64
ctypedef np.float64_t realtype
assert sizeof(np.float64_t) == sizeof(_lib.realtype) # paranoia

cdef extern from "Python.h":
    object PyBuffer_FromReadWriteMemory(void *ptr, Py_ssize_t size)


# # Debugging!
# import sys
# def fprint(*args):
#     print ' '.join(map(str, args))
#     sys.stdout.flush()


cdef object NumpyView(N_Vector v):
    """Create a Numpy array giving a view on the CVODE vector passed in."""
    cdef _lib.N_VectorContent_Serial v_content = <_lib.N_VectorContent_Serial>(v.content)
    ret = np.empty(v_content.length, dtype=np_dtype)
    ret.data = PyBuffer_FromReadWriteMemory(v_content.data, ret.nbytes)
    return ret

cdef int _RhsWrapper(realtype t, N_Vector y, N_Vector ydot, void* user_data):
    """Cython wrapper around a model RHS that uses numpy, for calling by CVODE."""
    # Create numpy views on the N_Vectors
    np_y = NumpyView(y)
    np_ydot = NumpyView(ydot)
    # Call the Python RHS function
    model = <object>user_data
    try:
        model.EvaluateRhs(t, np_y, np_ydot)
    except Exception, e:
        print e
        return 1 # recoverable error
    return 0


cdef class CvodeSolver:
    """Solver for simulating models using CVODE wrapped by Cython."""

    cdef void* cvode_mem # CVODE solver 'object'
    cdef N_Vector _state  # The state vector of the model being simulated
    
    cdef public object state # Numpy view of the state vector
    cdef public object model # The model being simulated

    def __cinit__(self):
        """Initialise C data attributes on object creation."""
        self.cvode_mem = NULL
        self._state = NULL

    def __dealloc__(self):
        """Free solver memory if allocated."""
        if self.cvode_mem != NULL:
            _lib.CVodeFree(&self.cvode_mem)
        if self._state != NULL:
            _lib.N_VDestroy_Serial(self._state)

    def __init__(self):
        """Python level object initialisation."""
        self.model = None
        self.state = None

    cpdef AssociateWithModel(self, model):
        """Set this as the solver to use for the given model."""
        if self.cvode_mem != NULL:
            _lib.CVodeFree(&self.cvode_mem)
        if self._state != NULL:
            _lib.N_VDestroy_Serial(self._state)
        self.model = model
        # Set our internal state vector as an N_Vector wrapper around the numpy state.
        # Note that model.SetSolver does "self.state = self.solver.state", which is unfortunate but required
        # by the pysundials solver, since that can only wrap an N_Vector with an ndarray, not v.v.
        assert isinstance(model.state, np.ndarray)
        self.state = model.state
        self._state = _lib.N_VMake_Serial(len(model.state), <realtype*>(<np.ndarray>self.state).data)
        # Initialise CVODE
        self.cvode_mem = _lib.CVodeCreate(_lib.CV_BDF, _lib.CV_NEWTON)
        if hasattr(model, 'GetRhsWrapper'):
            raise NotImplementedError
#             _lib.CVodeInit(self.cvode_mem, <_lib.CVRhsFn>(model.GetRhsWrapper()), 0.0, self._state)
        else:
            _lib.CVodeInit(self.cvode_mem, _RhsWrapper, 0.0, self._state)
            _lib.CVodeSetUserData(self.cvode_mem, <void*>(self.model))
        abstol = 1e-7
        reltol = 1e-5
        _lib.CVodeSStolerances(self.cvode_mem, reltol, abstol)
        _lib.CVDense(self.cvode_mem, len(self.state))
        _lib.CVodeSetMaxNumSteps(self.cvode_mem, 20000000)
        _lib.CVodeSetMaxStep(self.cvode_mem, 1.0)

    cpdef ResetState(self, np.ndarray resetTo):
        self.state[:] = resetTo
        _lib.CVodeReInit(self.cvode_mem, self.model.freeVariable, self._state)

    cpdef SetFreeVariable(self, realtype t):
        _lib.CVodeReInit(self.cvode_mem, t, self._state)

    cpdef Simulate(self, realtype endPoint):
        if self.model.dirty:
            # A model variable has changed, so reset the solver
            _lib.CVodeReInit(self.cvode_mem, self.model.freeVariable, self._state)
            self.model.dirty = False
        cdef realtype t = 0
        flag = _lib.CVode(self.cvode_mem, endPoint, self._state, &t, _lib.CV_NORMAL)
        assert t == endPoint
        assert flag == _lib.CV_SUCCESS
