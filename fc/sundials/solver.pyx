
# cython: profile=True

cimport numpy as np
import numpy as np
from cython cimport view

# NB: Relative cimport isn't yet implemented in Cython (although relative import should be)
cimport fc.sundials.sundials as _lib
from fc.utility.error_handling import ProtocolError

# Data type for numpy arrays
np_dtype = np.float64
assert sizeof(np.float64_t) == sizeof(_lib.realtype) # paranoia

# # Debugging!
# import sys
# def fprint(*args):
#     print ' '.join(map(str, args))
#     sys.stdout.flush()


cdef object NumpyView(N_Vector v):
    """Create a Numpy array giving a view on the CVODE vector passed in."""
    cdef _lib.N_VectorContent_Serial v_content = <_lib.N_VectorContent_Serial>(v.content)
    cdef view.array data_view = view.array(shape=(v_content.length,), itemsize=sizeof(realtype),
                                           format='d', mode='c', allocate_buffer=False)
    data_view.data = <char *> v_content.data
    ret = np.asarray(data_view, dtype=np_dtype)
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

    def __cinit__(self):
        """Initialise C data attributes on object creation."""
        self.cvode_mem = NULL
        self._state = NULL
        self._state_size = 0

        IF FC_SUNDIALS_MAJOR >= 3:
            self.sundense_matrix = NULL
            self.sundense_solver = NULL

    def __dealloc__(self):
        """Free solver memory if allocated."""
        if self.cvode_mem != NULL:
            _lib.CVodeFree(&self.cvode_mem)
        if self._state != NULL:
            _lib.N_VDestroy_Serial(self._state)
        IF FC_SUNDIALS_MAJOR >= 3:
            if self.sundense_solver != NULL:
                _lib.SUNLinSolFree(self.sundense_solver)
            if self.sundense_matrix != NULL:
                _lib.SUNMatDestroy(self.sundense_matrix)

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
        assert isinstance(model.state, np.ndarray)
        self.state = model.state
        self._state_size = len(model.state)
        self._state = _lib.N_VMake_Serial(self._state_size, <realtype*>(<np.ndarray>self.state).data)

        # Create CVode object
        IF FC_SUNDIALS_MAJOR >= 4:
            self.cvode_mem = _lib.CVodeCreate(_lib.CV_BDF)
        ELSE:
            self.cvode_mem = _lib.CVodeCreate(_lib.CV_BDF, _lib.CV_NEWTON)

        # Initialise CVode
        if hasattr(self, 'SetRhsWrapper'):
            # A subclass will take care of the RHS function
            self.SetRhsWrapper()
        else:
            flag = _lib.CVodeInit(self.cvode_mem, _RhsWrapper, 0.0, self._state)
            self.CheckFlag(flag, 'CVodeInit')

        # Pass model in as CVode user data
        flag = _lib.CVodeSetUserData(self.cvode_mem, <void*>(self.model))
        self.CheckFlag(flag, 'CVodeSetUserData')

        # Set CVode tolerances
        abstol = 1e-8
        reltol = 1e-6
        flag = _lib.CVodeSStolerances(self.cvode_mem, reltol, abstol)
        self.CheckFlag(flag, 'CVodeSStolerances')

        # Create dense matrix for use in linear solves
        if self._state_size > 0:
            IF FC_SUNDIALS_MAJOR >= 3:
                # Create dense matrix
                self.sundense_matrix = _lib.SUNDenseMatrix(self._state_size, self._state_size)
                # Not sure how to check these now! See comments for CheckFlag below
                #self.CheckFlag(<void*>self.sundense_matrix, 'SUNDenseMatrix')
                # Create linear solver
                self.sundense_solver = _lib.SUNDenseLinearSolver(self._state, self.sundense_matrix)
                #self.CheckFlag(<void*>self.sundense_solver, 'SUNDenseLinearSolver')
                # Tell cvode to use this solver
                flag = _lib.CVDlsSetLinearSolver(self.cvode_mem, self.sundense_solver, self.sundense_matrix)
                self.CheckFlag(flag, 'CVDlsSetLinearSolver')
            ELSE:
                # Create dense matrix
                flag = _lib.CVDense(self.cvode_mem, self._state_size)
                self.CheckFlag(flag, 'CVDense')

        _lib.CVodeSetMaxNumSteps(self.cvode_mem, 20000000)
        _lib.CVodeSetMaxStep(self.cvode_mem, 0.5)
        _lib.CVodeSetMaxErrTestFails(self.cvode_mem, 15)

    cpdef ResetSolver(self, np.ndarray[realtype, ndim=1] resetTo):
        self.state[:] = resetTo
        self.ReInit()

    cpdef SetFreeVariable(self, realtype t):
        self.model.freeVariable = t
        self.ReInit()

    cpdef Simulate(self, realtype endPoint):
        cdef realtype t = 0
        if self._state_size > 0:
            if self.model.dirty:
                # A model variable has changed, so reset the solver
                self.ReInit()
            # Stop CVODE going past the end of where we wanted and interpolating back
            flag = _lib.CVodeSetStopTime(self.cvode_mem, endPoint)
            assert flag == _lib.CV_SUCCESS
            # Do the solve
            flag = _lib.CVode(self.cvode_mem, endPoint, self._state, &t, _lib.CV_NORMAL)
            if flag < 0:
                flag_name = _lib.CVodeGetReturnFlagName(flag)
                raise ProtocolError("Failed to solve model ODE system at time %g: %s" % (t, flag_name))
            else:
                assert t == endPoint
        self.model.freeVariable = endPoint

    cdef ReInit(self):
        """Reset CVODE's state because time or the RHS function has changed (e.g. parameter or state var change)."""
        _lib.CVodeReInit(self.cvode_mem, self.model.freeVariable, self._state)
        self.model.dirty = False

    cdef CheckFlag(self, int flag, char* called):
        """Check for a successful call to a CVODE routine, and report the error if not."""
        #TODO: Sundials can also return a null pointer instead of an int, the
        # example implementation has a void* signature and then casts/dereferences
        # to int
        if flag != _lib.CV_SUCCESS:
            flag_name = _lib.CVodeGetReturnFlagName(flag)
            raise ProtocolError("Error calling CVODE routine %s: %s" % (called, flag_name))
