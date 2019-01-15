
cimport numpy as np
cimport fc.sundials.sundials as _lib

# Save typing
ctypedef _lib.N_Vector N_Vector
ctypedef np.float64_t realtype
IF FC_SUNDIALS_MAJOR >= 3:
    ctypedef _lib.SUNMatrix SUNMatrix
    ctypedef _lib.SUNLinearSolver SUNLinearSolver


#cdef object NumpyView(N_Vector v)

#cdef int _RhsWrapper(realtype t, N_Vector y, N_Vector ydot, void* user_data)

cdef class CvodeSolver:
    cdef void* cvode_mem # CVODE solver 'object'

    cdef N_Vector _state # The state vector of the model being simulated
    cdef int _state_size # The number of state variables / length of the state vector

    cdef public np.ndarray state # Numpy view of the state vector
    cdef public object model # The model being simulated

    cpdef AssociateWithModel(self, model)
    cpdef ResetSolver(self, np.ndarray[realtype, ndim=1] resetTo)
    cpdef SetFreeVariable(self, realtype t)
    cpdef Simulate(self, realtype endPoint)

    cdef ReInit(self)
    cdef CheckFlag(self, int flag, char* called)

    IF FC_SUNDIALS_MAJOR >= 3:
        # Linear matrix solving in sundials 3+
        cdef SUNMatrix sundense_matrix
        cdef SUNLinearSolver sundense_solver

