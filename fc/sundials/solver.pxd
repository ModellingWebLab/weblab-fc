
cimport numpy as np
cimport fc.sundials.sundials as _lib

# Save typing
ctypedef _lib.N_Vector N_Vector
ctypedef np.float64_t realtype


cdef class CvodeSolver:
    cdef void* cvode_mem # CVODE solver 'object'

    cdef N_Vector _state # The state vector of the model being simulated
    cdef int _state_size # The number of state variables / length of the state vector

    cdef public np.ndarray state # Numpy view of the state vector
    cdef public object model # The model being simulated

    # SUNContext (v6+; NULL for older versions)
    cdef void* sunctx

    # Linear matrix solving (opaque void* pointers to SUNMatrix and SUNLinearSolver)
    cdef void* sundense_matrix
    cdef void* sundense_solver

    cpdef associate_with_model(self, model)
    cpdef reset_solver(self, np.ndarray[realtype, ndim=1] reset_to)
    cpdef set_free_variable(self, realtype t)
    cpdef simulate(self, realtype end_point)

    cdef re_init(self)
    cdef check_flag(self, int flag, char* called)
