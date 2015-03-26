
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

"""
Minimal Cython interface to the (CVODE part of the) SUNDIALS library, for use by Functional Curation.

Handles both SUNDIALS 2.4 and 2.5, since the parts of the interface we use didn't change.

Based on http://code.google.com/p/python-sundials/source/browse/trunk/sundials/SundialsLib.pxd
"""

cdef extern from "sundials/sundials_types.h":
    ctypedef double realtype
    ctypedef bint booleantype

cdef extern from "sundials/sundials_nvector.h":
    cdef struct _generic_N_Vector:
        void *content
    ctypedef _generic_N_Vector *N_Vector

cdef extern from "nvector/nvector_serial.h":
    cdef N_Vector N_VMake_Serial(long int vec_length, realtype *v_data)
    N_Vector N_VNew_Serial(long int vec_length)
    void N_VDestroy_Serial(N_Vector v)
    void N_VPrint_Serial(N_Vector v)

    cdef struct _N_VectorContent_Serial:
        long int length
        realtype *data
    ctypedef _N_VectorContent_Serial *N_VectorContent_Serial

cdef extern from "cvode/cvode.h":
    int CV_ADAMS
    int CV_BDF
    int CV_FUNCTIONAL
    int CV_NEWTON
    int CV_NORMAL
    int CV_ONE_STEP

    int CV_SUCCESS
    int CV_TSTOP_RETURN
    int CV_ROOT_RETURN

    int CV_WARNING

    int CV_TOO_MUCH_WORK
    int CV_TOO_MUCH_ACC
    int CV_ERR_FAILURE
    int CV_CONV_FAILURE

    int CV_LINIT_FAIL
    int CV_LSETUP_FAIL
    int CV_LSOLVE_FAIL
    int CV_RHSFUNC_FAIL
    int CV_FIRST_RHSFUNC_ERR
    int CV_REPTD_RHSFUNC_ERR
    int CV_UNREC_RHSFUNC_ERR
    int CV_RTFUNC_FAIL

    int CV_MEM_FAIL
    int CV_MEM_NULL
    int CV_ILL_INPUT
    int CV_NO_MALLOC
    int CV_BAD_K
    int CV_BAD_T
    int CV_BAD_DKY
    int CV_TOO_CLOSE

    ctypedef int (*CVRhsFn)(realtype t, N_Vector y, N_Vector ydot, void *user_data)
    ctypedef int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data)

    void *CVodeCreate(int lmm, int iter)
    int CVodeSetUserData(void *cvode_mem, void *user_data)
    int CVodeInit(void *cvode_mem, CVRhsFn f, realtype t0, N_Vector y0)
    int CVodeReInit(void *cvode_mem, realtype t0, N_Vector y0)
    int CVodeSStolerances(void *cvode_mem, realtype reltol, realtype abstol)
    int CVodeRootInit(void *cvode_mem, int nrtfn, CVRootFn g)

#     int CVodeStep "CVode"(void *cvode_mem, realtype tout, N_Vector yout, realtype *tret, int itask) nogil
    int CVode(void *cvode_mem, realtype tout, N_Vector yout, realtype *tret, int itask)

#     int CVodeSetMaxOrd(void *cvode_mem, int maxord)
    int CVodeSetMaxNumSteps(void *cvode_mem, long int mxsteps)
#     int CVodeSetMaxHnilWarns(void *cvode_mem, int mxhnil)
#     int CVodeSetStabLimDet(void *cvode_mem, booleantype stldet)
#     int CVodeSetInitStep(void *cvode_mem, realtype hin)
#     int CVodeSetMinStep(void *cvode_mem, realtype hmin)
    int CVodeSetMaxStep(void *cvode_mem, realtype hmax)
    int CVodeSetStopTime(void *cvode_mem, realtype tstop)
    int CVodeSetMaxErrTestFails(void *cvode_mem, int maxnef)
#     int CVodeSetMaxNonlinIters(void *cvode_mem, int maxcor)
#     int CVodeSetMaxConvFails(void *cvode_mem, int maxncf)
#     int CVodeSetNonlinConvCoef(void *cvode_mem, realtype nlscoef)
#     int CVodeSetIterType(void *cvode_mem, int iter)
#     int CVodeSetRootDirection(void *cvode_mem, int *rootdir)
#     int CVodeSetNoInactiveRootWarn(void *cvode_mem)
#     int CVodeGetDky(void *cvode_mem, realtype t, int k, N_Vector dky)
#     int CVodeGetWorkSpace(void *cvode_mem, long int *lenrw, long int *leniw)
#     int CVodeGetNumSteps(void *cvode_mem, long int *nsteps)
#     int CVodeGetNumRhsEvals(void *cvode_mem, long int *nfevals)
#     int CVodeGetNumLinSolvSetups(void *cvode_mem, long int *nlinsetups)
#     int CVodeGetNumErrTestFails(void *cvode_mem, long int *netfails)
#     int CVodeGetLastOrder(void *cvode_mem, int *qlast)
#     int CVodeGetCurrentOrder(void *cvode_mem, int *qcur)
#     int CVodeGetNumStabLimOrderReds(void *cvode_mem, long int *nslred)
#     int CVodeGetActualInitStep(void *cvode_mem, realtype *hinused)
#     int CVodeGetLastStep(void *cvode_mem, realtype *hlast)
#     int CVodeGetCurrentStep(void *cvode_mem, realtype *hcur)
#     int CVodeGetCurrentTime(void *cvode_mem, realtype *tcur)
#     int CVodeGetTolScaleFactor(void *cvode_mem, realtype *tolsfac)
#     int CVodeGetErrWeights(void *cvode_mem, N_Vector eweight)
#     int CVodeGetEstLocalErrors(void *cvode_mem, N_Vector ele)
#     int CVodeGetNumGEvals(void *cvode_mem, long int *ngevals)
#     int CVodeGetRootInfo(void *cvode_mem, int *rootsfound)
#     int CVodeGetIntegratorStats(void *cvode_mem, long int *nsteps,
#                                 long int *nfevals, long int *nlinsetups,
#                                 long int *netfails, int *qlast,
#                                 int *qcur, realtype *hinused, realtype *hlast,
#                                 realtype *hcur, realtype *tcur)
#     int CVodeGetNumNonlinSolvIters(void *cvode_mem, long int *nniters)
#     int CVodeGetNumNonlinSolvConvFails(void *cvode_mem, long int *nncfails)
#     int CVodeGetNonlinSolvStats(void *cvode_mem, long int *nniters, long int *nncfails)
#     int CVDlsGetNumJacEvals(void *cvode_mem, long int *njevals)
#     int CVDlsGetNumRhsEvals(void *cvode_mem, long int *nrevalsLS)

    char *CVodeGetReturnFlagName(int flag)
    void CVodeFree(void **cvode_mem)

cdef extern from "cvode/cvode_dense.h":
    int CVDense(void *cvode_mem, int N)
