
"""
Minimal Cython interface to the (CVODE part of the) SUNDIALS library, for use by Functional Curation.

Handles SUNDIALS 3.x to 7.x

Based on http://code.google.com/p/python-sundials/source/browse/trunk/sundials/SundialsLib.pxd
"""

cdef extern from "sundials/sundials_types.h":
    ctypedef long int sunindextype
    ctypedef double realtype
    ctypedef bint booleantype

cdef extern from "sundials/sundials_nvector.h":
    cdef struct _generic_N_Vector:
        void *content
    ctypedef _generic_N_Vector *N_Vector

cdef extern from "nvector/nvector_serial.h":
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

# All version-dependent wrappers in one C verbatim block.
cdef extern from *:
    """
    #include <sundials/sundials_config.h>
    #include <cvode/cvode.h>
    #include <nvector/nvector_serial.h>
    #include <sunmatrix/sunmatrix_dense.h>
    #include <sunlinsol/sunlinsol_dense.h>
    #include <sundials/sundials_matrix.h>
    
    #if SUNDIALS_VERSION_MAJOR >= 6
        #include <sundials/sundials_context.h>
        #include <cvode/cvode_ls.h>
    #elif SUNDIALS_VERSION_MAJOR >= 4
        #include <cvode/cvode_ls.h>
    #else
        #include <cvode/cvode_direct.h>
    #endif

    /* In Sundials 7+, realtype was renamed to sunrealtype; provide alias for generated C code */
    #if SUNDIALS_VERSION_MAJOR >= 7
    typedef sunrealtype realtype;
    #endif

    /* Create / free a SUNContext (v6+) or return NULL no-op for older versions */
    static void* fc_SUNContext_Create(void) {
    #if SUNDIALS_VERSION_MAJOR >= 6
        SUNContext sunctx = NULL;
        SUNContext_Create(NULL, &sunctx);
        return (void*)sunctx;
    #else
        return NULL;
    #endif
    }

    static void fc_SUNContext_Free(void* sunctx_ptr) {
    #if SUNDIALS_VERSION_MAJOR >= 6
        SUNContext ctx = (SUNContext)sunctx_ptr;
        if (ctx) SUNContext_Free(&ctx);
    #endif
    }

    /* CVodeCreate wrapper: adds sunctx arg for v6+, drops iter arg in v4+ */
    static void* fc_CVodeCreate(int lmm, void* sunctx) {
    #if SUNDIALS_VERSION_MAJOR >= 6
        return CVodeCreate(lmm, (SUNContext)sunctx);
    #elif SUNDIALS_VERSION_MAJOR >= 4
        return CVodeCreate(lmm);
    #else
        return CVodeCreate(lmm, CV_NEWTON);
    #endif
    }

    /* N_VMake_Serial wrapper: adds sunctx arg for v6+ */
    static N_Vector fc_N_VMake_Serial(int len, double* data, void* sunctx) {
    #if SUNDIALS_VERSION_MAJOR >= 6
        return N_VMake_Serial((sunindextype)len, data, (SUNContext)sunctx);
    #else
        return N_VMake_Serial((long int)len, data);
    #endif
    }

    /* SUNDenseMatrix wrapper: adds sunctx arg for v6+ */
    static void* fc_SUNDenseMatrix(int M, int N, void* sunctx) {
    #if SUNDIALS_VERSION_MAJOR >= 6
        return (void*)SUNDenseMatrix((sunindextype)M, (sunindextype)N, (SUNContext)sunctx);
    #else
        return (void*)SUNDenseMatrix(M, N);
    #endif
    }

    /* Dense linear solver: renamed to SUNLinSol_Dense in v6; adds sunctx */
    static void* fc_SUNLinSol_Dense(void* y, void* A, void* sunctx) {
    #if SUNDIALS_VERSION_MAJOR >= 6
        return (void*)SUNLinSol_Dense((N_Vector)y, (SUNMatrix)A, (SUNContext)sunctx);
    #else
        return (void*)SUNDenseLinearSolver((N_Vector)y, (SUNMatrix)A);
    #endif
    }

    /* CVodeSetLinearSolver: renamed from CVDlsSetLinearSolver in v4+ */
    static int fc_CVodeSetLinearSolver(void* cvode_mem, void* LS, void* A) {
    #if SUNDIALS_VERSION_MAJOR >= 4
        return CVodeSetLinearSolver(cvode_mem, (SUNLinearSolver)LS, (SUNMatrix)A);
    #else
        return CVDlsSetLinearSolver(cvode_mem, (SUNLinearSolver)LS, (SUNMatrix)A);
    #endif
    }

    static void fc_SUNLinSolFree(void* LS) {
        if (LS) SUNLinSolFree((SUNLinearSolver)LS);
    }

    static void fc_SUNMatDestroy(void* A) {
        if (A) SUNMatDestroy((SUNMatrix)A);
    }
    """
    void* fc_SUNContext_Create()
    void fc_SUNContext_Free(void* sunctx)
    void* fc_CVodeCreate(int lmm, void* sunctx)
    N_Vector fc_N_VMake_Serial(int len, realtype* data, void* sunctx)
    void* fc_SUNDenseMatrix(int M, int N, void* sunctx)
    void* fc_SUNLinSol_Dense(void* y, void* A, void* sunctx)
    int fc_CVodeSetLinearSolver(void* cvode_mem, void* LS, void* A)
    void fc_SUNLinSolFree(void* LS)
    void fc_SUNMatDestroy(void* A)
