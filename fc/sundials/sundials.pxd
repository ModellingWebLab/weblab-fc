"""
Minimal Cython interface to the (CVODE part of the) SUNDIALS library, for use by Functional Curation.

Based on http://code.google.com/p/python-sundials/source/browse/trunk/sundials/SundialsLib.pxd
Includes compatibility wrappers so Cython code can avoid deprecated compile-time IF directives.
"""

cdef extern from "sundials/sundials_types.h":
    ctypedef long int sunindextype
    ctypedef double sunrealtype
    ctypedef bint sunbooleantype

cdef extern from "sundials/sundials_nvector.h":
    cdef struct _generic_N_Vector:
        void *content
    ctypedef _generic_N_Vector *N_Vector

cdef extern from *:
    ctypedef struct _generic_SUNMatrix:
        pass
    ctypedef _generic_SUNMatrix* SUNMatrix

    ctypedef struct _generic_SUNLinearSolver:
        pass
    ctypedef _generic_SUNLinearSolver* SUNLinearSolver

    ctypedef struct SUNContext_:
        pass
    ctypedef SUNContext_* SUNContext

cdef extern from "nvector/nvector_serial.h":
    N_Vector N_VNew_Serial(long int vec_length)
    void N_VDestroy_Serial(N_Vector v)
    void N_VPrint_Serial(N_Vector v)

    cdef struct _N_VectorContent_Serial:
        long int length
        sunrealtype *data
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

    ctypedef int (*CVRhsFn)(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
    ctypedef int (*CVRootFn)(sunrealtype t, N_Vector y, sunrealtype *gout, void *user_data)

    int CVodeSetUserData(void *cvode_mem, void *user_data)
    int CVodeInit(void *cvode_mem, CVRhsFn f, sunrealtype t0, N_Vector y0)
    int CVodeReInit(void *cvode_mem, sunrealtype t0, N_Vector y0)
    int CVodeSStolerances(void *cvode_mem, sunrealtype reltol, sunrealtype abstol)
    int CVodeRootInit(void *cvode_mem, int nrtfn, CVRootFn g)
    int CVode(void *cvode_mem, sunrealtype tout, N_Vector yout, sunrealtype *tret, int itask)
    int CVodeSetMaxNumSteps(void *cvode_mem, long int mxsteps)
    int CVodeSetMaxStep(void *cvode_mem, sunrealtype hmax)
    int CVodeSetStopTime(void *cvode_mem, sunrealtype tstop)
    int CVodeSetMaxErrTestFails(void *cvode_mem, int maxnef)
    char *CVodeGetReturnFlagName(int flag)
    void CVodeFree(void **cvode_mem)

cdef extern from *:
    """
    #include <sundials/sundials_config.h>
    #include <sundials/sundials_types.h>
    #include <sundials/sundials_nvector.h>
    #include <nvector/nvector_serial.h>
    #include <cvode/cvode.h>

    #if SUNDIALS_VERSION_MAJOR >= 3
    #include <sundials/sundials_matrix.h>
    #include <sunmatrix/sunmatrix_dense.h>
    #include <sunlinsol/sunlinsol_dense.h>
    #include <cvode/cvode_ls.h>
    #else
    #include <cvode/cvode_dense.h>
    typedef struct _generic_SUNMatrix* SUNMatrix;
    typedef struct _generic_SUNLinearSolver* SUNLinearSolver;
    #endif

    #if SUNDIALS_VERSION_MAJOR >= 6
    #include <sundials/sundials_context.h>
    #else
    typedef struct SUNContext_* SUNContext;
    #define SUN_COMM_NULL 0
    #endif

    static int FC_SundialsMajor(void)
    {
        return SUNDIALS_VERSION_MAJOR;
    }

    static int FC_SUNContext_Create(SUNContext* sunctx_out)
    {
    #if SUNDIALS_VERSION_MAJOR >= 6
        return SUNContext_Create(SUN_COMM_NULL, sunctx_out);
    #else
        *sunctx_out = NULL;
        return 0;
    #endif
    }

    static int FC_SUNContext_Free(SUNContext* ctx)
    {
    #if SUNDIALS_VERSION_MAJOR >= 6
        return SUNContext_Free(ctx);
    #else
        *ctx = NULL;
        return 0;
    #endif
    }

    static N_Vector FC_N_VMake_Serial(sunindextype vec_length, sunrealtype* v_data, SUNContext sunctx)
    {
    #if SUNDIALS_VERSION_MAJOR >= 6
        return N_VMake_Serial(vec_length, v_data, sunctx);
    #else
        (void)sunctx;
        return N_VMake_Serial((long int)vec_length, v_data);
    #endif
    }

    static void* FC_CVodeCreate(int lmm, int iter, SUNContext sunctx)
    {
    #if SUNDIALS_VERSION_MAJOR >= 6
        (void)iter;
        return CVodeCreate(lmm, sunctx);
    #elif SUNDIALS_VERSION_MAJOR >= 4
        (void)iter;
        (void)sunctx;
        return CVodeCreate(lmm);
    #else
        (void)sunctx;
        return CVodeCreate(lmm, iter);
    #endif
    }

    static SUNMatrix FC_SUNDenseMatrix(sunindextype m, sunindextype n, SUNContext sunctx)
    {
    #if SUNDIALS_VERSION_MAJOR >= 6
        return SUNDenseMatrix(m, n, sunctx);
    #elif SUNDIALS_VERSION_MAJOR >= 3
        (void)sunctx;
        return SUNDenseMatrix(m, n);
    #else
        (void)m; (void)n; (void)sunctx;
        return NULL;
    #endif
    }

    static SUNLinearSolver FC_SUNDenseLinearSolver(N_Vector y, SUNMatrix a, SUNContext sunctx)
    {
    #if SUNDIALS_VERSION_MAJOR >= 6
        return SUNLinSol_Dense(y, a, sunctx);
    #elif SUNDIALS_VERSION_MAJOR >= 3
        (void)sunctx;
        return SUNDenseLinearSolver(y, a);
    #else
        (void)y; (void)a; (void)sunctx;
        return NULL;
    #endif
    }

    static int FC_CVodeSetLinearSolver(void* cvode_mem, SUNLinearSolver ls, SUNMatrix a)
    {
    #if SUNDIALS_VERSION_MAJOR >= 3
        return CVodeSetLinearSolver(cvode_mem, ls, a);
    #else
        (void)cvode_mem; (void)ls; (void)a;
        return CV_SUCCESS;
    #endif
    }

    static int FC_CVDense(void* cvode_mem, int n)
    {
    #if SUNDIALS_VERSION_MAJOR >= 3
        (void)cvode_mem; (void)n;
        return CV_SUCCESS;
    #else
        return CVDense(cvode_mem, n);
    #endif
    }

    static void FC_SUNMatDestroy(SUNMatrix a)
    {
    #if SUNDIALS_VERSION_MAJOR >= 3
        if (a != NULL) SUNMatDestroy(a);
    #else
        (void)a;
    #endif
    }

    static void FC_SUNLinSolFree(SUNLinearSolver ls)
    {
    #if SUNDIALS_VERSION_MAJOR >= 3
        if (ls != NULL) SUNLinSolFree(ls);
    #else
        (void)ls;
    #endif
    }
    """
    int FC_SundialsMajor()
    int FC_SUNContext_Create(SUNContext* sunctx_out)
    int FC_SUNContext_Free(SUNContext* ctx)
    N_Vector FC_N_VMake_Serial(sunindextype vec_length, sunrealtype* v_data, SUNContext sunctx)
    void* FC_CVodeCreate(int lmm, int iter, SUNContext sunctx)
    SUNMatrix FC_SUNDenseMatrix(sunindextype m, sunindextype n, SUNContext sunctx)
    SUNLinearSolver FC_SUNDenseLinearSolver(N_Vector y, SUNMatrix a, SUNContext sunctx)
    int FC_CVodeSetLinearSolver(void* cvode_mem, SUNLinearSolver ls, SUNMatrix a)
    int FC_CVDense(void* cvode_mem, int n)
    void FC_SUNMatDestroy(SUNMatrix a)
    void FC_SUNLinSolFree(SUNLinearSolver ls)