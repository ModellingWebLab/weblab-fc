
__all__ = ['CvodeSolver', 'ScipySolver', 'PySundialsSolver', 'DefaultSolver']

from ..sundials.solver import CvodeSolver

import scipy.integrate


class ScipySolver(object):
    """Solver for simulating models using SciPy's builtin ODE solvers.  NB: slow!"""

    def ResetSolver(self, resetTo):
        self.state[:] = resetTo
        self.solver.set_initial_value(self.state, self.model.freeVariable)

    def Simulate(self, endPoint):
        if self.model.dirty:
            # A model variable has changed, so reset the solver
            self.solver.set_initial_value(self.state, self.model.freeVariable)
            self.model.dirty = False
        self.state[:] = self.solver.integrate(endPoint)
        assert self.solver.successful()

    def AssociateWithModel(self, model):
        self.model = model
        self.state = self.model.state
        assert self.state.dtype == float
        self.solver = scipy.integrate.ode(self.model.EvaluateRhs)
        self.solver.set_integrator('vode', atol=1e-7, rtol=1e-5, max_step=1.0, nsteps=2e7, method='bdf')

    def SetFreeVariable(self, t):
        self.solver.set_initial_value(self.state, self.model.freeVariable)


try:
    from pysundials import cvode
    import ctypes
except ImportError:
    cvode = None

if cvode:
    class PySundialsSolver(object):
        """Solver for simulating models using http://pysundials.sourceforge.net/"""

        def AssociateWithModel(self, model):
            self.model = model
            self._state = cvode.NVector(self.model.state)  # NB: This copies the data
            self.cvode_mem = cvode.CVodeCreate(cvode.CV_BDF, cvode.CV_NEWTON)
            abstol = cvode.realtype(1e-7)
            reltol = cvode.realtype(1e-5)
            cvode.CVodeInit(self.cvode_mem, self.RhsWrapper, 0.0, self._state)
            cvode.CVodeSetTolerances(self.cvode_mem, cvode.CV_SS, reltol, abstol)
            cvode.CVDense(self.cvode_mem, len(self.model.state))
            cvode.CVodeSetMaxNumSteps(self.cvode_mem, 20000000)
            cvode.CVodeSetMaxStep(self.cvode_mem, 1.0)

        @property
        def state(self):
            return self._state.asarray()

        def ResetSolver(self, resetTo):
            self._state.asarray()[:] = resetTo
            cvode.CVodeReInit(self.cvode_mem, cvode.realtype(self.model.freeVariable), self._state)

        def RhsWrapper(self, t, y, ydot, f_data):
            self.model.EvaluateRhs(t, y.asarray(), ydot.asarray())
            return 0

        def Simulate(self, endPoint):
            if self.model.dirty:
                # A model variable has changed, so reset the solver
                cvode.CVodeReInit(self.cvode_mem, cvode.realtype(self.model.freeVariable), self._state)
                self.model.dirty = False
            t = cvode.realtype(0)
            flag = cvode.CVode(self.cvode_mem, endPoint, self._state, ctypes.byref(t), cvode.CV_NORMAL)
            assert t.value == endPoint
            assert flag == cvode.CV_SUCCESS

        def SetFreeVariable(self, t):
            cvode.CVodeReInit(self.cvode_mem, cvode.realtype(t), self._state)
else:
    class PySundialsSolver(CvodeSolver):
        """Fake PySundials solver that in reality uses our own CVODE wrapper."""

        def __init__(self):
            import sys
            print("PySundials not found; using internal CVODE wrapper instead.", file=sys.stderr)
            super(PySundialsSolver, self).__init__()

# Which type of ODE solver to use by default
DefaultSolver = CvodeSolver
