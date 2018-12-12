
__all__ = ['CvodeSolver', 'ScipySolver', 'DefaultSolver']

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
        self.state = model.state
        assert self.state.dtype == float
        self.solver = scipy.integrate.ode(model.EvaluateRhs)
        self.solver.set_integrator('vode', atol=1e-7, rtol=1e-5, max_step=1.0, nsteps=2e7, method='bdf')

    def SetFreeVariable(self, t):
        self.solver.set_initial_value(self.state, self.model.freeVariable)


# Which type of ODE solver to use by default
DefaultSolver = CvodeSolver
