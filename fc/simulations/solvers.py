__all__ = ['CvodeSolver', 'DefaultSolver']

from ..sundials.solver import CvodeSolver

# Which type of ODE solver to use by default
DefaultSolver = CvodeSolver
