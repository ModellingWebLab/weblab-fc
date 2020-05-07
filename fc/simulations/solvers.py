"""
ODE solvers. Currently only CVODE is supported, as it outperformed any previous solvers.
"""
__all__ = ['CvodeSolver', 'DefaultSolver']

from ..sundials.solver import CvodeSolver

# Which type of ODE solver to use by default
DefaultSolver = CvodeSolver
