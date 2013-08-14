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

import ctypes
import numpy as np
import scipy.integrate
from pysundials import cvode

from ..utility import environment as Env
from ..language import values as V


class AbstractModel(object):
    """Base class for models in the protocol language."""
    def Simulate(self):
        raise NotImplementedError
    
    def SetOutputFolder(self, path):
        if os.path.isdir(path) and path.startswith('/tmp'):
            shutil.rmtree(path)
        os.mkdir(path)
        self.outputPath = path


class ScipySolver(object):
    def ResetState(self, resetTo):
        self.state = resetTo
        self.solver.set_initial_value(self.state, self.model.freeVariable)
    
    def Simulate(self, endPoint):
        self.state = self.solver.integrate(endPoint)
        assert self.solver.successful()
        
    def AssociateWithModel(self, model):
        self.model = model
        self.state = self.model.initialState.copy()
        self.solver = scipy.integrate.ode(self.model.EvaluateRhs)
        self.solver.set_integrator('vode', atol=1e-7, rtol=1e-5, max_step=1.0, nsteps=2e7, method='bdf')
        
    def SetFreeVariable(self, t):
        self.solver.set_initial_value(self.state, self.model.freeVariable)


class PySundialsSolver(object):
    def AssociateWithModel(self, model):
        self.model = model
        self._state = cvode.NVector(self.model.initialState.copy())
        self.cvode_mem = cvode.CVodeCreate(cvode.CV_BDF, cvode.CV_NEWTON)
        abstol = cvode.realtype(1e-7)
        reltol = cvode.realtype(1e-5)
        cvode.CVodeInit(self.cvode_mem, self.RhsWrapper, 0.0, self._state)
        cvode.CVodeSetTolerances(self.cvode_mem, cvode.CV_SS, reltol, abstol)
        cvode.CVDense(self.cvode_mem, len(self.model.initialState))
        cvode.CVodeSetMaxNumSteps(self.cvode_mem, 20000000)
        cvode.CVodeSetMaxStep(self.cvode_mem, 1.0)
    
    @property
    def state(self):
        return self._state.asarray()
        
    def ResetState(self, resetTo):
        self._state.asarray()[:] = resetTo
        flag = cvode.CVodeReInit(self.cvode_mem, cvode.realtype(self.model.freeVariable), self._state)
        
    def RhsWrapper(self, t, y, ydot, f_data):
        self.model.EvaluateRhs(t, y.asarray(), ydot.asarray())
        return 0
        
    def Simulate(self, endPoint):
        t = cvode.realtype(0)
        flag = cvode.CVode(self.cvode_mem, endPoint, self._state, ctypes.byref(t), cvode.CV_NORMAL)
        assert t.value == endPoint
        assert flag == cvode.CV_SUCCESS
        
    def SetFreeVariable(self, t):
        cvode.CVodeReInit(self.cvode_mem, cvode.realtype(t), self._state)


class AbstractOdeModel(AbstractModel):
    """This is a base class for ODE system models converted from CellML by PyCml.

    Subclasses must define three methods: __init__, EvaluateRhs and GetOutputs.
    See their documentation here for details.
      __init__: sets up self.stateVarMap, self.initialState, self.parameterMap and self.parameters
      EvaluateRhs(self, t, y): returns a numpy array containing the derivatives at the given state
      GetOutputs(self): returns an Environment

    TODO: Figure out the neatest way for protocols to be able to access the current value of a model
    output that isn't a state variable, parameter, or free variable.  (This is allowed in the C++ code.)
    Given that a protocol is only able to do this at a state for which it can also obtain outputs, we
    should probably cache the result of GetOutputs() for use in the ModelWrapperEnvironment.

    TODO: Variable names set up by subclass constructors are just simple strings in the C++, assumed
    to always live in the oxmeta namespace.  This constraint is still imposed by PyCml at present, but
    we should move to use full URIs eventually.

    TODO: Consider whether the ODE solver needs resetting (and how?) when a protocol changes the value
    of a model parameter or state variable.
    """
    def __init__(self, *args, **kwargs):
        """Construct a new ODE system model.

        Before calling this constructor, subclasses should set up the following object attributes:
         * initialState: a numpy array containing initial values for the state variables
         * stateVarMap: a mapping from variable name to index within the state variable vector, for use in our ModelWrapperEnvironment
         * parameters: a numpy array containing model parameter values
         * parameterMap: a mapping from parameter name to index within the parameters vector, for use in our ModelWrapperEnvironment
         * freeVariableName: the name of the free variable, for use in our ModelWrapperEnvironment

        This method will initialise self.state to the initial model state, and set up the ODE solver.
        """
        super(AbstractOdeModel, self).__init__(*args, **kwargs)
        self.savedStates = {}
        self.env = Env.ModelWrapperEnvironment(self)
        default_solver = ScipySolver()
        self.SetSolver(default_solver) 
    
    def SetSolver(self, solver):
        self.solver = solver
        solver.AssociateWithModel(self)
        self.state = self.solver.state
        self.SetFreeVariable(0) # A reasonable initial assumption; can be overridden by simulations
    
    def EvaluateRhs(self, t, y, ydot=np.empty(0)):
        """Compute the derivatives of the model.  This method must be implemented by subclasses.

        :param t:  the free variable, typically time
        :param y:  the ODE system state vector, a numpy array
        :param ydot:  if provided, a vector to be filled in with the derivatives.
            Otherwise the derivatives should be returned by this method.
        """
        raise NotImplementedError

    def GetEnvironmentMap(self):
        return {'oxmeta': self.env}
    
    def GetOutputs(self):
        """Return an Environment containing the model outputs at its current state.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def SetFreeVariable(self, t):
        """Set the value of the free variable (typically time), but retain the model's current state."""
        self.freeVariable = t
        self.solver.SetFreeVariable(t)

    def SaveState(self, name):
        """Save a copy of the current model state associated with the given name, to be restored using ResetState."""
        self.savedStates[name] = self.state.copy()

    def ResetState(self, name=None):
        """Reset the model to the given named saved state, or to initial conditions if no name given."""
        if name is None:
            self.solver.ResetState(self.initialState.copy())
        else:
            self.solver.ResetState(self.savedStates[name].copy())
        self.state = self.solver.state##
        
    def Simulate(self, endPoint):
        """Simulate the model up to the given end point (value of the free variable)."""
        self.solver.state[:] = self.state##
        self.solver.Simulate(endPoint)
        self.freeVariable = endPoint
        self.state = self.solver.state##


class NestedProtocol(AbstractModel):
    def __init__(self, proto, inputExprs, outputNames):
        from ..utility.protocol import Protocol
        self.proto = Protocol(proto)
        self.inputExprs = inputExprs
        self.outputNames = outputNames

    def GetOutputs(self):
        env = Env.Environment()
        for name in self.outputNames:
            env.DefineName(name, self.proto.outputEnv.LookUp(name))
        return env
    
    def GetEnvironmentMap(self):
        return {}
    
    def SetVariable(self, name, valueExpr):
        self.proto.SetInput(name, valueExpr)
    
    def Simulate(self, endPoint):
        for name in self.inputExprs.keys():
           self.proto.SetInput(name, self.inputExprs[name].Evaluate(self.simEnv))
#         self.proto.SetOutputfolder(self.outputPath)
        self.proto.Run()


class TestOdeModel(AbstractOdeModel):
    def __init__(self, a):
        self.initialState = np.array([0])
        self.stateVarMap = {'membrane_voltage': 0, 'y': 0}
        self.parameters = np.array([a])
        self.parameterMap = {'leakage_current': 0, 'a': 0}
        self.freeVariableName = 'time'
        super(TestOdeModel, self).__init__()
        
    def EvaluateRhs(self, t, y):
        return self.parameters[0]
        
    def GetOutputs(self):
        env = Env.Environment()
        env.DefineName('a', V.Simple(self.parameters[self.parameterMap['a']]))
        env.DefineName('y', V.Simple(self.state[self.stateVarMap['y']]))
        env.DefineName('state_variable', V.Array(self.state))
        env.DefineName('time', V.Simple(self.freeVariable))
        env.DefineName('leakage_current', V.Simple(self.parameters[self.parameterMap['leakage_current']]))
        env.DefineName('membrane_voltage', V.Simple(self.state[self.stateVarMap['membrane_voltage']]))
        return env
