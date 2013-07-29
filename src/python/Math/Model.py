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
import Environment as Env
import MathExpressions as M
import numpy as np
from ErrorHandling import ProtocolError
import scipy.integrate
import Values as V


class AbstractModel(object):
    """Base class for statements in the protocol language."""
    def Simulate(self):
        raise NotImplementedError    
    
class TestOdeModel(AbstractModel):
    def __init__(self, a):
        self.savedStates = {}
        self.a = a
        self.r = scipy.integrate.ode(self.f)
        self.r.set_initial_value(0, 0)
        self.y = self.r.y
        self.modifiers = []
        self.env = Env.ModelWrapperEnvironment(self)
        self.time = 0
        
    def f(self, t, y):
        return self.a
    
    def SaveState(self, when, name):
        self.savedStates[name] = self.r.y
        
    def SetInitialTime(self, t):
        # self.time is t
        self.r.set_initial_value(self.r.y, 0)
        
    def SetVariable(self, when, env, variableName, value):
        if not hasattr(self, variableName):
            raise ProtocolError("Type", type(self), "does not have a variable named", variableName)
        if variableName == 'y':
            self.r.y = np.array([value])
        else:
            setattr(self, variableName, value)
    
    def SetVariableNow(self, name, value):
        setattr(self, name, value)
        
    def GetEnvironmentMap(self):
#         self.env.OverwriteDefinition('membrane_voltage', V.Simple(self.r.y[0]))
#         self.env.OverwriteDefinition('leakage_current', V.Simple(self.a))
        #define self.env as an environment with the variables for the model in it
        # membrane_voltage = self.y, leakage_current = self.a
        return {'oxmeta': self.env}
    
    def ResetState(self, when, name):
        if name is None:
            self.r.set_initial_value(0, 0)   
        else:
            self.r.set_initial_value(self.savedStates[name], 0)
        self.y = self.r.y
                
    def Simulate(self, endPoint):
        self.y = self.r.integrate(endPoint)
        assert self.r.successful() # self.time is endpoint
    
    def GetOutputs(self):
        env = Env.Environment()
        env.DefineName('a', V.Simple(self.a))
        env.DefineName('y', V.Simple(self.r.y))
        env.DefineName('leakage_current', V.Simple(self.a))
        env.DefineName('membrane_voltage', V.Simple(self.r.y))
        return env


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
        self.state = self.initialState.copy()
        self.solver = scipy.integrate.ode(self.EvaluateRhs)
        self.SetFreeVariable(0) # A reasonable initial assumption; can be overridden by simulations
        self.savedStates = {}

    def EvaluateRhs(self, t, y):
        """Compute the derivatives of the model.  This method must be implemented by subclasses.

        :param t:  the free variable, typically time
        :param y:  the ODE system state vector, a numpy array
        """
        raise NotImplementedError

    def GetOutputs(self):
        """Return an Environment containing the model outputs at its current state.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def SetFreeVariable(self, t):
        """Set the value of the free variable (typically time), but retain the model's current state."""
        self.freeVariable = t
        self.solver.set_initial_value(self.state, self.freeVariable)

    def SaveState(self, name):
        """Save a copy of the current model state associated with the given name, to be restored using ResetState."""
        self.savedStates[name] = self.state.copy()

    def ResetState(self, name=None):
        """Reset the model to the given named saved state, or to initial conditions if no name given."""
        if name is None:
            self.state = self.initialState.copy()
        else:
            self.state = self.savedStates[name].copy()
        self.solver.set_initial_value(self.state, self.freeVariable)

    def Simulate(self, endPoint):
        """Simulate the model up to the given end point (value of the free variable)."""
        self.state = self.solver.integrate(endPoint)
        assert self.solver.successful()
        self.freeVariable = endPoint


