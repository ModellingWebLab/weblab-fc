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

import numpy as np

from .solvers import DefaultSolver
from ..utility import environment as Env


class AbstractModel(object):
    """Base class for models in the protocol language."""
    def Simulate(self, endPoint):
        """Simulate the model up to the given end point (value of the free variable).

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def GetOutputs(self):
        """Return a dictionary containing the model outputs at its current state.

        NB: this returns a Python dictionary containing the model outputs as numpy arrays,
        not subclasses of V.AbstractValue.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def SetOutputFolder(self, path):
        # TODO: Use file_handling instead?
        if os.path.isdir(path) and path.startswith('/tmp'):
            shutil.rmtree(path)
        os.mkdir(path)
        self.outputPath = path


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
        self.state = self.initialState.copy()
        self.dirty = False # whether the solver will need to be reset due to a model change before the next solve
        self.SetSolver(DefaultSolver())
        self.env = Env.ModelWrapperEnvironment(self)

    def SetSolver(self, solver):
        self.solver = solver
        solver.AssociateWithModel(self)
        self.state = self.solver.state # This is backwards, but required by PySundials!
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
            self.solver.ResetSolver(self.initialState.copy())
        else:
            # TOOD: Raise a nice ProtocolError if state not defined
            self.solver.ResetSolver(self.savedStates[name].copy())

    def Simulate(self, endPoint):
        """Simulate the model up to the given end point (value of the free variable)."""
        self.solver.Simulate(endPoint)
        self.freeVariable = endPoint


class NestedProtocol(AbstractModel):
    def __init__(self, proto, inputExprs, outputNames):
        from ..utility.protocol import Protocol
        self.proto = Protocol(proto)
        self.inputExprs = inputExprs
        self.outputNames = outputNames

    def GetOutputs(self):
        outputs = {}
        for name in self.outputNames:
            outputs[name] = self.proto.outputEnv.LookUp(name).unwrapped
        return outputs

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
        self.initialState = np.array([0.])
        self.stateVarMap = {'membrane_voltage': 0, 'y': 0}
        self.parameters = np.array([a])
        self.parameterMap = {'leakage_current': 0, 'a': 0}
        self.freeVariableName = 'time'
        super(TestOdeModel, self).__init__()

    def EvaluateRhs(self, t, y, ydot=np.empty(0)):
        if ydot.size == 0:
            return self.parameters[0]
        else:
            ydot[0] = self.parameters[0]

    def GetOutputs(self):
        outputs = {}
        outputs['a'] = np.array(self.parameters[self.parameterMap['a']])
        outputs['y'] = np.array(self.state[self.stateVarMap['y']])
        outputs['state_variable'] = self.state
        outputs['time'] = np.array(self.freeVariable)
        outputs['leakage_current'] = np.array(self.parameters[self.parameterMap['leakage_current']])
        outputs['membrane_voltage'] = np.array(self.state[self.stateVarMap['membrane_voltage']])
        return outputs
