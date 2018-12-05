
import numpy as np

from .solvers import DefaultSolver
from ..utility import environment as Env


class AbstractModel(object):
    """Base class for models in the protocol language.

    Note that generated models using Cython don't actually inherit from this class, but expose the same interface.
    This is due to Cython limitations that prevent an extension type using multiple inheritance and inheriting from
    a Python class.  Thus if you make changes here you must also change the code generation in translators.py.
    """

    def __init__(self):
        self.indentLevel = 0

    def Simulate(self, endPoint):
        """Simulate the model up to the given end point (value of the free variable).

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def GetOutputs(self):
        """Return a list of the model's outputs at its current state.

        NB: this returns a Python list containing the model outputs as numpy arrays, not subclasses of V.AbstractValue.
        The order of outputs in this list must match self.outputNames, a list of the output names,
        which must be set by subclass constructors.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def SetOutputFolder(self, path):
        # TODO: Use file_handling instead?
        import os
        import shutil
        if os.path.isdir(path) and path.startswith('/tmp'):
            shutil.rmtree(path)
        os.mkdir(path)
        self.outputPath = path

    def SetIndentLevel(self, indentLevel):
        """Set the level of indentation to use for progress output."""
        self.indentLevel = indentLevel


class AbstractOdeModel(AbstractModel):
    """This is a base class for ODE system models converted from CellML by PyCml.

    Subclasses must define three methods: __init__, EvaluateRhs and GetOutputs.
    See their documentation here for details.
      __init__: sets up self.stateVarMap, self.initialState, self.parameterMap and self.parameters
      EvaluateRhs(self, t, y): returns a numpy array containing the derivatives at the given state
      GetOutputs(self): returns a list of model outputs

    Note that generated models using Cython don't actually inherit from this class, but expose the same interface.
    This is due to Cython limitations that prevent an extension type using multiple inheritance and inheriting from
    a Python class.  Thus if you make changes here you must also change the code generation in translators.py.
    """

    def __init__(self, *args, **kwargs):
        """Construct a new ODE system model.

        Before calling this constructor, subclasses should set up the following object attributes:
         * initialState: a numpy array containing initial values for the state variables
         * stateVarMap: a mapping from variable name to index within the state variable vector,
            for use in our ModelWrapperEnvironment
         * parameters: a numpy array containing model parameter values
         * parameterMap: a mapping from parameter name to index within the parameters vector,
            for use in our ModelWrapperEnvironment
         * freeVariableName: the name of the free variable, for use in our ModelWrapperEnvironment
         * outputNames: ordered list of the names of the model outputs, as they will be returned by GetOutputs

        This method will initialise self.state to the initial model state, and set up the ODE solver.
        """
        super(AbstractOdeModel, self).__init__(*args, **kwargs)
        self.savedStates = {}
        self.state = self.initialState.copy()
        self.dirty = False  # whether the solver will need to be reset due to a model change before the next solve
        self.SetSolver(DefaultSolver())
        self.env = Env.ModelWrapperEnvironment(self)
        assert hasattr(self, 'outputNames')

    def SetSolver(self, solver):
        """Specify the ODE solver to use for this model."""
        self.solver = solver
        solver.AssociateWithModel(self)
        self.state = self.solver.state  # This is backwards, but required by PySundials!
        self.SetFreeVariable(0)  # A reasonable initial assumption; can be overridden by simulations

    def EvaluateRhs(self, t, y, ydot=np.empty(0)):
        """Compute the derivatives of the model.  This method must be implemented by subclasses.

        :param t:  the free variable, typically time
        :param y:  the ODE system state vector, a numpy array
        :param ydot:  if provided, a vector to be filled in with the derivatives.
            Otherwise the derivatives should be returned by this method.
        """
        raise NotImplementedError

    def GetEnvironmentMap(self):
        """Get a map from ontology prefix to the environment containing model variables annotated with that ontology."""
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
            # TODO: Raise a nice ProtocolError if state not defined
            self.solver.ResetSolver(self.savedStates[name].copy())

    def Simulate(self, endPoint):
        """Simulate the model up to the given end point (value of the free variable)."""
        self.solver.Simulate(endPoint)
        self.freeVariable = endPoint


class NestedProtocol(AbstractModel):
    """This type of model wraps the execution of an entire protocol."""

    def __init__(self, proto, inputExprs, outputNames, optionalFlags):
        """Create a new nested protocol.

        :param proto: the full path to the protocol description to nest
        :param inputExprs: a map from input name to defining expression, for setting inputs of the nested protocol
        :param outputNames: list of the names of the protocol outputs to keep as our outputs
        :param optionalFlags: list matching outputNames specifying whether each output is optional (i.e. may be missing)
        """
        from ..utility.protocol import Protocol
        self.proto = Protocol(proto)
        self.inputExprs = inputExprs
        self.outputNames = outputNames
        self.optionalFlags = optionalFlags

    def SetIndentLevel(self, indentLevel):
        """Set the level of indentation to use for progress output."""
        super(NestedProtocol, self).SetIndentLevel(indentLevel)
        self.proto.SetIndentLevel(indentLevel)

    def GetOutputs(self):
        """Return selected outputs from the nested protocol."""
        outputs = []
        missing = []
        for i, name in enumerate(self.outputNames[:]):
            try:
                value = self.proto.outputEnv.LookUp(name).unwrapped
            except BaseException:
                if self.optionalFlags[i]:
                    value = None
                    missing.append((i, name))
                else:
                    raise
            outputs.append(value)
        for i, name in reversed(missing):
            del outputs[i]
            del self.outputNames[i]
            del self.optionalFlags[i]
        if missing:
            return outputs, reversed(missing)
        return outputs

    def GetEnvironmentMap(self):
        return {}

    def SetVariable(self, name, valueExpr):
        self.proto.SetInput(name, valueExpr)

    def Simulate(self, endPoint):
        # TODO: Better to pass 'simEnv' to this method?
        for name, expr in self.inputExprs.items():
            self.proto.SetInput(name, expr.Evaluate(self.simEnv))
#         self.proto.SetOutputfolder(self.outputPath) #TODO
        self.proto.Run()


class TestOdeModel(AbstractOdeModel):
    """A very simple model for use in tests: dy/dt = a."""

    def __init__(self, a):
        self.initialState = np.array([0.])
        self.stateVarMap = {'membrane_voltage': 0, 'y': 0}
        self.parameters = np.array([a])
        self.parameterMap = {'SR_leak_current': 0, 'leakage_current': 0, 'a': 0}
        self.freeVariableName = 'time'
        self.outputNames = ['a', 'y', 'state_variable', 'time',
                            'leakage_current', 'SR_leak_current', 'membrane_voltage']
        super(TestOdeModel, self).__init__()

    def EvaluateRhs(self, t, y, ydot=np.empty(0)):
        if ydot.size == 0:
            return self.parameters[0]
        else:
            ydot[0] = self.parameters[0]

    def GetOutputs(self):
        outputs = [np.array(self.parameters[self.parameterMap['a']]),
                   np.array(self.state[self.stateVarMap['y']]),
                   self.state,
                   np.array(self.freeVariable),
                   np.array(self.parameters[self.parameterMap['leakage_current']]),
                   np.array(self.parameters[self.parameterMap['SR_leak_current']]),
                   np.array(self.state[self.stateVarMap['membrane_voltage']])]
        return outputs