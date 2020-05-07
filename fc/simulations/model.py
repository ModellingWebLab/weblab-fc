"""
Classes to represent models used in simulation (e.g. generated code or hard-coded models).
"""
import numpy as np

from .solvers import DefaultSolver
from .. import environment as Env


class AbstractModel(object):
    """Base class for models in the protocol language.

    Note that generated models using Cython don't actually inherit from this class, but expose the same interface.
    This is due to Cython limitations that prevent an extension type using multiple inheritance and inheriting from
    a Python class.  Thus if you make changes here you must also change the code generation in translators.py.
    """

    def __init__(self):
        self.indent_level = 0

    def simulate(self, end_point):
        """Simulate the model up to the given end point (value of the free variable).

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_outputs(self):
        """Return a list of the model's outputs at its current state.

        NB: this returns a Python list containing the model outputs as numpy arrays, not subclasses of V.AbstractValue.
        The order of outputs in this list must match self.output_names, a list of the output names,
        which must be set by subclass constructors.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def set_output_folder(self, path):
        # TODO: Use file_handling instead?
        import os
        import shutil
        if os.path.isdir(path) and path.startswith('/tmp'):
            shutil.rmtree(path)
        os.mkdir(path)
        self.output_path = path

    def set_indent_level(self, indent_level):
        """Set the level of indentation to use for progress output."""
        self.indent_level = indent_level


class AbstractOdeModel(AbstractModel):
    """This is a base class for ODE system models converted from CellML by PyCml.

    Subclasses must define three methods: __init__, evaluate_rhs and get_outputs.
    See their documentation here for details.
      __init__: sets up self.state_var_map, self.initial_state, self.parameter_map and self.parameters
      evaluate_rhs(self, t, y): returns a numpy array containing the derivatives at the given state
      get_outputs(self): returns a list of model outputs

    Note that generated models using Cython don't actually inherit from this class, but expose the same interface.
    This is due to Cython limitations that prevent an extension type using multiple inheritance and inheriting from
    a Python class.  Thus if you make changes here you must also change the code generation in translators.py.
    """

    def __init__(self, *args, **kwargs):
        """Construct a new ODE system model.

        Before calling this constructor, subclasses should set up the following object attributes:
         * initial_state: a numpy array containing initial values for the state variables
         * state_var_map: a mapping from variable name to index within the state variable vector,
            for use in our ModelWrapperEnvironment
         * parameters: a numpy array containing model parameter values
         * parameter_map: a mapping from parameter name to index within the parameters vector,
            for use in our ModelWrapperEnvironment
         * free_variable_name: the name of the free variable, for use in our ModelWrapperEnvironment
         * output_names: ordered list of the names of the model outputs, as they will be returned by get_outputs

        This method will initialise self.state to the initial model state, and set up the ODE solver.
        """
        super(AbstractOdeModel, self).__init__(*args, **kwargs)
        self.saved_states = {}
        self.state = self.initial_state.copy()
        self.dirty = False  # whether the solver will need to be reset due to a model change before the next solve
        self.set_solver(DefaultSolver())
        self.env = Env.ModelWrapperEnvironment(self)
        assert hasattr(self, 'output_names')

    def set_solver(self, solver):
        """Specify the ODE solver to use for this model."""
        self.solver = solver
        solver.associate_with_model(self)
        self.set_free_variable(0)  # A reasonable initial assumption; can be overridden by simulations

    def evaluate_rhs(self, t, y, ydot=np.empty(0)):
        """Compute the derivatives of the model.  This method must be implemented by subclasses.

        :param t:  the free variable, typically time
        :param y:  the ODE system state vector, a numpy array
        :param ydot:  if provided, a vector to be filled in with the derivatives.
            Otherwise the derivatives should be returned by this method.
        """
        raise NotImplementedError

    def get_environment_map(self):
        """Get a map from ontology prefix to the environment containing model variables annotated with that ontology."""
        return {'oxmeta': self.env}

    def set_free_variable(self, t):
        """Set the value of the free variable (typically time), but retain the model's current state."""
        self.free_variable = t
        self.solver.set_free_variable(t)

    def save_state(self, name):
        """Save a copy of the current model state associated with the given name, to be restored using reset_state."""
        self.saved_states[name] = self.state.copy()

    def reset_state(self, name=None):
        """Reset the model to the given named saved state, or to initial conditions if no name given."""
        if name is None:
            self.solver.reset_solver(self.initial_state.copy())
        else:
            # TODO: Raise a nice ProtocolError if state not defined
            self.solver.reset_solver(self.saved_states[name].copy())

    def simulate(self, end_point):
        """Simulate the model up to the given end point (value of the free variable)."""
        self.solver.simulate(end_point)
        self.free_variable = end_point


class NestedProtocol(AbstractModel):
    """This type of model wraps the execution of an entire protocol."""

    def __init__(self, proto, input_exprs, output_names, optional_flags):
        """Create a new nested protocol.

        :param proto: the full path to the protocol description to nest
        :param input_exprs: a map from input name to defining expression, for setting inputs of the nested protocol
        :param output_names: list of the names of the protocol outputs to keep as our outputs
        :param optional_flags: list matching output_names specifying whether each output is optional (i.e. may be
        missing)
        """
        from ..protocol import Protocol
        self.proto = Protocol(proto)
        self.input_exprs = input_exprs
        self.output_names = output_names
        self.optional_flags = optional_flags

    def set_indent_level(self, indent_level):
        """Set the level of indentation to use for progress output."""
        super(NestedProtocol, self).set_indent_level(indent_level)
        self.proto.set_indent_level(indent_level)

    def get_outputs(self):
        """Return selected outputs from the nested protocol."""
        outputs = []
        missing = []
        for i, name in enumerate(self.output_names[:]):
            try:
                value = self.proto.output_env.look_up(name).unwrapped
            except Exception:
                if self.optional_flags[i]:
                    value = None
                    missing.append((i, name))
                else:
                    raise
            outputs.append(value)
        for i, name in reversed(missing):
            del outputs[i]
            del self.output_names[i]
            del self.optional_flags[i]
        if missing:
            return outputs, reversed(missing)
        return outputs

    def get_environment_map(self):
        return {}

    def set_variable(self, name, value_expr):
        self.proto.set_input(name, value_expr)

    def simulate(self, end_point):
        # TODO: Better to pass 'sim_env' to this method?
        for name, expr in self.input_exprs.items():
            self.proto.set_input(name, expr.evaluate(self.sim_env))
#         self.proto.SetOutputfolder(self.output_path) #TODO
        self.proto.run()


class TestOdeModel(AbstractOdeModel):
    """A very simple model for use in tests: dy/dt = a."""

    def __init__(self, a):
        self.initial_state = np.array([0.])
        self.state_var_map = {'membrane_voltage': 0, 'y': 0}
        self.parameters = np.array([a])
        self.parameter_map = {'SR_leak_current': 0, 'leakage_current': 0, 'a': 0}
        self.free_variable_name = 'time'
        self.output_names = [
            'a', 'y', 'state_variable', 'time',
            'leakage_current', 'SR_leak_current', 'membrane_voltage']
        super(TestOdeModel, self).__init__()

    def evaluate_rhs(self, t, y, ydot=np.empty(0)):
        if ydot.size == 0:
            return self.parameters[0]
        else:
            ydot[0] = self.parameters[0]

    def get_outputs(self):
        outputs = [np.array(self.parameters[self.parameter_map['a']]),
                   np.array(self.state[self.state_var_map['y']]),
                   self.state,
                   np.array(self.free_variable),
                   np.array(self.parameters[self.parameter_map['leakage_current']]),
                   np.array(self.parameters[self.parameter_map['SR_leak_current']]),
                   np.array(self.state[self.state_var_map['membrane_voltage']])]
        return outputs
