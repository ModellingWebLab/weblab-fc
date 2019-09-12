
from ..error_handling import ProtocolError


class AbstractModifier(object):
    """Base class for modifiers in the protocol language."""

    START_ONLY = 0
    EACH_LOOP = 1
    END_ONLY = 2

    def apply(self, simul):
        """Apply this modifier to the given simulation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class SetVariable(AbstractModifier):
    """Modifier for changing the values of model variables (that were specified as protocol inputs)."""

    def __init__(self, when, variable_name, value_expr):
        self.when = when
        self.variable_name = variable_name
        self.value_expr = value_expr
        # Note that we must be modifying a model variable, so the name must be prefixed
        parts = variable_name.split(':')
        if len(parts) != 2:
            raise ProtocolError("A SetVariable modifier must set a model variable")
        self.var_prefix, self.var_local_name = parts

        # Save a reference so when an instance is unpickled it can revert to apply
        self._called_once = False

    # Override Object serialization methods to allow pickling with the dill module
    def __getstate__(self):
        odict = self.__dict__.copy()
        # For pickling, simulation modifiers can't save references to model environment
        if '_evaluate' in odict:
            del odict['_evaluate']
            del odict['_bindings']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self._called_once = False

    def apply(self, simul):
        if self._called_once:
            return self.fast_apply(simul)
        # Cache some attribute lookups locally in the class, and re-bind to use a faster method
        self._evaluate = self.value_expr.evaluate_compiled
        # Find the unwrapped bindings dictionary for the model prefix, in order to write there directly
        # Old slow version: simul.env.overwrite_definition(self.variable_name, value)
        self._bindings = simul.env.unwrapped_bindings.delegatees[self.var_prefix]
        self._called_once = True
        # self.apply = self.fast_apply
        return self.fast_apply(simul)

    def fast_apply(self, simul):
        """Optimised version of apply that gets used after the first call.

        We cache most of the lookups required locally, since the simulation object we are passed will be
        the same each time.
        """
        self._bindings[self.var_local_name] = self._evaluate(simul.env)


class SaveState(AbstractModifier):
    """Modifier to cache the state of a model with an associated name."""

    def __init__(self, when, state_name):
        self.when = when
        self.state_name = state_name

    def apply(self, simul):
        simul.model.save_state(self.state_name)


class ResetState(AbstractModifier):
    """Modifier to restore the state of a model to a previously cached version."""

    def __init__(self, when, state_name=None):
        self.when = when
        self.state_name = state_name

    def apply(self, simul):
        simul.model.reset_state(self.state_name)

