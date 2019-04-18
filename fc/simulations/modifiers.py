
from ..error_handling import ProtocolError


class AbstractModifier(object):
    """Base class for modifiers in the protocol language."""

    START_ONLY = 0
    EACH_LOOP = 1
    END_ONLY = 2

    def apply(self, simul):
        """apply this modifier to the given simulation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class set_variable(AbstractModifier):
    """Modifier for changing the values of model variables (that were specified as protocol inputs)."""

    def __init__(self, when, variableName, valueExpr):
        self.when = when
        self.variableName = variableName
        self.valueExpr = valueExpr
        # Note that we must be modifying a model variable, so the name must be prefixed
        parts = variableName.split(':')
        if len(parts) != 2:
            raise ProtocolError("A set_variable modifier must set a model variable")
        self.varPrefix, self.varLocalName = parts

        # Save a reference so when an instance is unpickled it can revert to apply
        self._calledOnce = False

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
        self._calledOnce = False

    def apply(self, simul):
        if self._calledOnce:
            return self.fast_apply(simul)
        # Cache some attribute lookups locally in the class, and re-bind to use a faster method
        self._evaluate = self.valueExpr.evaluate_compiled
        # Find the unwrapped bindings dictionary for the model prefix, in order to write there directly
        # Old slow version: simul.env.overwrite_definition(self.variableName, value)
        self._bindings = simul.env.unwrapped_bindings.delegatees[self.varPrefix]
        self._calledOnce = True
        # self.apply = self.fast_apply
        return self.fast_apply(simul)

    def fast_apply(self, simul):
        """Optimised version of apply that gets used after the first call.

        We cache most of the lookups required locally, since the simulation object we are passed will be
        the same each time.
        """
        self._bindings[self.varLocalName] = self._evaluate(simul.env)


class save_state(AbstractModifier):
    """Modifier to cache the state of a model with an associated name."""

    def __init__(self, when, stateName):
        self.when = when
        self.stateName = stateName

    def apply(self, simul):
        simul.model.save_state(self.stateName)


class reset_state(AbstractModifier):
    """Modifier to restore the state of a model to a previously cached version."""

    def __init__(self, when, stateName=None):
        self.when = when
        self.stateName = stateName

    def apply(self, simul):
        simul.model.reset_state(self.stateName)
