"""Copyright (c) 2005-2015, University of Oxford.
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

from ..utility.error_handling import ProtocolError

class AbstractModifier(object):
    """Base class for modifiers in the protocol language."""

    START_ONLY = 0
    EACH_LOOP = 1
    END_ONLY = 2

    def Apply(self, simul):
        """Apply this modifier to the given simulation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class SetVariable(AbstractModifier):
    """Modifier for changing the values of model variables (that were specified as protocol inputs)."""
    def __init__(self, when, variableName, valueExpr):
        self.when = when
        self.variableName = variableName
        self.valueExpr = valueExpr
        # Note that we must be modifying a model variable, so the name must be prefixed
        parts = variableName.split(':')
        if len(parts) != 2:
            raise ProtocolError("A SetVariable modifier must set a model variable")
        self.varPrefix, self.varLocalName = parts

        # Save a reference so when an instance is unpickled it can revert to Apply
        self._calledOnce = False

    # Override Object serialization methods to allow pickling with the dill module
    def __getstate__(self):
        odict = self.__dict__.copy()
        # For pickling, simulation modifiers can't save references to model environment
        if '_evaluate' in odict:
            del odict['_evaluate']
            del odict['_bindings']
        return odict
    def __setstate__(self,dict):
        self.__dict__.update(dict)
        self._calledOnce = False

    def Apply(self, simul):
        if self._calledOnce:
            return self.FastApply(simul)
        # Cache some attribute lookups locally in the class, and re-bind to use a faster method
        self._evaluate = self.valueExpr.EvaluateCompiled
        # Find the unwrapped bindings dictionary for the model prefix, in order to write there directly
        # Old slow version: simul.env.OverwriteDefinition(self.variableName, value)
        self._bindings = simul.env.unwrappedBindings.delegatees[self.varPrefix]
        self._calledOnce = True
        #self.Apply = self.FastApply
        return self.FastApply(simul)

    def FastApply(self, simul):
        """Optimised version of Apply that gets used after the first call.

        We cache most of the lookups required locally, since the simulation object we are passed will be
        the same each time.
        """
        self._bindings[self.varLocalName] = self._evaluate(simul.env)


class SaveState(AbstractModifier):
    """Modifier to cache the state of a model with an associated name."""
    def __init__(self, when, stateName):
        self.when = when
        self.stateName = stateName

    def Apply(self, simul):
        simul.model.SaveState(self.stateName)


class ResetState(AbstractModifier):
    """Modifier to restore the state of a model to a previously cached version."""
    def __init__(self, when, stateName=None):
        self.when = when
        self.stateName = stateName

    def Apply(self, simul):
        simul.model.ResetState(self.stateName)
