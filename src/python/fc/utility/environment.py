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

from .error_handling import ProtocolError
from ..language import values as V


class Environment(object):
    """Base class for environments in the protocol language."""
    next_ident = [0]

    def __init__(self, allowOverwrite=False, delegatee=None):
        self.allowOverwrite = allowOverwrite
        self.bindings = DelegatingDict()
        self.unwrappedBindings = DelegatingDict()
        self.unwrappedBindings['___np'] = np
        self.delegatees = {}
        if delegatee is not None:
            self.SetDelegateeEnv(delegatee, "")

        try:
            line_profile.add_function(self.DefineName)
        except NameError:
            pass

    def DefineName(self, name, value):
        if ':' in name:
            raise ProtocolError('Names such as', name, 'with a colon are not allowed.')
        if name in self.bindings:
            raise ProtocolError(name, "is already defined as", self.bindings[name], "and may not be re-bound")
        else:
            self.bindings[name] = value
            self.unwrappedBindings[name] = value.unwrapped

    def DefineNames(self, names, values):
        for i, name in enumerate(names):
            self.DefineName(name, values[i])

    def EvaluateExpr(self, exprStr, env):
        import CompactSyntaxParser as CSP
        csp = CSP.CompactSyntaxParser

        parse_action = csp.expr.parseString(exprStr, parseAll=True)
        expr = parse_action[0].expr()
        return expr.Evaluate(env)

    def EvaluateStatement(self, stmtStr, env):
        import CompactSyntaxParser as CSP
        csp = CSP.CompactSyntaxParser

        parse_action = csp.stmtList.parseString(stmtStr, parseAll=True)
        stmt_list = parse_action[0].expr()
        return env.ExecuteStatements(stmt_list)

    @staticmethod
    def FreshIdent():
        Environment.next_ident[0] += 1
        return "___%d" % Environment.next_ident[0]

    def LookUp(self, name):
        return self.bindings[name]

    def SetDelegateeEnv(self, delegatee, prefix=""):
        # TODO
#        if prefix in self.delegatees:
#            raise ProtocolError("Tried to assign multiple delegatee environments to the same prefix:", prefix)
        self.delegatees[prefix] = delegatee
        self.bindings.SetDelegatee(delegatee.bindings, prefix)
        self.unwrappedBindings.SetDelegatee(delegatee.unwrappedBindings, prefix)

    def Merge(self, env):
        self.DefineNames(env.bindings.keys(), env.bindings.values())

    def Remove(self, name):
        if not self.allowOverwrite:
            raise ProtocolError("This environment does not support overwriting mappings")
        if name not in self.bindings:
            raise ProtocolError(name, "is not defined in this environment and thus cannot be removed")
        del self.bindings[name]
        del self.unwrappedBindings[name]

    def OverwriteDefinition(self, name, value):
        def find_definition(env, name):
            if name in env.bindings: ## ~15% of function time
                if not env.allowOverwrite:
                    raise ProtocolError("This environment does not support overwriting mappings")
                env.bindings[name] = value ## ~8% of time
                env.unwrappedBindings[name] = value.unwrapped ## ~45% of function time
                return True
            parts = name.split(':', 1)
            if len(parts) == 2:
                prefix, local_name = parts
                if prefix in self.delegatees:
                    return find_definition(self.delegatees[prefix], local_name)
            if '' in self.delegatees:
                return find_definition(self.delegatees[''], name)
            return False
#         try:
#             line_profile.add_function(find_definition)
#         except NameError:
#             pass
        if not find_definition(self, name):
            raise ProtocolError(name, "is not defined in this env or a delegating env and thus cannot be overwritten")

    def Clear(self):
        self.bindings.clear()
        self.unwrappedBindings.clear()

    def DefinedNames(self):
        return self.bindings.keys()

    def ExecuteStatements(self, statements, returnAllowed=False):
        result = V.Null()
        for statement in statements:
            result = statement.Evaluate(self)
            if not isinstance(result, V.Null) and result is not None:
                if returnAllowed == True:
                    break
                else:
                    raise ProtocolError("Return statement not allowed outside of function")
        return result

    # Methods to make an Environment behave a little like a (read-only) dictionary

    def __len__(self):
        return len(self.bindings)

    def __getitem__(self, key):
        return self.LookUp(key)

    def __contains__(self, key):
        return key in self.bindings

    def __iter__(self):
        """Return an iterator over the names defined in the environment."""
        return iter(self.bindings)


class DelegatingDict(dict):
    def __init__(self, *args, **kwargs):
        super(DelegatingDict, self).__init__(*args, **kwargs)
        self.delegatees = {}
        from ..language.expressions import NameLookUp
        self._marker = NameLookUp.PREFIXED_NAME
        self._marker_len = len(self._marker)

    def __missing__(self, key):
        if key.startswith(self._marker):
            parts = key[self._marker_len:].split(self._marker, 1)
        else:
            parts = key.split(':', 1)
        if len(parts) == 2:
            prefix, name = parts
            if prefix in self.delegatees:
                return self.delegatees[prefix][name]
        if '' in self.delegatees:
            return self.delegatees[''][key]
        raise KeyError("Name", key, "is not defined in env or any delegatee env")

    def SetDelegatee(self, delegatee, prefix):
        self.delegatees[prefix] = delegatee


class ModelWrapperEnvironment(Environment):
    """This environment subclass provides access to a model's variables using the Environment interface.

    It supports variable lookups and the information methods, but doesn't allow defining new names.
    OverwriteDefinition must be used to change an existing variable's value.
    """

    class _BindingsDict(dict):
        """A dictionary subclass wrapping the protocol language versions of a model's variables."""
        def __init__(self, unwrapped):
            self._unwrapped = unwrapped
            try:
                line_profile.add_function(self.__getitem__)
            except NameError:
                pass

        def __getitem__(self, key):
            val = self._unwrapped[key] ## ~61%
            if isinstance(val, np.ndarray):
                return V.Array(val)
            else:
                return V.Simple(val) ## ~21%

        def __setitem__(self, key, value):
            pass

        def __contains__(self, key):
            return key in self._unwrapped

    class _UnwrappedBindingsDict(dict):
        """A dictionary subclass wrapping the Python versions of a model's variables.

        TODO: look at the efficiency of get/set methods, and whether these matter for overall performance (c.f. #2459).
        """
        class _FreeVarList(list):
            """A single element list for wrapping the model's free variable."""
            def __init__(self, model):
                self._model = model
            def __getitem__(self, key):
                return self._model.freeVariable
            def __setitem__(self, key, value):
                setattr(self._model, key, value)

        def __init__(self, model):
            self._model = model
            # Make the underlying dict store a map from name to (vector, index) for fast lookup
            self._freeVars = self._FreeVarList(model)
            for key in model.parameterMap:
                dict.__setitem__(self, key, (model.parameters, model.parameterMap[key]))
            for key in model.stateVarMap:
                dict.__setitem__(self, key, (model.state, model.stateVarMap[key]))
            dict.__setitem__(self, model.freeVariableName, (self._freeVars, 0))

            try:
                line_profile.add_function(self.__getitem__)
                line_profile.add_function(self.__setitem__)
            except NameError:
                pass

        def __getitem__(self, key):
            # TODO: Catch KeyError?
            vector, index = dict.__getitem__(self, key)
            return vector[index]

        def __setitem__(self, key, value):
            # TODO: Catch KeyError?
            vector, index = dict.__getitem__(self, key)
            if vector[index] != value:
                self._model.dirty = True
                vector[index] = value

    def __init__(self, model):
        super(ModelWrapperEnvironment, self).__init__(allowOverwrite=True)
        self.model = model
        self.unwrappedBindings = self._UnwrappedBindingsDict(model)
        self.bindings = self._BindingsDict(self.unwrappedBindings)
        self.names = self.unwrappedBindings.keys()

    def DefineName(self, name, value):
        raise ProtocolError("Defining names in a model is not allowed.")

    def Remove(self, name):
        raise ProtocolError("Removing names from a model is not allowed.")

    def DefinedNames(self):
        return self.names
