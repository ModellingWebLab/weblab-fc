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
        
    def DefineName(self, name, value):
        if ':' in name:
            raise ProtocolError('Names such as', name, 'with a colon are not allowed.')
        if name in self.bindings:
            raise ProtocolError(name, "is already defined as", self.bindings[name], "and may not be re-bound")
        else:
            self.bindings[name] = value
            # TODO: Give values a .unwrapped property to handle this more neatly
            if isinstance(value, V.Array):
                self.unwrappedBindings[name] = value.array
            elif isinstance(value, V.Simple):
                self.unwrappedBindings[name] = value.value
            elif isinstance(value, V.Null):
                self.unwrappedBindings[name] = None
    
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
        result = self.bindings[name]
        return result

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
        if isinstance(value, V.Array):
            unwrapped_val = value.array
        elif isinstance(value, V.Simple):
            unwrapped_val = value.value
        elif isinstance(value, V.Null):
            unwrapped_val = None
        else:
            print 'Unexpected overwrite:', self, name, value
        def find_definition(env, name):
            if name in env.bindings:
                if not env.allowOverwrite:
                    raise ProtocolError("This environment does not support overwriting mappings")
                env.bindings[name] = value
                if unwrapped_val is not False:
                    env.unwrappedBindings[name] = unwrapped_val
                return True
            else:
                parts = name.split(':', 1)
            if len(parts) == 2:
                prefix, local_name = parts
                if prefix in self.delegatees:
                    return find_definition(self.delegatees[prefix], local_name)
            if '' in self.delegatees:
                return find_definition(self.delegatees[''], name)
            return False
        if not find_definition(self, name):
            raise ProtocolError(name, "is not defined in this env or a delegating env and thus cannot be overwritten")
    
    def Clear(self):
        self.bindings.clear()
        self.unwrappedBindings.clear()
            
    def __len__(self):
        return len(self.bindings)
    
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


class DelegatingDict(dict):
    def __init__(self, *args, **kwargs):
        super(DelegatingDict, self).__init__(*args, **kwargs)
        self.delegatees = {}
        
    def __missing__(self, key):
        parts = key.split(":", 1)
        if len(parts) == 2:
            prefix, name = parts
            if prefix in self.delegatees:
                return self.delegatees[prefix][name]
        if '' in self.delegatees:
            return self.delegatees[''][key]
        raise ProtocolError("Name", key, "is not defined in env or any delegatee env")
        
    def SetDelegatee(self, delegatee, prefix):
        self.delegatees[prefix] = delegatee


class ModelWrapperEnvironment(Environment):
    class _BindingsDict(dict):
        def __init__(self, unwrapped):
            self._unwrapped = unwrapped
            
        def __getitem__(self, key):
            val = self._unwrapped[key]
            if isinstance(val, np.ndarray):
                return V.Array(val)
            else:
                return V.Simple(val)
            
        def __setitem__(self, key, value):
            pass

        def __contains__(self, key):
            return key in self._unwrapped
    
    class _UnwrappedBindingsDict(dict):
        def __init__(self, model):
            self._model = model
            
        def __getitem__(self, key):
            if key in self._model.parameterMap:
                return self._model.parameters[self._model.parameterMap[key]]
            elif key in self._model.stateVarMap:
                return self._model.state[self._model.stateVarMap[key]]
            elif key == self._model.freeVariableName:
                return self._model.freeVariable
            else:
                raise ProtocolError('Name', key, 'is not defined.')
        
        def __setitem__(self, key, value):
            if key in self._model.parameterMap:
                self._model.parameters[self._model.parameterMap[key]] = value
            elif key in self._model.stateVarMap:
                self._model.state[self._model.stateVarMap[key]] = value
            elif key == self._model.freeVariableName:
                setattr(self._model, key, value)
            else:
                raise ProtocolError('Name', key, 'is not defined.')
        
        def __contains__(self, key):
            return key in self._model.env.DefinedNames()
    
    def __init__(self, model):
        super(ModelWrapperEnvironment, self).__init__(allowOverwrite=True)
        self.model = model
        self.names = []
        self.names.extend(self.model.stateVarMap.keys())
        self.names.extend(self.model.parameterMap.keys())
        self.names.append(self.model.freeVariableName)
        self.unwrappedBindings = self._UnwrappedBindingsDict(model)
        self.bindings = self._BindingsDict(self.unwrappedBindings)
            
    def DefineName(self, name, value):
        raise ProtocolError("Defining names in a model is not allowed.")
    
    def DefinedNames(self):
        return self.names
