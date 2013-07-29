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
import Values as V
import AbstractValue as AV
import ErrorHandling
import sys
from Locatable import Locatable

AbstractValue = AV.AbstractValue
ProtocolError = ErrorHandling.ProtocolError

class Environment(object):
    nextIdent = [0]
    
    def __init__(self, allowOverwrite=False, delegatee=None):
        self.allowOverwrite = allowOverwrite
        self.bindings = DelegatingDict()
        self.unwrappedBindings = DelegatingDict()
        self.unwrappedBindings['___np'] = np
        if delegatee is not None:
            self.SetDelegateeEnv(delegatee, "")
        
    def DefineName(self, name, value):
#         if not isinstance(value, AbstractValue):
#             raise ProtocolError(value, "is not a value type")
        if ':' in name:
            raise ProtocolError('Names such as', name, 'with a colon are not allowed.')
        if name in self.bindings:
            raise ProtocolError(name, "is already defined as", self.bindings[name], "and may not be re-bound")
        else:
            self.bindings[name] = value
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
        CSP.ImportPythonImplementation()
        csp = CSP.CompactSyntaxParser
        
        parse_action = csp.expr.parseString(exprStr, parseAll=True)
        expr = parse_action[0].expr()
        return expr.Evaluate(env)
    
    def EvaluateStatement(self, stmtStr, env):
        import CompactSyntaxParser as CSP
        CSP.ImportPythonImplementation()
        csp = CSP.CompactSyntaxParser
        
        parse_action = csp.stmtList.parseString(stmtStr, parseAll=True)
        stmt_list = parse_action[0].expr()
        return env.ExecuteStatements(stmt_list)
                
    def FreshIdent(self):
        self.nextIdent[0] += 1
        return "___%d" % self.nextIdent[0] 
        
    def LookUp(self, name):
#         if name == 'oxmeta:leakage_current':
#             name = 'a'
#             #tc3:a
#         if name == 'oxmeta:membrane_voltage':
#             name = 'y'
            #tc3:y
        result = self.bindings[name]
        return result
    
    def SetDelegateeEnv(self, delegatee, prefix=""):
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
        if not self.allowOverwrite:
            raise ProtocolError("This environment does not support overwriting mappings")
        if name not in self.bindings:
            raise ProtocolError(name, "is not defined in this environment and thus cannot be overwritten")
        self.bindings[name]= value #can't use DefineName because error would be thrown for name already being in environment
        if isinstance(value, V.Array):
                self.unwrappedBindings[name] = value.array
        elif isinstance(value, V.Simple):
                self.unwrappedBindings[name] = value.value
        elif isinstance(value, V.Null):
                self.unwrappedBindings[name] = None  
            
    def Clear(self):
        self.bindings.clear()
            
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
            if key == 'leakage_current':
                key = 'a'
            if key == 'membrane_voltage':
                key = 'y'
            py_value = self._unwrapped[key]
            if isinstance(py_value, np.ndarray):
                return V.Array(py_value)
            else:
                return V.Simple(py_value)
        def __setitem__(self, key, value):
            pass
#             try:
#                 self._unwrapped[key] = value.value
#             except AttributeError:
#                 self._unwrapped[key] = value.array
        def __contains__(self, key):
            return key in ['a', 'y', 'leakage_current', 'membrane_voltage', 'time']
    
    class _UnwrappedBindingsDict(dict):
        def __init__(self, model):
            self._model = model
        def __getitem__(self, key):
            if key == 'leakage_current':
                key = 'a'
            if key == 'membrane_voltage':
                key = 'y'
            return getattr(self._model, key)
        def __setitem__(self, key, value):
            self._model.SetVariableNow(key, value)
        def __contains__(self, key):
            return key in ['a', 'y', 'leakage_current', 'membrane_voltage', 'time']
    
    def __init__(self, model):
        super(ModelWrapperEnvironment, self).__init__(allowOverwrite=True)
        self.model = model
        self.unwrappedBindings = self._UnwrappedBindingsDict(model)
        self.bindings = self._BindingsDict(self.unwrappedBindings)
            
    def DefineName(self, name, value):
        raise ProtocolError("Defining names in a model is not allowed.")
    
#     def LookUp(self, name):
#         if name == 'leakage_current':
#             name = 'a'
#         if name == 'membrane_voltage':
#             name = 'y'
#         return getattr(model, name)
#     
#     def OverwriteDefinition(self, name, value):
#         self.model.SetVariableNow(name, value.value)
        
        
 # class for bindings/unwrapped bindings called DelegatingDict(dict)
 # __missing__ method (self, key) for delegation
 # missing means key isn't found locally, so call parts=key.split(":", 1) to see if it's prefixed or not
 # if parts is length one then look up in default delegatee, so if len(parts) ==1 
 # then prefix, name = "", parts[0] else prefix,name = parts
 # just look up the prefix in self.delegatees and return value, otherwise say it's not found anywhere and raise protocolerror
 # setDelegatee(self, prefix, delegatee) will be called by set delegatee in environment class
 # __init__(self, *args, **kwargs) will need super(delegatingdict, self).init__(*args, **kwargs)
 # self.delegatees = {}
 # look up keys in both wrapped and unwrapped bindings
        