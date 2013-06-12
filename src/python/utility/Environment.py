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
import Values as V
from ErrorHandling import ProtocolError
from AbstractValue import AbstractValue

class Environment(object):
    nextIdent = [0]
    
    def __init__(self, allowOverwrite=False, delegatee=None):
        self.allowOverwrite = allowOverwrite
        self.bindings = {}
        self.delegates = {}
        self.delegates[""] = delegatee
        
    def DefineName(self, name, value):
        if not isinstance(value, AbstractValue):
            raise ProtocolError(value, "is not a value type")
        if name in self.bindings:
            raise ProtocolError(name, "is already defined and may not be re-bound")
        else:
            self.bindings[name] = value
    
    def DefineNames(self, names, values):
        for i, name in enumerate(names):
            self.DefineName(name, values[i])
                
    def FreshIdent(self):
        self.nextIdent[0] += 1
        return "~%d" % self.nextIdent[0] 
        
    def LookUp(self, name):
        if name in self.bindings:
            result = self.bindings[name]
        elif len(self.delegates) != 1 or self.delegates[""] is not None:
            for val in self.delegates.values():
                if val is not None:
                    result = val.LookUp(name)
        else:
            raise ProtocolError("The name", name, "does not exist in the environment")
        return result
    
    def SetDelegateeEnv(self, delegatee, prefix):
        self.delegates[prefix] = delegatee
        
    def Merge(self, env):
        self.DefineNames(env.bindings.keys(), env.bindings.values())
            
    def Remove(self, name):
        if not self.allowOverwrite:
            raise ProtocolError("This environment does not support overwriting mappings")
        if name not in self.bindings:
            raise ProtocolError(name, "is not defined in this environment and thus cannot be removed")
        del (self.bindings[name])
        
    def OverwriteDefinition(self, name, value):
        if not self.allowOverwrite:
            raise ProtocolError("This environment does not support overwriting mappings")
        if name not in self.bindings:
            raise ProtocolError(name, "is not defined in this environment and thus cannot be overwritten")
        self.bindings[name]= value #can't use DefineName because error would be thrown for name already being in environment
            
    def Clear(self):
        self.bindings.clear()
            
    def __len__(self):
        return len(self.bindings)
    
    def DefinedNames(self):
        return self.bindings.keys()
    
    def ExecuteStatements(self, statements, returnAllowed=False):
        for statement in statements:
            result = statement.Evaluate(self)
            if not isinstance(result, V.Null) and result is not None:
                if returnAllowed == True:
                    break
                else:
                    raise ProtocolError("Return statement not allowed outside of function")           
        return result
            
        # evaluate each statement until one is null and returns are allowed, then return that value, otherwise returns null
        # reassign result to the next statement evaluated and break when its null
        
        #lambda closure is a value, functioncall and lambdaexpression are both expressions
        #closure just has information about formal parameters (names for parameters, list of strings) so for f(a, b=1, c), the list of formal parameters are a,b,c
        # if the function f from above returns a+b+c
        # default parameters list would be 
        # defaultparameter is its only value in values.py and is like the null
        # you can call the f function like f(3,defaultparameter,1)
        # default parameters here would be [None, v.simple(1), None] or [Default, v.simple(1), default]
        # body = [statements]
        # definingEnv is an environment and is env that function is evaluated in
        # in evaluate for lambdaexpression, you return lambdaclosure(..., env) and so closure stores this env as definingEnv
        # lambda expression just takes formalparams, body, default params and returns lambda closure with those things
        # in __init for lamexpr, if isinstance(body, abstractexpression) then body = [returnstatement(body)]
        # function call takes in function to call (an expression), and the arguments (parameters) which is a list of expressions
        # function call in its evaluate method just calls closure = func.evaluate (the func that is passed in), test to make sure it returns a closure, if it doesn't then error because its not a function
        # function call evaluates children to get the parameter values and calls closure (from above, closure.evaluate(parameters
        # closure evaluate just takes in actual params as a list of values which have already been evaluated
        # evaluate method in closure: sets up env to execute statements in body
        # execute body is just calling execute statements on environment created just before
        # created env delegates to definingEnv and within the local env, it defines names for local parameters and the assigned value is actual param if its assigned or the default parameters (if no default parameter, cuz value of default is none or default, then throw an error)
        # closure evaluate returns local_env.executestatements(body, returnAllowed=True)
            
        