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
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGEnv.
"""
import AbstractValue
import numpy as np
import Environment as Env

class Simple(AbstractValue.AbstractValue):
    def __init__(self, value):
        self.value = float(value)
    
    @property
    def array(self):
        return np.array(self.value)
        
class Array(AbstractValue.AbstractValue):
    def __init__(self, array):
        assert isinstance(array, (np.ndarray, np.float))
        self.array = np.array(array, dtype=float, copy=False)
    
    @property
    def value(self):
        if self.array.ndim == 0:
            return self.array[()]
        else:
            raise AttributeError("An array with more than 0 dimensions cannot be treated as a single value.")
        
class Tuple(AbstractValue.AbstractValue):
    def __init__(self, *values):
        self.values = tuple(values)
        
class Null(AbstractValue.AbstractValue):
    pass

class String(AbstractValue.AbstractValue):
    def __init__(self, value):
        self.value = value
        
class DefaultParameter(AbstractValue.AbstractValue):
    pass
        
class LambdaClosure(AbstractValue.AbstractValue):
    def __init__(self, definingEnv, formalParameters, body, defaultParameters):
        self.formalParameters = formalParameters
        self.body = body
        self.defaultParameters = defaultParameters
        self.definingEnv = definingEnv
    
    def Compile(self, env, actualParameters):
        local_env = Env.Environment(delegatee=self.definingEnv)
        if len(actualParameters) < len(self.formalParameters):
            actualParameters.extend([DefaultParameter()] * (len(self.formalParameters) - len(actualParameters)))
        for i,param in enumerate(actualParameters):
            if not isinstance(param, DefaultParameter):
                local_env.DefineName(self.formalParameters[i], param)
            elif self.defaultParameters[i] is not None and not isinstance(self.defaultParameters[i], DefaultParameter):
                if isinstance(self.defaultParameters[i], Simple):
                    local_env.DefineName(self.formalParameters[i], self.defaultParameters[i])
                else:
                    raise ProtocolError("The default parameters must be simple values.")
                #check default is a protocol simple value for a compile case
            else:
                raise ProtocolError("One of the parameters is not defined and has no default value")
        if len(self.body) == 1:
            expression = self.body[0].Compile(env)
         #compile each statement in the body if there's only a return statement
         # returns compiled expression and env
        return expression,local_env
    
    def Evaluate(self, env, actualParameters):
        local_env = Env.Environment(delegatee=self.definingEnv)
        if len(actualParameters) < len(self.formalParameters):
            actualParameters.extend([DefaultParameter()] * (len(self.formalParameters) - len(actualParameters)))
        for i,param in enumerate(actualParameters):
            if not isinstance(param, DefaultParameter):
                local_env.DefineName(self.formalParameters[i], param)
            elif self.defaultParameters[i] is not None and not isinstance(self.defaultParameters[i], DefaultParameter):
                local_env.DefineName(self.formalParameters[i], self.defaultParameters[i])
                #check default is a protocol simple value for a compile case
            else:
                raise ProtocolError("One of the parameters is not defined and has no default value")
        result = local_env.ExecuteStatements(self.body, returnAllowed=True)
        return result
            
        
    
