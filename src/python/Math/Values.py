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
import AbstractValue
import numpy as np
import Environment as E

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
    
    def Evaluate(self, env, actualParameters):
        local_env = E.Environment(delegatee=self.definingEnv)
        if len(actualParameters) < len(self.formalParameters):
            actualParameters.extend([DefaultParameter()] * (len(self.formalParameters) - len(actualParameters)))
        for i,param in enumerate(actualParameters):
            if not isinstance(param, DefaultParameter):
                local_env.DefineName(self.formalParameters[i], param)
            elif self.defaultParameters[i] is not None and not isinstance(self.defaultParameters[i], DefaultParameter):
                local_env.DefineName(self.formalParameters[i], self.defaultParameters[i])
            else:
                raise ProtocolError("One of the parameters is not defined and has no default value")
        result = local_env.ExecuteStatements(self.body, returnAllowed=True)
        return result
        
        
         #lambda closure is a value, functioncall and lambdaexpression are both expressions
        #closure just has information about formal parameters (names for parameters, 
        #...list of strings) so for f(a, b=1, c), the list of formal parameters are a,b,c
        # if the function f from above returns a+b+c
        # defaultparameter is its own value in values.py and is like the null
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
        # created env delegates to definingEnv and within the local env, it defines names 
        #...for local parameters and the assigned value is actual param if its 
        #...assigned or the default parameters (if no default parameter, cuz value 
        #...of default is none or default, then throw an error)
        # closure evaluate returns local_env.executestatements(body, returnAllowed=True)
            
        
    
