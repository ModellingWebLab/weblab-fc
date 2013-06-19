
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
import numpy as np
import numexpr as ne

from ErrorHandling import ProtocolError
from AbstractExpression import AbstractExpression

class FunctionCall(AbstractExpression):
    def __init__(self, functionOrName, children):
        if isinstance(functionOrName, str):
            self.function = NameLookUp(functionOrName)
        else:
            self.function = functionOrName
        self.children = children
        
    def Evaluate(self, env):
        actual_params = self.EvaluateChildren(env)
        function = self.function.Evaluate(env)
        if not isinstance(function, V.LambdaClosure):
            raise ProtocolError(function, "is not a function")
        return function.Evaluate(env, actual_params)
    
class NameLookUp(AbstractExpression):
    """Used to look up a name for a given environment"""
    def __init__(self, name):
        self.name = name
        
    def Evaluate(self, env):
        return env.LookUp(self.name)
    
    def Compile(self):
        return self.name
    
class TupleExpression(AbstractExpression):
    def Evaluate(self, env):
        if len(self.children) < 1:
            raise ProtocolError("Empty tuple expressions are not allowed")
        return V.Tuple(*self.EvaluateChildren(env))
    
        
class LambdaExpression(AbstractExpression):
    def __init__(self, formalParameters, body, defaultParameters=None):
        self.formalParameters = formalParameters
        self.body = body
        self.defaultParameters = defaultParameters
        
    def Evaluate(self, env):
        return V.LambdaClosure(env, self.formalParameters, self.body, self.defaultParameters)  