
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
import Statements as S

import AbstractExpression as AE
import ErrorHandling

AbstractExpression = AE.AbstractExpression
ProtocolError = ErrorHandling.ProtocolError

class FunctionCall(AbstractExpression):
    def __init__(self, functionOrName, children):
        super(FunctionCall, self).__init__()
        if isinstance(functionOrName, str):
            self.function = NameLookUp(functionOrName)
        else:
            self.function = functionOrName
        self.children = children
        
    def Interpret(self, env):
        actual_params = self.EvaluateChildren(env)
        function = self.function.Evaluate(env)
        if not isinstance(function, V.LambdaClosure):
            raise ProtocolError(function, "is not a function")
        return function.Evaluate(env, actual_params)
    
class If(AbstractExpression):
    def __init__(self, testExpr, thenExpr, elseExpr):
        super(If, self).__init__()
        self.testExpr = testExpr
        self.thenExpr = thenExpr
        self.elseExpr = elseExpr
    
    def Interpret(self, env):
        test = self.testExpr.Evaluate(env)
        if not hasattr(test, 'value'):
            raise ProtocolError("The test in an if expression must be a Simple value or 0-d array.")
        if test.value:
            result = self.thenExpr.Evaluate(env)
        else:
            result = self.elseExpr.Evaluate(env)
        return result
    
    def GetUsedVariables(self):
        result = self.testExpr.GetUsedVariables()
        result |= self.thenExpr.GetUsedVariables()
        result |= self.elseExpr.GetUsedVariables()
        return result

      
      #compile using where
    def Compile(self):
        test = self.testExpr.Compile()
        then = self.thenExpr.Compile()
        else_ = self.elseExpr.Compile()
        return '___np.where(%s,%s,%s)' % (test, then, else_)
        
class NameLookUp(AbstractExpression):
    """Used to look up a name for a given environment"""
    def __init__(self, name):
        super(NameLookUp, self).__init__()
        self.name = name
        
    def Interpret(self, env):
        return env.LookUp(self.name)
    
    def Compile(self):
        return self.name
    
    def GetUsedVariables(self):
        return set([self.name])
    
class TupleExpression(AbstractExpression):
    def __init__(self, *children):
        super(TupleExpression, self).__init__(*children)
        if len(self.children) < 1:
            raise ProtocolError("Empty tuple expressions are not allowed")
        
    def Interpret(self, env):
        return V.Tuple(*self.EvaluateChildren(env))
        
class LambdaExpression(AbstractExpression):
    def __init__(self, formalParameters, body, defaultParameters=None):
        super(LambdaExpression, self).__init__()
        self.formalParameters = formalParameters
        self.body = body
        self.defaultParameters = defaultParameters
        
    def Interpret(self, env):
        return V.LambdaClosure(env, self.formalParameters, self.body, self.defaultParameters) 
    
    @staticmethod
    def Wrap(operator, numParams): 
        parameters = []
        look_up_list = []
        for i in range(numParams):
            parameters.append("___" + str(i))
            look_up_list.append(NameLookUp("___" + str(i))) 
        body = [S.Return(operator(*look_up_list))]
        function = LambdaExpression(parameters, body)
        return function

class Accessor(AbstractExpression):
    IS_SIMPLE_VALUE = 0
    IS_ARRAY = 1
    IS_STRING = 2
    IS_FUNCTION = 3
    IS_TUPLE = 4
    IS_NULL = 5
    IS_DEFAULT = 6 
    NUM_DIMS = 7
    NUM_ELEMENTS = 8
    SHAPE = 9
    
    def __init__(self, variableExpr, attribute):
        super(Accessor, self).__init__()
        self.variableExpr = variableExpr
        self.attribute = attribute
        
    def Interpret(self, env):
        variable = self.variableExpr.Evaluate(env)
        if self.attribute == self.IS_SIMPLE_VALUE:
            result = hasattr(variable, 'value')
        elif self.attribute == self.IS_ARRAY:
            result = hasattr(variable, 'array')
        elif self.attribute == self.IS_STRING:
            result = isinstance(variable, V.String)
        elif self.attribute == self.IS_FUNCTION:
            result = isinstance(variable, V.LambdaClosure)
        elif self.attribute == self.IS_TUPLE:
            result = isinstance(variable, V.Tuple)
        elif self.attribute == self.IS_NULL:
            result = isinstance(variable, V.Null)
        elif self.attribute == self.IS_DEFAULT:
            result = isinstance(variable, V.DefaultParameter)
        elif self.attribute == self.NUM_DIMS:
            try:
                result = variable.array.ndim
            except AttributeError:
                raise ProtocolError("Cannot get number of dimensions of type", type(variable))
        elif self.attribute == 8:
            try:
                result = variable.array.size
            except AttributeError:
                raise ProtocolError("Cannot get number of elements of type", type(variable))
        elif self.attribute == 9:
            try:
                result = V.Array(np.array(variable.array.shape))
            except AttributeError:
                raise ProtocolError("Cannot get shape of type", type(variable))
        if isinstance(result, V.Array):
            return result
        else:
            return V.Simple(result)

    