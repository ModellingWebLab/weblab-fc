
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

class ProtocolError(Exception):
    pass

class AbstractExpression(object):
    """Base class for expressions in the protocol language."""
    def __init__(self, *children):
        """Create a new expression node, with a list of child expressions, possibly empty."""
        self.children = children

    def EvaluateChildren(self,env):
        """Evaluate our child expressions and return a list of their values."""
        childList = [child.Evaluate(env) for child in self.children]
        return childList
    
    def Evaluate(self,env):
        """Subclasses must implement this method."""
        raise NotImplementedError

class Const(AbstractExpression):
    """Class for constant value as expression."""
    def __init__(self, value):
        self.value = value
        
    def Evaluate(self,env):
        return self.value
    
class Eq(AbstractExpression):
    """Equality Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Equality operator requires 2 operands, not", len(self.children))
        try:
            result = operands[0].value == operands[1].value
        except AttributeError:
            raise ProtocolError("Equality operator requires its operands to be simple values")
        return V.Simple(result)
    
class Neq(AbstractExpression):
    """Not equal Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Not equal operator requires 2 operands, not", len(self.children))
        try:
            result = operands[0].value != operands[1].value
        except AttributeError:
            raise ProtocolError("Not equal operator requires its operands to be simple values")
        return V.Simple(result)    
    
class Lt(AbstractExpression):
    """Less than Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Less than operator requires 2 operands, not", len(self.children))
        try:
            result = operands[0].value < operands[1].value
        except AttributeError:
            raise ProtocolError("Less than operator requires its operands to be simple values")
        return V.Simple(result)        
    
class Gt(AbstractExpression):
    """Greater than Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Greater than operator requires 2 operands, not", len(self.children))
        try:
            result = operands[0].value > operands[1].value
        except AttributeError:
            raise ProtocolError("Greater than operator requires its operands to be simple values")
        return V.Simple(result) 
    
class Leq(AbstractExpression):
    """Less than or equal to Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Less than or equal to operator requires 2 operands, not", len(self.children))
        try:
            result = operands[0].value <= operands[1].value
        except AttributeError:
            raise ProtocolError("Less than or equal to operator requires its operands to be simple values")
        return V.Simple(result) 
    
class Geq(AbstractExpression):
    """Greater than or equal to Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Greater than or equal to operator requires 2 operands, not", len(self.children))
        try:
            result = operands[0].value >= operands[1].value
        except AttributeError:
            raise ProtocolError("Greater than or equal to operator requires its operands to be simple values")
        return V.Simple(result)  