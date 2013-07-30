
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
import AbstractExpression as AE
import Locatable
import MathExpressions as M
from ErrorHandling import ProtocolError
import Values as V

AbstractExpression = AE.AbstractExpression


class AbstractStatement(Locatable.Locatable):
    """Base class for statements in the protocol language."""
    def Evaluate(self, env):
        raise NotImplementedError

class Assign(AbstractStatement):
    """Assign statements in the protocol language."""
    def __init__(self, names, rhs):
        super(Assign, self).__init__()
        self.names = names
        self.rhs = rhs
                
    def Evaluate(self, env):
        results = self.rhs.Evaluate(env)
        if len(self.names) > 1:
            if not isinstance(results, V.Tuple):
                raise ProtocolError("When assigning multiple names the value to assign must be a tuple.")
            env.DefineNames(self.names, results.values)
        else:
            env.DefineName(self.names[0], results)
        return V.Null()
    
class Assert(AbstractStatement):
    def __init__(self, expr):
        super(Assert, self).__init__()
        self.expr = expr
        
    def Evaluate(self, env):
        result = self.expr.Evaluate(env)
        if hasattr(result, 'value'):
            if not result.value:
                raise ProtocolError("Assertion failed.")
        else:
            raise ProtocolError("Assertion did not yield a Simple value or 0-d Array.")
        return V.Null()

class Return(AbstractStatement):
    def __init__(self, *parameters):
        super(Return, self).__init__()
        self.parameters = parameters
                
    def Evaluate(self, env):
        results = [expr.Evaluate(env) for expr in self.parameters]
        if len(results) == 0:
            return V.Null()
        elif len(results) == 1:
            return results[0]
        else:
            return V.Tuple(*results)
    
    def Compile(self, env):
        if len(self.parameters) == 1:
            if isinstance(self.parameters[0], M.Const):
                raise NotImplementedError
            expression = self.parameters[0].Compile()
        else:
            raise NotImplementedError
        return expression