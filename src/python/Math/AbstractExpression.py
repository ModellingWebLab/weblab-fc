
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
import numexpr as ne
import numpy
import Locatable
import Values as V


class AbstractExpression(Locatable.Locatable):
    """Base class for expressions in the protocol language."""
    
    def __init__(self, *children):
        """Create a new expression node, with a list of child expressions, possibly empty."""
        super(AbstractExpression, self).__init__()
        self.children = children

#        try:
#            line_profile.add_function(self.Evaluate)
#        except NameError:
#            pass

    @property
    def compiled(self):
        """Compile this expression and cache the result."""
        try:
            return self._compiled
        except AttributeError:
            # We haven't called Compile yet; cache the result
            c = self._compiled = self.Compile()
            return c

    def EvaluateChildren(self, env):
        """Evaluate our child expressions and return a list of their values."""
        childList = [child.Evaluate(env) for child in self.children]
        return childList
    
    def Interpret(self, env):
        """Evaluate this expression by interpreting the expression tree.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def Compile(self):
        """Create a string representation of this expression for evaluation by numexpr or numpy.
        
        Evaluation of the generated string is only guaranteed to give correct results (as defined
        by the Interpret method) in certain array contexts, for instance maps and array comprehensions.
        """
        raise NotImplementedError
    
    def GetUsedVariables(self):
        """Get the set of (non-local) identifiers referenced within this expression."""
        result = set()
        for child in self.children:
            result |= child.GetUsedVariables()
        return result
    
    def Evaluate(self, env):
        """We always default to using Interpret for evaluating standalone expressions."""
        return self.Interpret(env)
