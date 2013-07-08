
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
from Locatable import Locatable

class AbstractExpression(Locatable):
    """Base class for expressions in the protocol language."""
    
    def __init__(self, *children):
        """Create a new expression node, with a list of child expressions, possibly empty."""
        self.children = children
        self._compiled = None

    def EvaluateChildren(self, env):
        """Evaluate our child expressions and return a list of their values."""
        childList = [child.Evaluate(env) for child in self.children]
        return childList
    
    def Interpret(self, env):
        """Old evaluate method. Called if numexpr and numpy can't evaluate string from compile"""
        raise NotImplementedError
    
    def Compile(self):
        raise NotImplementedError
    
    def Evaluate(self, env):
        """Subclasses must implement this method."""
        # try self.Compile(), if works, try ne.evaluate then eval; if compile fails straight to Interpret
        try:
            compiled = self.Compile()
            try:
                results = V.Array(ne.evaluate(compiled, local_dict=env.unwrappedBindings)) 
            except:
                results = V.Array(eval(compiled, globals(), env.unwrappedBindings))               
        except:
            results = self.Interpret(env)
        return results
    
    