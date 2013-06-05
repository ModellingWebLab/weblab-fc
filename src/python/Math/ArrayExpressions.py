
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
import MathExpressions as M
import numpy as np

from AbstractExpression import AbstractExpression
from ErrorHandling import ProtocolError

class NewArray(AbstractExpression):
    """Used to create new arrays."""
    def Evaluate(self, env):
        elements = self.EvaluateChildren(env)
        elementsArr = np.array([elt.array for elt in elements])
        return V.Array(elementsArr)
    
class View(AbstractExpression):
    def __init__(self, array, *children):     
        self.arrayExpression = array
        self.children = children
        
    def GetValue(self, arg):
        if isinstance(arg, V.Null):
            return None
        else:
            return arg.value
        
    def Evaluate(self, env):
        array = self.arrayExpression.Evaluate(env)
        if not isinstance(array, V.Array):
            raise ProtocolError("First argument must be of type Values.Array")
      
        indices = self.EvaluateChildren(env) # list of tuples with indices
        #if len(indices) > self.arrayExpression.Evaluate(env).array.ndim: # check to make sure indices = number of dimensions
         #   raise ProtocolError("You entered", len(indices), "indices, but the array has", self.array.ndim, "dimensions.")
        #try:
        slices = []
        for index in indices:
            if len(index.values) == 1:
                start = self.GetValue(index.values[0]) # if isinstance(arg, Null) return None else arg.value
                step = None
                end = start + 1
            elif len(index.values) == 2:
                start = self.GetValue(index.values[0])
                step = None
                end = self.GetValue(index.values[1])
            elif len(index.values) == 3:
                start = self.GetValue(index.values[0])
                step = self.GetValue(index.values[1])
                end = self.GetValue(index.values[2])
            else:
                raise ProtocolError("Each slice must be a tuple that contains 1, 2, or 3 values, not", len(index))
            if step == 0:
                slices.append(start)
            else:
                slices.append(slice(start, end, step))
        print type(array.array)
        view = array.array[tuple(slices)]
        print type(view)
        #except IndexError: # make sure indices don't go out of range
        #    raise ProtocolError("The indices for the view must be in the range of the array") # see if there are two or three elements in the tuple and return the proper array for each using slicing
        return V.Array(view)
        
        
    