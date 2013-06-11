
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
import Environment as E
import itertools
from operator import mul

from AbstractExpression import AbstractExpression
from ErrorHandling import ProtocolError

class NewArray(AbstractExpression):
    """Used to create new arrays."""
    def __init__(self, *children, **kwargs):
        self.comprehension = kwargs.get('comprehension', False)
        if self.comprehension:
            self.children = children[1:]
            self.genExpr = children[0]
        else:
            self.children = children
        
    def GetValue(self, arg):
        if isinstance(arg, V.Null):
            return None
        else:
            return int(arg.value)
            
    def Evaluate(self, env):
        if self.comprehension:
            return self._DoComprehension(env)
        else:
            return self._DoListMembers(env)
    
    def _DoComprehension(self, env):
        rangeSpecs = self.EvaluateChildren(env)
        ranges = []
        range_name = []
        implicit_dim_slices = []
        implicit_dim_names = []
        explicit_dim_slices = {}
        explicit_dim_names = {}
        for spec in rangeSpecs:
            if len(spec.values) == 4:
                dim = None
                start = self.GetValue(spec.values[0])
                step = self.GetValue(spec.values[1])
                end = self.GetValue(spec.values[2])
                if range(int(start), int(end), int(step)) == []:
                    raise ProtocolError("The indices you entered created an empty range")
                implicit_dim_slices.append(range(int(start), int(end), int(step)))
                implicit_dim_names.append(spec.values[3].value)
            elif len(spec.values) == 5:
                dim = self.GetValue(spec.values[0])
                start = self.GetValue(spec.values[1])
                step = self.GetValue(spec.values[2])
                end = self.GetValue(spec.values[3])
                if dim in explicit_dim_slices:
                    raise ProtocolError("Dimension", dim, "has already been assigned")
                if range(int(start), int(end), int(step)) == []:
                    raise ProtocolError("The indices you entered created an empty range")
                explicit_dim_slices[dim] = range(int(start), int(end), int(step))
                explicit_dim_names[dim] = spec.values[4].value
            else:
                raise ProtocolError("Each slice must be a tuple that contains 4, or 5 values, not", len(spec.values))
        if explicit_dim_slices:
            ranges = [None] * (max(explicit_dim_slices) + 1)
            range_name = [None] * (max(explicit_dim_slices) + 1)
        for key in explicit_dim_slices:
            ranges[key] = explicit_dim_slices[key]
            range_name[key] = explicit_dim_names[key]
        
        numGaps = 0
        for i,each in enumerate(ranges):
            if each is None:
                if implicit_dim_slices:
                    ranges[i] = implicit_dim_slices.pop(0)
                    range_name[i] = implicit_dim_names.pop(0)
                else:
                    ranges[i] = slice(None, None, 1)
                    numGaps += 1
                    
        for i,implicit_slice in enumerate(implicit_dim_slices):
            ranges.append(implicit_slice)
            range_name.append(implicit_dim_names[i])
        
        range_name = filter(None, range_name) # Remove None entries
                    
        product = 1
        dims = []
        rangeDims = []
        for each in ranges:
            if isinstance(each, slice):
                dims.append(None)
                rangeDims.append([slice(None, None, 1)])
            else:
                dims.append(len(each))
                rangeDims.append(range(len(each)))
        result = None
        for range_spec_indices in itertools.product(*rangeDims):
             # collect everything in range_spec_indices that is a number, not a slice
            range_specs = [ranges[dim][idx]
                           for dim, idx in enumerate(range_spec_indices) if not isinstance(idx, slice)]
            sub_env = E.Environment(delegatee=env)
            for i, range_value in enumerate(range_specs):
                sub_env.DefineName(range_name[i], V.Simple(range_value))
            sub_array = self.genExpr.Evaluate(sub_env).array
            if result is None:
                # Create result array
                if sub_array.ndim < numGaps:
                    raise ProtocolError("The sub-array only has", sub_array.ndim, 
                                        "dimensions, which is not enough to fill", numGaps, "gaps")
                sub_array_shape = sub_array.shape
                count = 0
                for i,dimension in enumerate(dims):
                    if dimension is None:
                        dims[i] = sub_array_shape[count]
                        count += 1
                dims.extend(sub_array_shape[count:])
                result = np.empty(tuple(dims), dtype=float)
            # Check sub_array shape
            if sub_array.shape != sub_array_shape:
                raise ProtocolError("The given sub-array has shape:", sub_array.shape, 
                                    "when it should be", sub_array_shape)
            result[tuple(range_spec_indices)] = sub_array
        return V.Array(np.array(result))
    
        
    def _DoListMembers(self, env):
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
            return int(arg.value)
        
    def Evaluate(self, env):
        array = self.arrayExpression.Evaluate(env)
        if not isinstance(array, V.Array):
            raise ProtocolError("First argument must be of type Values.Array")
        if len(self.children) > array.array.ndim: # check to make sure indices = number of dimensions
            raise ProtocolError("You entered", len(self.children), "indices, but the array has", array.array.ndim, "dimensions.")
      
        indices = self.EvaluateChildren(env) # list of tuples with indices
        
        #try:
        implicit_dim_slices = []
        slices = [None] * array.array.ndim
                
        for index in indices:
            if len(index.values) == 1:
                dim = None
                start = self.GetValue(index.values[0]) # if isinstance(arg, Null) return None else arg.value
                step = None
                end = start + 1
            elif len(index.values) == 2:
                dim = None
                start = self.GetValue(index.values[0])
                step = None
                end = self.GetValue(index.values[1])
            elif len(index.values) == 3:
                dim = None
                start = self.GetValue(index.values[0])
                step = self.GetValue(index.values[1])
                end = self.GetValue(index.values[2])
            elif len(index.values) == 4:
                dim = self.GetValue(index.values[0])
                start = self.GetValue(index.values[1])
                step = self.GetValue(index.values[2])
                end = self.GetValue(index.values[3])
            else:
                raise ProtocolError("Each slice must be a tuple that contains 1, 2, 3 or 4 values, not", len(index.values))
            
            
            if dim != None:
                if dim > array.array.ndim - 1:
                    raise ProtocolError("Array only has", array.array.ndim, "dimensions, not", dim + 1) # plus one to account for dimension zero
                if step == 0:
                    if start != end:
                        raise ProtocolError("Step is zero and start does not equal end")
                    slices[int(dim)] = start
                else:
                    slices[int(dim)] = slice(start, end, step)
            else:
                if step == 0:
                    if start != end:
                        raise ProtocolError("Step is zero and start does not equal end")
                    implicit_dim_slices.append(start)
                else:
                    implicit_dim_slices.append(slice(start, end, step))
            
        for i, each in enumerate(slices):
            dimLen = array.array.shape[i]
            if each is None:
                if implicit_dim_slices:
                    if isinstance(implicit_dim_slices[0], slice):
                        if implicit_dim_slices[0].start is not None:
                            if abs(implicit_dim_slices[0].start + 1) > dimLen:
                                raise ProtocolError("The start of the slice is not within the range of the dimension")
                        if implicit_dim_slices[0].stop is not None:
                            if abs(implicit_dim_slices[0].stop + 1) > dimLen:
                                raise ProtocolError("The end of the slice is not within the range of the dimension")
                        if implicit_dim_slices[0].step is not None and implicit_dim_slices[0].stop is not None and implicit_dim_slices[0].start is not None:
                            if (implicit_dim_slices[0].stop - implicit_dim_slices[0].start) * implicit_dim_slices[0].step <= 0:
                                raise ProtocolError("The sign of the step does not make sense")
                        
                    slices[i] = implicit_dim_slices.pop(0)
                else:
                    slices[i] = slice(None, None, 1)

        try:
            view = array.array[tuple(slices)]
        except IndexError:
            raise ProtocolError("The indices must be in the range of the array")
        return V.Array(view)
        
        
    