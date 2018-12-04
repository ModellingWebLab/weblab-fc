
"""Copyright (c) 2005-2016, University of Oxford.
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

import itertools
import sys

import numexpr as ne
import numpy as np

from .abstract import AbstractExpression
from .general import TupleExpression
from .maths import Max, Min, Plus, Times
from .. import values as V
from ...utility import environment as Env
from ...utility.error_handling import ProtocolError


class NewArray(AbstractExpression):
    """Used to create new arrays, has keyword input comprehension."""

    def __init__(self, *children, **kwargs):
        super(NewArray, self).__init__()
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

    def Interpret(self, env):
        if self.comprehension:
            return self._DoComprehension(env)
        else:
            return self._DoListMembers(env)

    def DevelopResultWithCompile(self, range_name, ranges, compiled_gen_expr, env):
        local_dict = {}
        defined_ranges = 0
        for i, name in enumerate(range_name):
            local_dict[name] = np.array(ranges[i], dtype=float)
            defined_ranges += 1
        if defined_ranges > 1:
            raise NotImplementedError

        result = ne.evaluate(compiled_gen_expr, local_dict, env.unwrappedBindings)
        return result

    def DevelopResultWithInterpret(self, range_dims, range_name, ranges, env, num_gaps, dims):
        # print 'Comprehension', self.location
        result = None
        for range_spec_indices in itertools.product(*range_dims):
             # collect everything in range_spec_indices that is a number, not a slice
            range_specs = [ranges[dim][idx]
                           for dim, idx in enumerate(range_spec_indices) if not isinstance(idx, slice)]
            sub_env = Env.Environment(delegatee=env)
            for i, range_value in enumerate(range_specs):
                sub_env.DefineName(range_name[i], V.Simple(range_value))
            sub_array = self.genExpr.Evaluate(sub_env).array
            if result is None:
                # Create result array
                if sub_array.ndim < num_gaps:
                    raise ProtocolError("The sub-array only has", sub_array.ndim,
                                        "dimensions, which is not enough to fill", num_gaps, "gaps")
                sub_array_shape = sub_array.shape
                count = 0
                for i, dimension in enumerate(dims):
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
        return np.array(result)

    def GetUsedVariables(self):
        result = super(NewArray, self).GetUsedVariables()
        if self.comprehension:
            iterator_vars = set()
            for child in self.children:
                assert isinstance(child, TupleExpression)
                assert isinstance(child.children[-1].value, V.String)
                iterator_vars |= {child.children[-1].value.value}
            result |= self.genExpr.GetUsedVariables()
            result = result - set.intersection(result, iterator_vars)
        return result

    def _DoComprehension(self, env):
        range_specs = self.EvaluateChildren(env)
        ranges = []
        range_name = []
        implicit_dim_slices = []
        implicit_dim_names = []
        explicit_dim_slices = {}
        explicit_dim_names = {}
        for spec in range_specs:
            if len(spec.values) == 4:
                dim = None
                start = self.GetValue(spec.values[0])
                step = self.GetValue(spec.values[1])
                end = self.GetValue(spec.values[2])
                if list(range(int(start), int(end), int(step))) == []:
                    raise ProtocolError("The indices you entered created an empty range")
                implicit_dim_slices.append(list(range(int(start), int(end), int(step))))
                implicit_dim_names.append(spec.values[3].value)
            elif len(spec.values) == 5:
                dim = self.GetValue(spec.values[0])
                start = self.GetValue(spec.values[1])
                step = self.GetValue(spec.values[2])
                end = self.GetValue(spec.values[3])
                if dim in explicit_dim_slices:
                    raise ProtocolError("Dimension", dim, "has already been assigned")
                if list(range(int(start), int(end), int(step))) == []:
                    raise ProtocolError("The indices you entered created an empty range")
                explicit_dim_slices[dim] = list(range(int(start), int(end), int(step)))
                explicit_dim_names[dim] = spec.values[4].value
            else:
                raise ProtocolError("Each slice must be a tuple that contains 4, or 5 values, not", len(spec.values))
        if explicit_dim_slices:
            ranges = [None] * (max(explicit_dim_slices) + 1)
            range_name = [None] * (max(explicit_dim_slices) + 1)
        for key in explicit_dim_slices:
            ranges[key] = explicit_dim_slices[key]
            range_name[key] = explicit_dim_names[key]

        num_gaps = 0
        for i, each in enumerate(ranges):
            if each is None:
                if implicit_dim_slices:
                    ranges[i] = implicit_dim_slices.pop(0)
                    range_name[i] = implicit_dim_names.pop(0)
                else:
                    ranges[i] = slice(None, None, 1)
                    num_gaps += 1

        for i, implicit_slice in enumerate(implicit_dim_slices):
            ranges.append(implicit_slice)
            range_name.append(implicit_dim_names[i])

        range_name = [_f for _f in range_name if _f]  # Remove None entries
        product = 1
        dims = []
        range_dims = []
        for i, each in enumerate(ranges):
            if isinstance(each, slice):
                dims.append(None)
                range_dims.append([slice(None, None, 1)])
            else:
                dims.append(len(each))
                range_dims.append(list(range(len(each))))
                last_spec_dim = i
        # stretch
        if len(range_name) == 1:
            names_used = self.genExpr.GetUsedVariables()
            local_names = set(range_name)
            if names_used.isdisjoint(local_names):
                repeated_array = self.genExpr.Evaluate(env).array
                shape = list(repeated_array.shape)
                shape.insert(last_spec_dim, 1)
                repeated_array = repeated_array.reshape(tuple(shape))
                reps = [1] * repeated_array.ndim
                reps[last_spec_dim] = dims[last_spec_dim]
                result = np.tile(repeated_array, tuple(reps))
                return V.Array(result)
        try:
            if set.intersection(names_used, local_names) == local_names:
                compiled_gen_expr = self.genExpr.Compile()
                compiled = True
            else:
                raise NotImplementedError
        except BaseException:
            compiled = False
        if compiled and num_gaps == 0 and len(ranges) <= 1:
            result = self.DevelopResultWithCompile(range_name, ranges, compiled_gen_expr, env)
        else:
            result = self.DevelopResultWithInterpret(range_dims, range_name, ranges, env, num_gaps, dims)
        return V.Array(result)

    def _DoListMembers(self, env):
        elements = self.EvaluateChildren(env)
        elements_arr = np.array([elt.array for elt in elements])
        return V.Array(elements_arr)


class View(AbstractExpression):
    """Take a view of an already existing array."""

    def __init__(self, array, *children):
        super(View, self).__init__()
        self.arrayExpression = array
        self.children = children

        try:
            line_profile.add_function(self.Interpret)
        except NameError:
            pass

    def GetValue(self, arg, arg_name='not dim'):
        if isinstance(arg, V.Null):
            if arg_name == 'dim':
                return V.Null()
            else:
                return None
        if isinstance(arg, V.DefaultParameter):
            return 'default'
        else:
            return int(arg.value)

    def GetArray(self, env):
        array = self.arrayExpression.Evaluate(env)
        return array.array

    def Interpret(self, env):
        # print 'View', self.location
        array = self.arrayExpression.Evaluate(env)
        if len(self.children) > array.array.ndim:
            raise ProtocolError("You entered", len(self.children),
                                "indices, but the array has", array.array.ndim, "dimensions.")
        indices = self.EvaluateChildren(env)  # list of tuples with indices
        if not isinstance(array, V.Array):
            raise ProtocolError("First argument must be an Array, not", type(array))
        # try:
        implicit_dim_slices = []
        slices = [None] * array.array.ndim
        apply_to_rest = False
        for index in indices:
            if hasattr(index, 'value'):
                dim = None
                start = end = index.value
                step = 0
            elif len(index.values) == 1:
                dim = None
                start = self.GetValue(index.values[0])
                step = 0
                end = start
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
                # if this is null, then every dim in the input array that isn't specified uses this
                dim = self.GetValue(index.values[0], 'dim')
                start = self.GetValue(index.values[1])
                step = self.GetValue(index.values[2])
                end = self.GetValue(index.values[3])
            else:
                raise ProtocolError(
                    "Each slice must be a tuple that contains 1, 2, 3 or 4 values, not", len(index.values))
            if dim == 'default':
                dim = None
            if step == 'default':
                step = 0
            if end == 'default':
                end = start

            if dim is not None and not isinstance(dim, V.Null):
                if dim > array.array.ndim - 1:
                    raise ProtocolError("Array only has", array.array.ndim, "dimensions, not",
                                        dim + 1)  # plus one to account for dimension zero
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
                    if isinstance(dim, V.Null):
                        null_slice = start
                        apply_to_rest = True
                    else:
                        implicit_dim_slices.append(start)
                else:
                    if isinstance(dim, V.Null):
                        null_slice = slice(start, end, step)
                        apply_to_rest = True
                    else:
                        implicit_dim_slices.append(slice(start, end, step))

        for i, each in enumerate(slices):
            dim_len = array.array.shape[i]
            if each is None:
                if implicit_dim_slices:
                    if isinstance(implicit_dim_slices[0], slice):
                        if implicit_dim_slices[0].start is not None:
                            if (implicit_dim_slices[0].start < -dim_len or
                                    implicit_dim_slices[0].start >= dim_len):
                                raise ProtocolError("The start of the slice is not within the range of the dimension")
                        if implicit_dim_slices[0].stop is not None:
                            if (implicit_dim_slices[0].stop < -(dim_len + 1) or
                                    implicit_dim_slices[0].stop >= (dim_len + 1)):
                                raise ProtocolError("The end of the slice is not within the range of the dimension")
                        if implicit_dim_slices[0].step is not None and implicit_dim_slices[0].stop is not None and implicit_dim_slices[0].start is not None:
                            if ((implicit_dim_slices[0].stop - implicit_dim_slices[0].start) *
                                    implicit_dim_slices[0].step <= 0):
                                raise ProtocolError("The sign of the step does not make sense")

                    slices[i] = implicit_dim_slices.pop(0)
                else:
                    if apply_to_rest:
                        slices[i] = null_slice
                    else:
                        slices[i] = slice(None, None, 1)

        try:
            view = array.array[tuple(slices)]
        except IndexError:
            raise ProtocolError("The indices must be in the range of the array")
        return V.Array(view)


class Fold(AbstractExpression):
    "Fold an array along a specified dimension using a specified function."

    def __init__(self, *children):
        super(Fold, self).__init__(*children)
        if len(self.children) < 2 or len(self.children) > 4:
            raise ProtocolError("Fold requires 2-4 inputs, not", len(self.children))

    def GetValue(self, arg):
        if isinstance(arg, V.Null):
            return None
        else:
            return int(arg.value)

    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        default_params = [V.Null(), V.Null(), V.Null(), V.Simple(operands[1].array.ndim - 1)]
        for i, oper in enumerate(operands):
            if isinstance(oper, V.DefaultParameter):
                operands[i] = default_params[i]
        if len(self.children) == 2:
            function = operands[0]
            array = operands[1].array
            initial = None
            dimension = int(array.ndim - 1)
        elif len(self.children) == 3:
            function = operands[0]
            array = operands[1].array
            initial = self.GetValue(operands[2])
            dimension = int(array.ndim - 1)
        elif len(self.children) == 4:
            function = operands[0]
            array = operands[1].array
            initial = self.GetValue(operands[2])
            dimension = self.GetValue(operands[3])
            if dimension > array.ndim:
                raise ProtocolError("Cannot operate on dimension", dimension,
                                    "because the array only has", array.ndim, "dimensions")
        if array.ndim == 0:
            raise ProtocolError('Array has zero dimensions.')
        shape = list(array.shape)

        size = shape[dimension]

        if not isinstance(function, V.LambdaClosure):
            raise ProtocolError("The function passed into fold must be a lambda expression, not", type(function))
        # if the function is plus, then do sum...etc from numpy except sum and prod in numexpr
        if len(function.body[0].parameters) == 1:
            shape[dimension] = 1
            if isinstance(function.body[0].parameters[0], Plus):
                result = ne.evaluate('sum(array, axis=' + str(dimension) + ')')
                if initial is not None:
                    result = ne.evaluate('result + initial')
                return V.Array(np.array([result]).reshape(tuple(shape)))
            if isinstance(function.body[0].parameters[0], Times):
                result = ne.evaluate('prod(array, axis=' + str(dimension) + ')')
                if initial is not None:
                    result = ne.evaluate('result * initial')
                return V.Array(np.array([result]).reshape(tuple(shape)))
            if isinstance(function.body[0].parameters[0], Max):
                if initial is not None:
                    max_array = np.maximum(initial, array)
                else:
                    max_array = array
                return V.Array(np.amax(max_array, axis=dimension).reshape(tuple(shape)))
            if isinstance(function.body[0].parameters[0], Min):
                if initial is not None:
                    min_array = np.minimum(initial, array)
                else:
                    min_array = array
                return V.Array(np.amin(min_array, axis=dimension).reshape(tuple(shape)))
        env = Env.Environment()
        result_shape = list(shape)
        result_shape[dimension] = 1
        result = np.empty(result_shape)
        dim_ranges = []

        total_so_far = initial

        for i, dim in enumerate(shape):
            if i == dimension:
                dim_ranges.append(list(range(1)))
            else:
                dim_ranges.append(list(range(dim)))

        for indices in itertools.product(*dim_ranges):
            modifiable_indices = list(indices)
            for index in range(size):
                modifiable_indices[dimension] = index
                next_index = tuple(modifiable_indices)
                if total_so_far is None:
                    total_so_far = array[next_index]
                else:
                    args = [V.Simple(total_so_far), V.Simple(array[next_index])]
                    total_so_far = function.Evaluate(env, args).value
            result[indices] = total_so_far
            total_so_far = initial

        return V.Array(result)


class Map(AbstractExpression):
    """Mapping function for n-dimensional arrays"""

    def __init__(self, functionExpr, *children):
        super(Map, self).__init__()
        self.functionExpr = functionExpr
        self.children = children
        if len(self.children) < 1:
            raise ProtocolError("Map requires at least one parameter")

    def Evaluate(self, env):
        function = self.functionExpr.Evaluate(env)
        if not isinstance(function, V.LambdaClosure):
            raise ProtocolError("Function passed is not a function")
        arrays = self.EvaluateChildren(env)

        shape = arrays[0].array.shape
        for array in arrays:
            if array.array.shape != shape:
                raise ProtocolError(array, "is not the same shape as the first array input")
        interpret = False
        try:
            expression, local_env = function.Compile(env, arrays)
        except NotImplementedError:
            interpret = True
        else:
            try:
                protocol_result = V.Array(ne.evaluate(
                    expression, local_dict=local_env.unwrappedBindings, global_dict=env.unwrappedBindings))
            except Exception:
                try:
                    protocol_result = V.Array(eval(expression, env.unwrappedBindings, local_env.unwrappedBindings))
                except Exception:
                    interpret = True
        if interpret:
            protocol_result = self.Interpret(env, arrays, function)
        return protocol_result

    def Interpret(self, env, arrays, function):
        result = np.empty_like(arrays[0].array)
        dim_range = []
        shape = arrays[0].array.shape
        for dim in shape:
            dim_range.append(list(range(dim)))
        for index in itertools.product(*dim_range):
            function_inputs = []
            for array in arrays:
                function_inputs.append(V.Simple(float(array.array[index])))
            fn_result = function.Evaluate(env, function_inputs)
            try:
                result[index] = fn_result.value
            except AttributeError:
                raise ProtocolError("A mapped function must always return simple values, not " + str(fn_result))
        protocol_result = V.Array(result)

        return protocol_result


class Find(AbstractExpression):
    """Find function for n-dimensional arrays."""

    def __init__(self, operandExpr):
        super(Find, self).__init__()
        self.operandExpr = operandExpr

    def Interpret(self, env):
        operand = self.operandExpr.Evaluate(env)
        if not isinstance(operand, V.Array) or operand.array.ndim == 0:
            raise ProtocolError("Operand for find must be a non-generate Array, not " + str(operand))
        return V.Array(np.transpose(np.nonzero(operand.array)))


class Index(AbstractExpression):
    """Index function for n-dimensional arrays."""

    def __init__(self, *children):
        super(Index, self).__init__(*children)
        if len(self.children) < 2 or len(self.children) > 6:
            raise ProtocolError("Index requires 2-6 operands, not", len(self.children))

    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        default_params = [None, None, V.Simple(operands[0].array.ndim - 1),
                          V.Simple(0), V.Simple(0), V.Simple(sys.float_info.max)]
        for i, oper in enumerate(operands):
            if isinstance(oper, V.DefaultParameter):
                operands[i] = default_params[i]
        if len(self.children) == 2:
            operand = operands[0]
            indices = operands[1]
            dim = V.Simple(operand.array.ndim - 1)
            shrink = V.Simple(0)
            pad = V.Simple(0)
            pad_value = sys.float_info.max
        elif len(self.children) == 3:
            operand = operands[0]
            indices = operands[1]
            dim = operands[2]
            shrink = V.Simple(0)
            pad = V.Simple(0)
            pad_value = sys.float_info.max
        elif len(self.children) == 4:
            operand = operands[0]
            indices = operands[1]
            dim = operands[2]
            shrink = operands[3]
            pad = V.Simple(0)
            pad_value = sys.float_info.max
        elif len(self.children) == 5:
            operand = operands[0]
            indices = operands[1]
            dim = operands[2]
            shrink = operands[3]
            pad = operands[4]
            pad_value = sys.float_info.max
        elif len(self.children) == 6:
            operand = operands[0]
            indices = operands[1]
            dim = operands[2]
            shrink = operands[3]
            pad = operands[4]
            pad_value = operands[5].value

        # check for errors in inputs
        if not isinstance(operand, V.Array) or not isinstance(indices, V.Array):
            raise ProtocolError("The first two inputs should be Arrays.")
        if indices.array.ndim != 2:
            raise ProtocolError("The dimension of the indices array must be 2, not", indices.array.ndim)
        if not hasattr(dim, 'value'):
            raise ProtocolError("The dimension input should be a simple value, not a", type(dim))
        if dim.value >= operand.array.ndim:
            raise ProtocolError("The operand to index has", operand.array.ndim,
                                "dimensions, so it cannot be folded along dimension", dim.value)
        if not hasattr(shrink, 'value'):
            raise ProtocolError("The shrink input should be a simple value, not a", type(shrink))
        if not hasattr(pad, 'value'):
            raise ProtocolError("The pad input should be a simple value, not a", type(pad))
        if shrink.value != 0 and pad.value != 0:
            raise ProtocolError("You cannot both pad and shrink!")
        if not hasattr(pad, 'value'):
            raise ProtocolError("The pad_value input should be a simple value, not a", type(pad_value))

        dim_val = int(dim.value)
        shape = list(operand.array.shape)
        num_entries = indices.array.shape[0]
        shape[dim_val] = 1
        extents = np.zeros(tuple(shape), dtype=float)
        for index in indices.array:
            extents_index = list(index)
            extents_index[dim_val] = 0
            extents[tuple(extents_index)] += 1
        max_extent = np.amax(extents)
        min_extent = np.amin(extents)
        if min_extent == 0 and pad.value == 0 or (min_extent != max_extent and shrink.value == 0 and pad.value == 0):
            raise ProtocolError("Cannot index if the result is irregular (extent ranges from",
                                min_extent, "to", max_extent, ").")
        if pad.value != 0:
            extent = max_extent
        else:
            extent = min_extent
        shape[dim_val] = extent

        result = np.empty(shape)

        if pad != 0:
            result.fill(pad_value)
        if shrink.value + pad.value < 0:
            begin = num_entries - 1
            end = -1
            move = -1
        else:
            begin = 0
            end = num_entries
            move = 1

        # The next_index array keeps track of how far along dimension dim_val we
        # should put the next kept entry at each location
        shape[dim_val] = 1
        next_index = np.zeros(shape)
        for i in range(begin, end, move):
            idxs = list(indices.array[i])
            value = operand.array[tuple(idxs)]
            # Figure out where to put this value in the result
            idxs[dim_val] = 0
            this_next_index = next_index[tuple(idxs)]
            if this_next_index < extent:
                if shrink.value + pad.value < 0:
                    idxs[dim_val] = extent - this_next_index - 1
                else:
                    idxs[dim_val] = this_next_index
                result[tuple(idxs)] = value
                idxs[dim_val] = 0
                next_index[tuple(idxs)] += 1

        return V.Array(result)
