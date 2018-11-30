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

import numpy as np

from ..utility.error_handling import ProtocolError


class AbstractValue(object):
    """Base class for values in the protocol language."""
    def __init__(self, units=None):
        self.units = units

    @property
    def unwrapped(self):
        """Return the underlying Python value."""
        return None
    
    def __str__(self):
        """Return the string representation of the underlying Python value."""
        return repr(self.unwrapped)


class Simple(AbstractValue):
    """Simple value class in the protocol language for numbers."""
    def __init__(self, value):
        self.value = float(value)

    @property
    def array(self):
        """View this number as a 0-d array."""
        return np.array(self.value)
    
    @property
    def unwrapped(self):
        return self.value


class Array(AbstractValue):
    """Class in the protocol language for arrays."""
    def __init__(self, array):
        assert isinstance(array, (np.ndarray, np.float))
        self.array = np.array(array, dtype=float, copy=False)

    @property
    def value(self):
        if self.array.ndim == 0:
            return self.array[()]
        else:
            raise AttributeError("An array with", self.array.ndim, "dimensions cannot be treated as a single value.")
    
    @property
    def unwrapped(self):
        return self.array


class Tuple(AbstractValue):
    """Tuple class in the protocol language."""
    def __init__(self, *values):
        self.values = tuple(values)

    @property
    def unwrapped(self):
        return self.values


class Null(AbstractValue):
    """Null class in the protocol language."""
    def __str__(self):
        return "null"


class String(AbstractValue):
    """String class in the protocol language."""
    def __init__(self, value):
        self.value = value

    @property
    def unwrapped(self):
        return self.value


class DefaultParameter(AbstractValue):
    """Class in protocol language used for default values."""
    def __str__(self):
        return "default"


class LambdaClosure(AbstractValue):
    """Class for functions in the protocol language."""
    def __init__(self, definingEnv, formalParameters, body, defaultParameters):
        self.formalParameters = formalParameters
        self.body = body
        self.defaultParameters = defaultParameters
        self.definingEnv = definingEnv

    def __str__(self):
        """Return a string representation of this function."""
        return "function" + str(tuple(self.formalParameters))

    def Compile(self, env, actualParameters):
        from ..utility.environment import Environment
        local_env = Environment(delegatee=self.definingEnv)
        params = actualParameters[:]
        if len(params) < len(self.formalParameters):
            params.extend([DefaultParameter()] * (len(self.formalParameters) - len(params)))
        for i,param in enumerate(params):
            if not isinstance(param, DefaultParameter):
                local_env.DefineName(self.formalParameters[i], param)
            elif self.defaultParameters[i] is not None and not isinstance(self.defaultParameters[i], DefaultParameter):
                if not hasattr(self.defaultParameters[i], 'value'):
                    raise NotImplementedError
                local_env.DefineName(self.formalParameters[i], self.defaultParameters[i])
            else:
                raise ProtocolError("One of the parameters is not defined and has no default value")
        if len(self.body) == 1:
            expression = self.body[0].Compile(env)
        return expression, local_env

    def Evaluate(self, env, actualParameters):
        from ..utility.environment import Environment
        local_env = Environment(delegatee=self.definingEnv)
        if len(actualParameters) < len(self.formalParameters):
            actualParameters.extend([DefaultParameter()] * (len(self.formalParameters) - len(actualParameters)))
        for i,param in enumerate(actualParameters):
            if not isinstance(param, DefaultParameter):
                local_env.DefineName(self.formalParameters[i], param)
            elif self.defaultParameters[i] is not None:
                local_env.DefineName(self.formalParameters[i], self.defaultParameters[i])
            else:
                raise ProtocolError("One of the parameters is not defined and has no default value")
        result = local_env.ExecuteStatements(self.body, returnAllowed=True)
        return result

class LoadFunction(LambdaClosure):
    """A built-in function for loading data files from disk.

    This gets inserted into the inputs environment under the name 'load'.
    """
    def __init__(self, basePath):
        """Initialise an instance of the load() built-in.

        :param basePath: path with respect to which to resolve relative data file paths.
        """
        self.basePath = basePath

    def __str__(self):
        """Return a string representation of this function."""
        return "load()"

    def Compile(self, env, actualParameters):
        raise NotImplementedError

    def Evaluate(self, env, actualParameters):
        """Evaluate a load() function call.
        
        :param env: the environment within which to evaluate this call
        :param actualParameters: the values of the parameters to the call; should be a single
            string value containing the path of the file to load
        :returns: an Array containing the file's data, if successful
        """
        if len(actualParameters) != 1:
            raise ProtocolError("A load() call takes a single parameter, not %d." % len(actualParameters))
        if not isinstance(actualParameters[0], String):
            raise ProtocolError("A load() call takes a string parameter with the file path to load.")
        import os
        file_path = os.path.join(self.basePath, actualParameters[0].value)
        from ..utility.test_support import Load2d
        return Load2d(file_path)
