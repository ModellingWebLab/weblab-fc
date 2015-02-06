"""Copyright (c) 2005-2015, University of Oxford.
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

from ..language import values as V

class AbstractRange(V.Simple):
    """Base class for ranges in the protocol language.

    Handles enforcing the update of the range's value in the simulation environment's Python bindings dictionary.
    """
    def __init__(self, name):
        """Initialise the common range properties."""
        self.name = name
        self.count = -1
        self._value = float('nan')
        self.numberOfOutputs = 0
        # Set an initial empty environment so calls to set our value in constructors don't fail
        # (since Initialise hasn't been called by our simulation yet)
        from ..utility.environment import Environment
        AbstractRange.Initialise(self, Environment())

    def Initialise(self, env):
        """Called by the associated simulation when its environment is initialised.

        Here we define the range variable within the environment.
        Subclasses should also evaluate any expressions used to define the range.
        """
        self.env = env
        env.DefineName(self.name, self)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.env.unwrappedBindings[self.name] = value

    @property
    def unwrapped(self):
        return self._value

    @property
    def current(self):
        return self._value

    def GetCurrentOutputPoint(self):
        return self._value

    def GetCurrentOutputNumber(self):
        return self.count

    def GetNumberOfOutputPoints(self):
        return self.numberOfOutputs


class UniformRange(AbstractRange):
    def __init__(self, name, startExpr, endExpr, stepExpr):
        super(UniformRange, self).__init__(name)
        self.startExpr = startExpr
        self.endExpr = endExpr
        self.stepExpr = stepExpr

        try:
            line_profile.add_function(self.next)
        except NameError:
            pass

    def __iter__(self):
        self.count = -1
        self.value = self.start
        return self

    def next(self):
        self.count += 1
        if self.count >= self.numberOfOutputs:
            self.count = -1
            raise StopIteration
        else:
            self.value = self.start + self.step * self.count
            return self.value

    def Initialise(self, env):
        super(UniformRange, self).Initialise(env)
        self.start = self.startExpr.Evaluate(self.env).value
        self.step = self.stepExpr.Evaluate(self.env).value
        self.end = self.endExpr.Evaluate(self.env).value
        self.value = self.start
        self.numberOfOutputs = int((round(self.end-self.start)/self.step)) + 1


class VectorRange(AbstractRange):
    def __init__(self, name, arrOrExpr):
        super(VectorRange, self).__init__(name)
        if isinstance(arrOrExpr, V.Array):
            self.expr = None
            self.arrRange = arrOrExpr.array
            self.value = self.arrRange[0]
            self.numberOfOutputs = len(self.arrRange)
        else:
            self.expr = arrOrExpr
            self.value = float('nan')
        self.count = -1

    def Initialise(self, env):
        super(VectorRange, self).Initialise(env)
        if self.expr:
            self.arrRange = self.expr.Evaluate(env).array
            self.value = self.arrRange[0]
            self.numberOfOutputs = len(self.arrRange)

    def __iter__(self):
        self.count = -1
        self.value = 0
        return self

    def next(self):
        self.count += 1
        if self.count >= self.numberOfOutputs:
            self.count = -1
            raise StopIteration
        else:
            self.value = self.arrRange[self.count]
            return self.current


class While(AbstractRange):
    def __init__(self, name, condition):
        super(While, self).__init__(name)
        self.condition = condition
        self._Init()
    
    def _Init(self):
        """(Re-)Initialise the range loop."""
        self.count = -1
        self.value = -1
        self.numberOfOutputs = 1000
    
    def Initialise(self, env):
        super(While, self).Initialise(env)
        self._Init()

    def __iter__(self):
        self.count = -1
        self.value = -1
        return self

    def next(self):
        self.count += 1
        self.value += 1
        if self.count >= self.numberOfOutputs:
            self.numberOfOutputs += 1000
        if self.count > 0 and not self.condition.Evaluate(self.env).value:
            self.numberOfOutputs = self.count
            raise StopIteration
        else:
            return self.value
