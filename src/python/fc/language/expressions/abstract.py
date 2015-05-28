
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

import os

from ...utility import locatable
from .. import values as V

class AbstractExpression(locatable.Locatable):
    """Base class for expressions in the protocol language."""   
    def __init__(self, *children):
        """Create a new expression node, with a list of child expressions, possibly empty."""
        super(AbstractExpression, self).__init__()
        self.children = children
        
        try:
            line_profile.add_function(self.RealEvaluateCompiled)
        except NameError:
            pass

    # Override Object serialization methods to allow pickling with the dill module
    def __getstate__(self):
        odict = self.__dict__.copy()
        # These properties cause namespace errors during pickling, and will be 
        # automatically regenerated on first reference after unpickling.
        if '_compiledFunction' in odict:
            del odict['_compiledFunction']
        if '_evalGlobals' in odict:
            del odict['_evalGlobals']
        return odict

    def GetUsedVariables(self):
        """Get the set of (non-local) identifiers referenced within this expression."""
        result = set()
        for child in self.children:
            result |= child.GetUsedVariables()
        return result

    def EvaluateChildren(self, env):
        """Evaluate our child expressions and return a list of their values."""
        return [child.Evaluate(env) for child in self.children]

    def Interpret(self, env):
        """Evaluate this expression by interpreting the expression tree.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def Compile(self, arrayContext=True):
        """Create a string representation of this expression for evaluation by numexpr or numpy.

        By default, evaluation of the generated string is only guaranteed to give correct results (as defined
        by the Interpret method) in certain array contexts, for instance maps and array comprehensions.
        If arrayContext is set to False, the generated string will instead only evaluate correctly when
        considered as a standalone expression operating on simple values (useful in the SetVariable modifier,
        for instance).

        TODO: Always stop compilation succeeding if an expression or subexpression is traced.
        (Make trace on the base class a property, which when set replaces the Compile method with one
        that always fails.)
        """
        raise NotImplementedError

    def Evaluate(self, env):
        """We always default to using Interpret for evaluating standalone expressions."""
        result = self.Interpret(env)
        if self.trace:
            self.Trace(result)
        return result

    def EvaluateCompiled(self, env):
        """Try to evaluate the compiled version of this expression, returning an unwrapped Python value.

        This method only gets used on the first call.  If compilation fails, this method's name will be rebound
        to the normal Evaluate method (or rather, a lambda which unwraps the result).  If compilation succeeds,
        the RealEvaluateCompiled method will be used.
        """
        try:
            value = self.RealEvaluateCompiled(env)
        except NotImplementedError:
            self.EvaluateCompiled = lambda env: self.Evaluate(env).value
            value = self.EvaluateCompiled(env)
        else:
            self.EvaluateCompiled = self.RealEvaluateCompiled
        return value

    @property
    def compiled(self):
        """Compile this expression and cache the result."""
        try:
            return self._compiled
        except AttributeError:
            # We haven't called Compile yet; cache the result
            c = self._compiled = self.Compile(arrayContext=False)
            return c

    @property
    def evalGlobals(self):
        try:
            return self._evalGlobals
        except AttributeError:
            d = self._evalGlobals = {}
            d['abs'] = abs
            import math
            for name in ['log', 'log10', 'exp']:
                d[name] = getattr(math, name)
            import numpy
            d['___np'] = numpy
            return d

    def RealEvaluateCompiled(self, env):
        """Evaluate the compiled form of this expression using eval().

        This is suitable for use only in non-array contexts, such as by the SetVariable modifier.
        """
        func = self.compiledFunction
        arg_envs = self.GetDefiningEnvironments(env)
        assert env is self._rootDefiningEnv, "Internal implementation assumption violated"
        args = [arg_envs[name].unwrappedBindings[name] for name in self._usedVarLocalNames]
        return func(*args)

    def GetDefiningEnvironments(self, env):
        """Cache which environment each variable used in this expression is actually defined in.

        Stores each environment indexed by the local name of the variable within that environment.
        
        TODO: Handle local name conflicts!
        """
        try:
            return self._definingEnvs
        except AttributeError:
            self._rootDefiningEnv = env  # For paranoia checking that the cache is valid
            d = self._definingEnvs = {}
            l = self._usedVarLocalNames = []
            for name in self.usedVariableList:
                local_name = name[name.rfind(':')+1:]
                l.append(local_name)
                d[local_name] = env.FindDefiningEnvironment(name)
            return d

    @property
    def compiledFunction(self):
        """A version of self.compiled that has been converted to a Python function by eval()."""
        try:
            return self._compiledFunction
        except AttributeError:
            from .general import NameLookUp
            arg_defs = ', '.join(NameLookUp.PythonizeName(name) for name in self.usedVariableList)
            f = self._compiledFunction = eval('lambda ' + arg_defs + ': ' + self.compiled, self.evalGlobals)
            return f

    @property
    def usedVariableList(self):
        """Cached property version of self.usedVariables that's in a predictable order."""
        try:
            return self._usedVarList
        except AttributeError:
            l = self._usedVarList = list(self.usedVariables)
            l.sort()
            return l

    @property
    def usedVariables(self):
        """Cached property version of self.GetUsedVariables()."""
        try:
            return self._usedVars
        except AttributeError:
            # Create the cache
            u = self._usedVars = self.GetUsedVariables()
            return u
