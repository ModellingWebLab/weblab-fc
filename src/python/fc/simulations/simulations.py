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

import itertools
import operator
import sys
import numpy as np

from itertools import izip
from operator import attrgetter

from . import ranges as R
from .model import NestedProtocol
from .modifiers import AbstractModifier
from ..language import values as V
from ..utility import environment as Env
from ..utility import locatable


class AbstractSimulation(locatable.Locatable):
    """Base class for simulations in the protocol language."""
    def __init__(self, prefix=None):
        super(AbstractSimulation, self).__init__()
        self.prefix = prefix
        self.ranges = [self.range_]
        self.model = None
        self.results = Env.Environment()
        self.resultsList = [] # An ordered view on the unwrapped versions of simulation results
        self.env = Env.Environment()
        self.viewEnv = None
        self.indentLevel = 0
        
        try:
            line_profile.add_function(self.AddIterationOutputs)
            line_profile.add_function(self.LoopBodyStartHook)
        except NameError:
            pass

    def __getstate__(self):
        # Must remove Model class and regenerate during unpickling
        # (Pickling errors from nested class structure of ModelWrapperEnvironment)
        
        # Undo Simulation.SetModel
        if self.model is not None:
            modelenv = self.model.GetEnvironmentMap()
            for prefix in modelenv:
                if isinstance(self,Nested):
                    self.nestedSim.env.ClearDelegateeEnv(prefix)
                self.results.ClearDelegateeEnv(prefix)
                self.env.ClearDelegateeEnv(prefix)

        odict = self.__dict__.copy()
        odict['model'] = None
        return odict

    def __setstate__(self,dict):
        self.__dict__.update(dict)

    def Initialise(self, initialiseRange=True):
        if initialiseRange:
            self.range_.Initialise(self.env)
        if self.viewEnv is None and isinstance(self.range_, R.While) and self.prefix:
            # NB: We can't do this in the constructor as self.prefix may not be set until later
            self.viewEnv = Env.Environment(allowOverwrite=True)
            self.env.SetDelegateeEnv(self.viewEnv, self.prefix)

    def Clear(self):
        self.env.Clear()
        self.results.Clear()
        self.resultsList[:] = []
        if self.viewEnv:
            self.viewEnv.Clear()
    
    def SetIndentLevel(self, indentLevel):
        """Set the level of indentation to use for progress output."""
        self.indentLevel = indentLevel
        if self.model:
            self.model.SetIndentLevel(indentLevel)

    def LogProgress(self, *args):
        """Print a progress line showing how far through the simulation we are.
        
        Arguments are converted to strings and space separated, as for the print builtin.
        """
        print '  ' * self.indentLevel + ' '.join(map(str, args))
        sys.stdout.flush()

    def InternalRun(self, verbose=True):
        raise NotImplementedError

    def SetOutputFolder(self, folder):
        self.outputFolder = folder
        if self.trace:
            self.model.SetOutputFolder(folder)

    def LoopBodyStartHook(self):
        if isinstance(self.range_, R.While) and self.range_.count > 0 and self.resultsList and self.range_.GetNumberOfOutputPoints() > self.resultsList[0].shape[0]:
            for name in self.results:
                result = self.results.LookUp(name).array
                shape = list(result.shape)
                shape[0] = self.range_.GetNumberOfOutputPoints()
                result.resize(tuple(shape), refcheck=False)
                # TODO: Check if the next line is needed?
                self.viewEnv.OverwriteDefinition(name, V.Array(result[0:1+self.range_.count]))
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.START_ONLY and self.range_.count == 0:
                modifier.Apply(self)
            elif modifier.when == AbstractModifier.EACH_LOOP:
                modifier.Apply(self)

    def LoopEndHook(self):
        if isinstance(self.range_, R.While):
            for name in self.results:
                result = self.results.LookUp(name)
                result.array = result.array[0:self.range_.GetNumberOfOutputPoints()] #resize function doesn't work with references
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.END_ONLY:
                modifier.Apply(self)

    def LoopBodyEndHook(self):
        if self.viewEnv is not None:
            for name in self.results:
                if name not in self.viewEnv:
                    self.viewEnv.DefineName(name, V.Array(self.results.LookUp(name).array[0:1+self.range_.count]))
                else:
                    self.viewEnv.OverwriteDefinition(name, V.Array(self.results.LookUp(name).array[0:1+self.range_.count]))

    def SetModel(self, model):
        if isinstance(self.model, NestedProtocol):
            self.model.proto.SetModel(model)
        else:
            self.model = model
        self.model.SetIndentLevel(self.indentLevel)
        model_env = model.GetEnvironmentMap()
        model.simEnv = self.env # TODO: this breaks if a model is used in multiple simulations!  Only needed for NestedProtocol?
        for prefix, env in model_env.iteritems():
            self.env.SetDelegateeEnv(env, prefix)
            self.results.SetDelegateeEnv(env, prefix)

    def Run(self, verbose=True):
        self.InternalRun(verbose)
        return self.results

    def AddIterationOutputs(self, outputsList):
        """Copy model outputs from one simulation step into the overall output arrays for the (possibly nested) simulation."""
        self_results, results_list = self.results, self.resultsList
        if self_results is not None:
            if isinstance(outputsList, tuple):
                # Some simulation outputs were missing
                outputsList, missing_outputs = outputsList
            else:
                missing_outputs = []
            if not self_results:
                # First iteration - create empty output arrays of the correct shape
                range_dims = tuple(r.GetNumberOfOutputPoints() for r in self.ranges)
                for name, output in itertools.izip(self.model.outputNames, outputsList):
                    result = V.Array(np.empty(range_dims + output.shape))
                    self_results.DefineName(name, result)
                    results_list.append(result.unwrapped)
            elif missing_outputs:
                for i, name in missing_outputs:
                    del results_list[i]
                    self_results.allowOverwrite = True
                    self_results.Remove(name)
                    self_results.allowOverwrite = False
        if results_list:
            # Note that the tuple conversion in the next line is very quick
            range_indices = tuple(map(attrgetter('count'), self.ranges))  # tuple(r.count for r in self.ranges)
            for output, result in izip(outputsList, results_list):
                result[range_indices] = output


class Timecourse(AbstractSimulation):
    """Simulate a simple loop over time."""
    def __init__(self, range_, modifiers=[]):
        self.range_ = range_
        super(Timecourse, self).__init__()
        self.ranges = [self.range_]
        self.modifiers = modifiers

        try:
            line_profile.add_function(self.InternalRun)
        except NameError:
            pass

    def InternalRun(self, verbose=True):
        r = self.range_
        m = self.model
        start_hook, end_hook = self.LoopBodyStartHook, self.LoopBodyEndHook
        add_outputs, get_outputs = self.AddIterationOutputs, m.GetOutputs
        set_time, simulate = m.SetFreeVariable, m.Simulate
        for t in r:
            if r.count == 0:
                # Record initial conditions
                start_hook()
                set_time(t)
            else:
                # Loop through remaining time points.
                # Note that the start_hook is called *after* simulate in order to match the C++ implementation:
                # in effect it is the hook for the *next* iteration of the loop.
                simulate(t)
                start_hook()
            add_outputs(get_outputs())
            end_hook()
        self.LoopEndHook()


class OneStep(AbstractSimulation):
    """Simulate one logical execution of a model."""
    
    class NullRange(R.AbstractRange):
        pass
    
    def __init__(self, step, modifiers=[]):
        self.step = step
        self.modifiers = modifiers
        self.range_ = self.NullRange('_')
        super(OneStep, self).__init__()
        self.ranges = []

    def InternalRun(self, verbose=True):
        self.LoopBodyStartHook()
        self.model.Simulate(self.step)
        self.AddIterationOutputs(self.model.GetOutputs())
        self.LoopEndHook()


class Nested(AbstractSimulation):
    """The main nested loop simulation construct."""
    def __init__(self, nestedSim, range_, modifiers=[]):
        self.range_ = range_
        super(Nested, self).__init__()
        self.nestedSim = nestedSim
        self.modifiers = modifiers
        self.ranges = self.nestedSim.ranges
        self.ranges.insert(0, self.range_)
        self.results = self.nestedSim.results
        self.resultsList = self.nestedSim.resultsList
        nestedSim.env.SetDelegateeEnv(self.env)

    def Initialise(self):
        self.range_.Initialise(self.env)
        self.nestedSim.Initialise()
        if self.trace:
            self.nestedSim.trace = True
        super(Nested, self).Initialise(initialiseRange=False)

    def Clear(self):
        self.nestedSim.Clear()
        super(Nested, self).Clear()
    
    def SetIndentLevel(self, indentLevel):
        super(Nested, self).SetIndentLevel(indentLevel)
        self.nestedSim.SetIndentLevel(indentLevel + 1)

    def InternalRun(self, verbose=True):
        for t in self.range_:
            if verbose:
                self.LogProgress('nested simulation', self.range_.name, 'step', self.range_.GetCurrentOutputNumber(), '(value', self.range_.GetCurrentOutputPoint(), ')')
            self.LoopBodyStartHook()
            if self.outputFolder:
                self.nestedSim.SetOutputFolder(self.outputFolder.CreateSubfolder('run_%d' %self.range_.GetCurrentOutputNumber()))
            self.nestedSim.Run()
            self.LoopBodyEndHook()
        self.LoopEndHook()

    def SetModel(self, model):
        super(Nested, self).SetModel(model)
        self.nestedSim.SetModel(model)
