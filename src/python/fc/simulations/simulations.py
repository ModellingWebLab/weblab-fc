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
        
        try:
            line_profile.add_function(self.AddIterationOutputs)
            line_profile.add_function(self.LoopBodyStartHook)
        except NameError:
            pass

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

    def InternalRun(self):
        raise NotImplementedError

    def SetOutputFolder(self, folder):
        self.outputFolder = folder
        if self.trace:
            self.model.SetOutputFolder(folder)

    def LoopBodyStartHook(self):
        if isinstance(self.range_, R.While) and self.range_.count > 0 and self.range_.GetNumberOfOutputPoints() > self.results.LookUp(self.results.DefinedNames()[0]).array.shape[0]:
            for name in self.results:
                self.results.LookUp(name).array.resize(self.range_.GetNumberOfOutputPoints(), refcheck=False)
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.START_ONLY and self.range_.count == 0:
                modifier.Apply(self)
            elif modifier.when == AbstractModifier.EACH_LOOP:
                modifier.Apply(self) ## ~96% of time

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
            for result in self.results:
                if result not in self.viewEnv:
                    self.viewEnv.DefineName(result, V.Array(self.results.LookUp(result).array[0:1+self.range_.count]))
                else:
                    self.viewEnv.OverwriteDefinition(result, V.Array(self.results.LookUp(result).array[0:1+self.range_.count]))

    def SetModel(self, model):
        if isinstance(self.model, NestedProtocol):
            self.model.proto.SetModel(model)
        else:
            self.model = model
        model_env = model.GetEnvironmentMap()
        model.simEnv = self.env # TODO: this breaks if a model is used in multiple simulations!  Only needed for NestedProtocol?
        for prefix in model_env.keys():
            self.env.SetDelegateeEnv(model_env[prefix], prefix)
            self.results.SetDelegateeEnv(model_env[prefix], prefix)

    def Run(self):
        self.InternalRun()
        return self.results

    def AddIterationOutputs(self, outputsList):
        """Copy model outputs from one simulation step into the overall output arrays for the (possibly nested) simulation."""
        self_results, results_list = self.results, self.resultsList
        if self_results is not None and not self_results:
            # First iteration - create empty output arrays of the correct shape
            range_dims = tuple(r.GetNumberOfOutputPoints() for r in self.ranges)
            for name, output in itertools.izip(self.model.outputNames, outputsList):
                result = V.Array(np.empty(range_dims + output.shape))
                self_results.DefineName(name, result)
                results_list.append(result.unwrapped)
        if results_list:
            # Note that the tuple conversion in the next line is very quick
            range_indices = tuple(map(attrgetter('count'), self.ranges))  # tuple(r.count for r in self.ranges)
            for output, result in izip(outputsList, results_list):
                result[range_indices] = output


class Timecourse(AbstractSimulation):
    def __init__(self, range_, modifiers=[]):
        self.range_ = range_
        super(Timecourse, self).__init__()
        self.ranges = [self.range_]
        self.modifiers = modifiers

        try:
            line_profile.add_function(self.InternalRun)
        except NameError:
            pass

    def InternalRun(self):
        r = self.range_
        m = self.model
        start_hook, end_hook = self.LoopBodyStartHook, self.LoopBodyEndHook
        add_outputs, get_outputs = self.AddIterationOutputs, m.GetOutputs
        set_time, simulate = m.SetFreeVariable, m.Simulate
        for i, t in enumerate(r):
            start_hook() ## ~ 50% of time
            if r.count == 0:
                # Record initial conditions
                set_time(t)
                add_outputs(get_outputs())
            else:
                simulate(t)
                add_outputs(get_outputs())  ## ~45% of time
            end_hook()
        self.LoopEndHook()


class OneStep(AbstractSimulation):
    
    class NullRange(R.AbstractRange):
        pass
    
    def __init__(self, step, modifiers=[]):
        self.step = step
        self.modifiers = modifiers
        self.range_ = self.NullRange('_')
        super(OneStep, self).__init__()
        self.ranges = []

    def InternalRun(self):
        self.LoopBodyStartHook()
        self.model.Simulate(self.step)
        self.AddIterationOutputs(self.model.GetOutputs())
        self.LoopEndHook()


class Nested(AbstractSimulation):
    def __init__(self, nestedSim, range_, modifiers=[]):
        self.range_ = range_
        super(Nested, self).__init__()
        self.nestedSim = nestedSim
        self.modifiers = modifiers
        self.ranges = self.nestedSim.ranges
        self.ranges.insert(0, self.range_)
        self.results = self.nestedSim.results
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

    def InternalRun(self):
        for t in self.range_:
            print 'nested simulation', self.range_.name, 'step', self.range_.GetCurrentOutputNumber(), '(value', self.range_.GetCurrentOutputPoint(), ')' 
            self.LoopBodyStartHook()
            if self.outputFolder:
                self.nestedSim.SetOutputFolder(self.outputFolder.CreateSubfolder('run_%d' %self.range_.GetCurrentOutputNumber()))
            self.nestedSim.Run()
            self.LoopBodyEndHook()
        self.LoopEndHook()

    def SetModel(self, model):
        super(Nested, self).SetModel(model)
        self.nestedSim.SetModel(model)
