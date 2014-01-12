"""Copyright (c) 2005-2014, University of Oxford.
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
        self.env = Env.Environment()
        
        try:
            line_profile.add_function(self.AddIterationOutputs)
            line_profile.add_function(self.LoopBodyStartHook)
        except NameError:
            pass

    def Initialise(self, initialiseRange=True):
        if initialiseRange:
            self.range_.Initialise(self.env)
        if isinstance(self.range_, R.While) and self.prefix:
            self.viewEnv = Env.Environment(allowOverwrite=True)
            self.env.SetDelegateeEnv(self.viewEnv, self.prefix)

    def InternalRun(self):
        raise NotImplementedError

    def SetOutputFolder(self, folder):
        self.outputFolder = folder
        if self.trace:
            self.model.SetOutputFolder(folder)

    def LoopBodyStartHook(self):
        if isinstance(self.range_, R.While) and self.range_.count > 1 and self.range_.GetNumberOfOutputPoints() > self.results.LookUp(self.results.DefinedNames()[0]).array.shape[0]:
            for name in self.results:
                self.results.LookUp(name).array.resize(self.range_.GetNumberOfOutputPoints(), refcheck=False)
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.START_ONLY and self.range_.count == 1:
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
        if isinstance(self.range_, R.While) and self.prefix:
            for result in self.results:
                if result not in self.viewEnv:
                    self.viewEnv.DefineName(result, V.Array(self.results.LookUp(result).array[0:self.range_.count]))
                else:
                    self.viewEnv.OverwriteDefinition(result, V.Array(self.results.LookUp(result).array[0:self.range_.count]))

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

    def AddIterationOutputs(self, env):
        self_results = self.results
        if self_results is not None and not self_results:
            # First iteration - create empty output arrays of the correct shape
            range_dims = tuple(r.GetNumberOfOutputPoints() for r in self.ranges)
            for name in env:
                output = env[name]
                results = np.empty(range_dims + output.shape)
                self_results.DefineName(name, V.Array(results))
        if self_results:
            unwrapped_results = self_results.unwrappedBindings
            range_indices = tuple(r.GetCurrentOutputNumber() for r in self.ranges) ## ~30% of time; tuple conversion is minimal
            for name in env:
                result = unwrapped_results[name]
                result[range_indices] = env[name]


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
            if r.count == 1:
                # Record initial conditions
                set_time(t)
                add_outputs(get_outputs())
            else:
                simulate(t)
                add_outputs(get_outputs())  ## ~45% of time
            end_hook()
        self.LoopEndHook()


class OneStep(AbstractSimulation):
    def __init__(self, step, modifiers=[]):
        self.step = step
        self.modifiers = modifiers
        self.range_ = R.VectorRange('count', V.Array(np.array([1])))
        self.ranges = [self.range_]
        super(OneStep, self).__init__()

    def InternalRun(self):
        self.LoopBodyStartHook()
        for t in self.range_:
            self.model.Simulate(t + self.step)
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
