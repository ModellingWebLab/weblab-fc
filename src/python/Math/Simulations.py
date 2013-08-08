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
from Modifiers import AbstractModifier
import ArrayExpressions as A
import Environment as Env
import Expressions as E
import scipy.integrate
from Locatable import Locatable
import MathExpressions as M
import Model
import numpy as np
import os
import Ranges
import Values as V

def N(number):
    return M.Const(V.Simple(number))

class AbstractSimulation(Locatable):
    """Base class for simulations in the protocol language."""
    def __init__(self, prefix=None):
        super(AbstractSimulation, self).__init__()
        self.prefix = prefix
        self.ranges = [self.range_]
        self.model = None
        self.results = Env.Environment()
        self.env = Env.Environment()
    
    def Initialise(self):
        self.env.DefineName(self.range_.name, self.range_)
        self.range_.Initialise(self.env)
        if isinstance(self.range_, Ranges.While) and self.prefix:
            self.viewEnv = Env.Environment(allowOverwrite=True)
            self.env.SetDelegateeEnv(self.viewEnv, self.prefix)
        
    def InternalRun(self):
        raise NotImplementedError
    
    def SetOutputFolder(self, folder):
        self.outputFolder = folder
        if self.trace:
            self.model.SetOutputFolder(folder)
        # called by protocol with subfolder of its output
        # store and set on model
    
    def LoopBodyStartHook(self):
        # if the simulations being traced and protocol has an output folder, 
        #then set the model's output folder 
        #to be a subfolder of this with a name unique to the particular iteration around the loop
        # if any of the runs don't have files in them after the simulation,delete the empty folders
        # if tracing, call setoutputfolder on the model
        # tmp folder that has model.py and proto.xml (created in protocol.set model) 
        #can also be in the same directory
        #you can still use the mdtemp and a keyword to save this file to the right place
        #within the test output folder for each test you have a folder called simulation_<prefix>
        #and in that you have one for each run of the outer loop and in those one for each run of
        #the inner loop
                
        if isinstance(self.range_, Ranges.While) and self.range_.count > 1 and self.range_.GetNumberOfOutputPoints() > self.results.LookUp(self.results.DefinedNames()[0]).array.shape[0]:
            for name in self.results.DefinedNames():
                self.results.LookUp(name).array.resize(self.range_.GetNumberOfOutputPoints(), refcheck=False)
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.START_ONLY and self.range_.count == 1:
                modifier.Apply(self)
            elif modifier.when == AbstractModifier.EACH_LOOP:
                modifier.Apply(self)
    
    def LoopEndHook(self):
        if isinstance(self.range_, Ranges.While):
            for name in self.results.DefinedNames():
                result = self.results.LookUp(name)
                result.array = result.array[0:self.range_.GetNumberOfOutputPoints()] #resize function doesn't work with references
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.END_ONLY:
                modifier.Apply(self)
                
    def LoopBodyEndHook(self):
        if isinstance(self.range_, Ranges.While) and self.prefix:
            for result in self.results.DefinedNames():
                if result not in self.viewEnv.DefinedNames():
                    self.viewEnv.DefineName(result, V.Array(self.results.LookUp(result).array[0:self.range_.count]))
                else:
                    self.viewEnv.OverwriteDefinition(result, V.Array(self.results.LookUp(result).array[0:self.range_.count]))
    
    def SetModel(self, model):
        if isinstance(self.model, Model.NestedProtocol):
            self.model.proto.SetModel(model)
        else:
            self.model = model
        model_env = model.GetEnvironmentMap()
        model.simEnv = self.env
        for prefix in model_env.keys():
            self.env.SetDelegateeEnv(model_env[prefix], prefix)
            self.results.SetDelegateeEnv(model_env[prefix], prefix)
        
    def Run(self):
        self.InternalRun()
        return self.results
    
    def AddIterationOutputs(self, env):
        if self.results is not None and not self.results:  
            range_dims = tuple([r.GetNumberOfOutputPoints() for r in self.ranges])         
            for name in env.DefinedNames():
                output = env.LookUp(name)
                results = np.empty(range_dims + output.array.shape)
                self.results.DefineName(name, V.Array(results))
        if self.results:
            range_indices = tuple([r.GetCurrentOutputNumber() for r in self.ranges])
            for name in env.DefinedNames():
                result = self.results.LookUp(name).array
                result[range_indices] = env.LookUp(name).array
        
class Timecourse(AbstractSimulation):   
    def __init__(self, range_, modifiers=[]):
        self.range_ = range_
        super(Timecourse, self).__init__()
        self.ranges = [self.range_]     
        self.modifiers = modifiers
        
    def InternalRun(self):
        for i,t in enumerate(self.range_):
            self.LoopBodyStartHook()
            if self.range_.count == 1:
                self.model.SetFreeVariable(t)
                self.AddIterationOutputs(self.model.GetOutputs())
            else:
                self.model.Simulate(t)
                self.AddIterationOutputs(self.model.GetOutputs())
            self.LoopBodyEndHook()
        self.LoopEndHook()
        
class OneStep(AbstractSimulation):
    def __init__(self, step, modifiers=[]):
        self.step = step
        self.modifiers = modifiers
        self.range_ = Ranges.VectorRange('count', V.Array(np.array([1])))
        self.ranges = [self.range_]
        super(OneStep, self).__init__()
        
    def InternalRun(self):
        self.LoopBodyStartHook()
        for t in self.range_:
            self.model.Simulate(t + self.step)
            self.AddIterationOutputs(self.model.GetOutputs())
        self.LoopEndHook()

# like timecourse but takes one step instead of a few steps, simulates from current 
#time up to time + step
# self.model.Simulate(t + step)
# self.AddIterationOutputs
# startloopbodyhoook
# range is vector range with zero in it
# ends with loopendhook
    
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
        super(Nested, self).Initialise()
        
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