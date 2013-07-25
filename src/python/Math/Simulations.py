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
import Environment as Env
import scipy.integrate
import Values as V
import numpy as np
from Modifiers import AbstractModifier

class AbstractSimulation(object):
    """Base class for simulations in the protocol language."""
    def __init__(self, prefix=None):
        self.prefix = prefix
        self.ranges = [self.range_]
        self.model = None
#         if prefix:
        self.results = Env.Environment()
#         else:
#             self.results = None
        self.env = Env.Environment()
        self.env.DefineName(self.range_.name, self.range_)
    
    def Initialise(self):
        for r in self.ranges:
            r.Initialise(self.env)
        
    def InternalRun(self):
        raise NotImplementedError 
    
    def LoopBodyStartHook(self):
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.START_ONLY and self.range_.count == 1:
                modifier.Apply(self)
            elif modifier.when == AbstractModifier.EACH_LOOP:
                modifier.Apply(self)
    
    def LoopEndHook(self):
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.END_ONLY:
                modifier.Apply(self)
    #loopbodyend happens at the end of the loop but still inside the loop. before next iteration
    # loopbodyend checks if its a while loop and iterates over output names in results and defines
    #the same names in the fake environment but defines the names as a view from the beginning
    #of each array to the current count of the while loop for the 0th dimension and pretty much
    #everything for all other dimensions
    #self.viewEnv is added as delegatee to self.env (with simulations prefix) and only created
    #if you're in a while loop
    
    def SetModel(self, model):
        self.model = model
        model_env = model.GetEnvironmentMap()
        for prefix in model_env.keys():
            self.env.SetDelegateeEnv(model_env[prefix], prefix)
            self.results.SetDelegateeEnv(model_env[prefix], prefix)
        # add model's environment(s) as prefixed delegatees to self.env and self.results
        
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
        for t in self.range_:
            self.LoopBodyStartHook()
            if self.range_.count == 1:
                self.model.SetInitialTime(t)
                self.AddIterationOutputs(self.model.GetOutputs())
            else:
                self.model.Simulate(t)
                self.AddIterationOutputs(self.model.GetOutputs())
        self.LoopBodyEndHook()
    
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
    
    def ZeroInitialiseResults(self):
        pass
    
    def InternalRun(self):
        for t in self.range_:
            self.LoopBodyStartHook()
            self.nestedSim.Run()
        self.LoopBodyEndHook()
        
    def SetModel(self, model):
        self.model = model
        self.nestedSim.SetModel(model)
    
    
    