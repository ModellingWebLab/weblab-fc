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
from ErrorHandling import ProtocolError
from AbstractValue import AbstractValue

class Environment(object):
    
    def __init__(self, allowOverwrite=False):
        self.allowOverwrite = allowOverwrite
        self.bindings = {}
        
    def DefineName(self, name, value):
        if not isinstance(value, AbstractValue):
            raise ProtocolError(value, "is not a value type")
        if name in self.bindings:
            raise ProtocolError(name, "is already defined and may not be re-bound")
        else:
            self.bindings[name] = value
    
    def DefineNames(self, names, values):
        for i, name in enumerate(names):
            self.DefineName(name, values[i])
                
    def LookUp(self, name):
        try:
            result = self.bindings[name]
        except KeyError:
            raise ProtocolError("The name", name, "does not exist in the environment")
        return result
    
    def Merge(self, env):
        self.DefineNames(env.bindings.keys(), env.bindings.values())
            
    def Remove(self, name):
        if not self.allowOverwrite:
            raise ProtocolError("This environment does not support overwriting mappings")
        if name not in self.bindings:
            raise ProtocolError(name, "is not defined in this environment and thus cannot be removed")
        del (self.bindings[name])
        
    def OverwriteDefinition(self, name, value):
        if not self.allowOverwrite:
            raise ProtocolError("This environment does not support overwriting mappings")
        if name not in self.bindings:
            raise ProtocolError(name, "is not defined in this environment and thus cannot be overwritten")
        self.bindings[name]= value #can't use DefineName because error would be thrown for name already being in environment
            
    def Clear(self):
        self.bindings.clear()
            
    def __len__(self):
        return len(self.bindings)
    
    def DefinedNames(self):
        return self.bindings.keys()
    
            
        