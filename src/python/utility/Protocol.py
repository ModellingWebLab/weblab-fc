
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
import CompactSyntaxParser as CSP
import Environment as Env
from Locatable import Locatable
import os
from ErrorHandling import ProtocolError
import sys

csp = CSP.CompactSyntaxParser

class Protocol(object):
    """Base class for protocols in the protocol language."""
    def __init__(self, protoFile):
        self.protoFile = protoFile
        self.env = Env.Environment()
        self.inputEnv = Env.Environment(allowOverwrite=True)
        self.libraryEnv = Env.Environment()
        self.postProcessingEnv = Env.Environment(delegatee=self.libraryEnv)
        self.library = []
        self.simulations = []
        self.postProcessing = []
        parser = csp()
        CSP.Actions.source_file = protoFile
        generator = parser._Try(csp.protocol.parseFile, protoFile, parseAll=True)[0]
        assert isinstance(generator, CSP.Actions.Protocol)
        details = generator.expr()
        assert isinstance(details, dict)
        for prefix, path in details.get('imports', []):
            imported_proto = Protocol(self.GetPath(protoFile, path))
            if prefix == "":    
                self.library.extend(imported_proto.library)
                self.simulations.extend(imported_proto.simulations)
                self.postProcessing.extend(imported_proto.postProcessing)
            else:
                self.libraryEnv.SetDelegateeEnv(imported_proto.libraryEnv, prefix)
                imported_proto.libraryEnv.ExecuteStatements(imported_proto.library)
        self.inputEnv.ExecuteStatements(details.get('inputs', []))
        self.libraryEnv.SetDelegateeEnv(self.inputEnv)
        self.library.extend(details.get('library', []))
        self.simulations.extend(details.get('simulations', []))
        self.postProcessing.extend(details.get('postprocessing', []))      

    def Run(self):
        try:
            self.libraryEnv.ExecuteStatements(self.library)
            for sim in self.simulations:
                sim.env.SetDelegateeEnv(self.libraryEnv)
                sim.Initialise()
                if sim.prefix:
                    self.libraryEnv.SetDelegateeEnv(sim.results, sim.prefix)
                results = sim.Run()
            self.postProcessingEnv.ExecuteStatements(self.postProcessing)
        except ProtocolError:
            locations = []
            current_trace = sys.exc_info()[2]               
            while current_trace is not None:
                local_vars = current_trace.tb_frame.f_locals 
                if 'self' in local_vars:
                    if isinstance(local_vars['self'], Locatable) and (not locations or local_vars['self'].location != locations[-1]):
                        locations.append(local_vars['self'].location)
                current_trace = current_trace.tb_next
            for location in locations:
                print location
            raise
        
    def SetInput(self, name, valueExpr):
        value = valueExpr.Evaluate(self.inputEnv)
        self.inputEnv.OverwriteDefinition(name, value)
        
    def SetModel(self, model):
        if isinstance(model, str):
            import tempfile, subprocess, sys, imp
            import subprocess
            dir = tempfile.mkdtemp()
            xml_file = subprocess.check_output(['python', 'projects/FunctionalCuration/src/proto/parsing/CompactSyntaxParser.py', self.protoFile, dir])
            xml_file = xml_file.strip()
            class_name = 'GeneratedModel'
            code = subprocess.check_output(['./python/pycml/translate.py', '-t', 'Python', '-p', '--Wu', '--protocol=' + xml_file, 'projects/FunctionalCuration/cellml/' + model, '-c', class_name, '-o', '-'])
            module = imp.new_module(class_name)
            exec code in module.__dict__
            for name in module.__dict__.keys():
                if name.startswith(class_name):
                    model = getattr(module, name)()
            for sim in self.simulations:
                sim.SetModel(model)
        else:
            for sim in self.simulations:
                sim.SetModel(model)
        
    def GetPath(self, basePath, path):
        return os.path.join(os.path.dirname(basePath), path)
