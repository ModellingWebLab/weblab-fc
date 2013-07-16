
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
CSP.ImportPythonImplementation()
csp = CSP.CompactSyntaxParser
import Environment as Env
import os

class Protocol(object):
    
    def __init__(self, protoFile):
        self.protoFile = protoFile
        self.env = Env.Environment()
        self.libraryEnv = Env.Environment() #is where its own library is executed
        self.postProcessingEnv = Env.Environment(delegatee=self.libraryEnv) #for own postprocessing which delegates to library env
        self.library = []
        self.postProcessing = []
        parser = csp()
        CSP.Actions.source_file = protoFile
        generator = parser._Try(csp.protocol.parseFile, protoFile, parseAll=True)[0]
        assert isinstance(generator, CSP.Actions.Protocol)
        details = generator.expr()
        assert isinstance(details, dict)
        # if prefix is empty then do below, if not then do self.libraryenv.setdelegatee(importedproto.libraryenv, prefix)
        for prefix, path in details.get('imports', []):
            # if prefix is not empty then do importedproto.libraryenv.executestaement(importedproto.library)        
            imported_proto = Protocol(self.GetPath(protoFile, path))
            if prefix == "":    
                self.library.extend(imported_proto.library)
                self.postProcessing.extend(imported_proto.postProcessing)
            else:
                self.libraryEnv.SetDelegateeEnv(imported_proto.libraryEnv, prefix)
                imported_proto.libraryEnv.ExecuteStatements(imported_proto.library)
        self.library.extend(details.get('library', []))
        self.postProcessing.extend(details.get('postprocessing', []))      

    def Run(self):
        self.libraryEnv.ExecuteStatements(self.library)
        self.postProcessingEnv.ExecuteStatements(self.postProcessing)
        
    def GetPath(self, basePath, path):
        return os.path.join(os.path.dirname(basePath), path)
       # join and dirname on protoFile and path in details, joins dirnmae of basepath and path
