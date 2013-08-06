
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
from tables import *
import shutil
from AbstractValue import AbstractValue

csp = CSP.CompactSyntaxParser

class Protocol(object):
    """Base class for protocols in the protocol language."""
    def __init__(self, protoFile):
        self.outputPath = None
        self.protoFile = protoFile
        self.env = Env.Environment()
        self.inputEnv = Env.Environment(allowOverwrite=True)
        self.libraryEnv = Env.Environment()
        self.outputEnv = Env.Environment()
        self.postProcessingEnv = Env.Environment(delegatee=self.libraryEnv)
        self.library = []
        self.simulations = []
        self.postProcessing = []
        self.outputs = [] #details.get outputs is going to be a list of dictionaries, if there's 
        # a ref then look that up, otherwise just look up the name in the postprocessing env and
        # define those names in outputenv as the value that you get from postprocessingenv
        self.plots = [] #details.get plots returns a list of dictionaries too that map plot title
        # to a list of curves, 
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
        self.outputs.extend(details.get('outputs', []))      
        self.plots.extend(details.get('plots', []))

    def Initialise(self):
        self.libraryEnv.Clear()
        self.postProcessingEnv.Clear()
        self.outputEnv.Clear()
        for sim in self.simulations:
            sim.env.Clear()
            sim.results.Clear()
        
    def SetOutputFolder(self, path):
        if os.path.isdir(path) and path.startswith('/tmp'):
            shutil.rmtree(path)
        os.mkdir(path)
        self.outputPath = path
    
    def OutputsAndPlots(self):
        plot_vars = []
        plot_descriptions = {}
        for plot in self.plots:
            plot_vars.append(plot['x'])
            plot_vars.append(plot['y'])
        for output in self.outputs:
            if 'ref' in output:
                self.outputEnv.DefineName(output['name'], self.postProcessingEnv.LookUp(output['ref']))
            else:
                self.outputEnv.DefineName(output['name'], self.postProcessingEnv.LookUp(output['name']))
            if output['name'] in plot_vars:
                if output['description']:
                    plot_descriptions[output['name']] = output['description']
                else:
                    plot_descriptions[output['name']] = output['name']
        if not self.outputPath:
            print >>sys.stderr, "Warning: protocol output folder not set, so not writing outputs to file"
            return
        filename = 'output file'
        h5file = open_file(filename, mode='w', title=self.plots[0]['title'])
        group = h5file.create_group('/', 'output', 'output parent')
        for output in self.outputs:
            if not 'description' in output:
                output['description'] = output['name']
            h5file.create_array(group, output['name'], self.outputEnv.unwrappedBindings[output['name']],
                                title=output['description'])
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pylab
        for plot in self.plots:
            x_data = []
            y_data = []
            x_data.append(self.outputEnv.LookUp(plot['x']))
            y_data.append(self.outputEnv.LookUp(plot['y']))
            
            # Plot the data.
            fig, host = plt.subplots()
            for i,x in enumerate(x_data):
                if y_data[i].array.ndim > 1:
                    for j in range(y_data[i].array.shape[0]):
                        host.plot(x.array, y_data[i].array[j])
                else:
                    host.plot(x.array, y_data[i].array)
                plt.title(plot['title'])
                plt.xlabel(plot_descriptions[plot['x']])
                plt.ylabel(plot_descriptions[plot['y']])  
            plt.savefig(self.outputPath + '/' + plot['title'] + '.png')
            plt.close()
        h5file.close()
        
#         for output in self.outputs:
#             if 'ref' in output:
#                 print 'ref', output['ref']
#                 self.outputEnv.DefineName(output['name'], self.postProcessingEnv.LookUp(output['ref']))
#             else:
#                 print 'name', output['name']
#                 self.outputEnv.DefineName(output['name'], self.postProcessingEnv.LookUp(output['name']))
#         filename = 'output file'
#         h5file = open_file(filename, mode='w', title=self.plots[0]['title'])
#         group = h5file.create_group('/', 'output', 'output parent')
#         for output in self.outputs:
#             if not 'description' in output:
#                 output['description'] = output['name']
#             h5file.create_array(group, output['name'], self.outputEnv.unwrappedBindings[output['name']],
#                                 title=output['description'])
#          
#         import matplotlib
#         matplotlib.use('Agg')
#         import matplotlib.pyplot as plt
#         import pylab
#         for plot in self.plots:
#             x_data = h5file.get_node('/output/' + plot['x'])
#             y_data = h5file.get_node('/output/' + plot['y'])
#             # Plot the data.
#             fig, host = plt.subplots()
#             if y_data.ndim > 1:
#                 for i in range(y_data.shape[0]):
#                     host.plot(x_data, y_data[i])
#             else:
#                 host.plot(x_data, y_data)
#             plt.title(plot['title'])
#             plt.xlabel(h5file.get_node_attr('/output/' + plot['x'], 'TITLE'))
#             plt.ylabel(h5file.get_node_attr('/output/' + plot['y'], 'TITLE'))  
#             plt.savefig(self.outputPath + '/' + plot['title'] + '.png')
#         
#         h5file.close()

    def Run(self):
        self.Initialise()
        try:
            for sim in self.simulations:
                sim.env.SetDelegateeEnv(self.libraryEnv)
                if sim.prefix:
                    self.libraryEnv.SetDelegateeEnv(sim.results, sim.prefix)
            self.libraryEnv.ExecuteStatements(self.library)     
            for sim in self.simulations:
                sim.Initialise()   
                results = sim.Run()
            self.postProcessingEnv.ExecuteStatements(self.postProcessing)
            if self.plots:
                self.OutputsAndPlots();            
            

            # save outputs to disk as hdf5
            # plot, save png to disk within folder in setoutputfolder
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
        if isinstance(valueExpr, AbstractValue):
            value = valueExpr
        else:
            value = valueExpr.Evaluate(self.inputEnv)
        self.inputEnv.OverwriteDefinition(name, value)
        
    def SetModel(self, model, useNumba=True):
        if isinstance(model, str):
            import tempfile, subprocess, imp, sys
            dir = tempfile.mkdtemp()
            xml_file = subprocess.check_output(['python', 'projects/FunctionalCuration/src/proto/parsing/CompactSyntaxParser.py', self.protoFile, dir])
            xml_file = xml_file.strip()
            model_py_file = os.path.join(dir, 'model.py')
            class_name = 'GeneratedModel'
            code_gen_cmd = ['./python/pycml/translate.py', '-t', 'Python', '-p', '--Wu',
                            '--protocol=' + xml_file,  model, '-c', class_name, '-o', model_py_file]
            if not useNumba:
                code_gen_cmd.append('--no-numba')
            code = subprocess.check_output(code_gen_cmd)
            sys.path.insert(0, dir)
            import model as module
#             module = imp.new_module(class_name)
#             exec code in module.__dict__
            for name in module.__dict__.keys():
                if name.startswith(class_name):
                    model = getattr(module, name)()
                    model._module = module
                    #model._code = code
            del sys.modules['model']
#             try:
#                 shutil.rmtree(dir)  # delete directory
#             except OSError as exc:
#                 if exc.errno != 2:  # code 2 - no such file or directory
#                     raise  # re-raise exception
        for sim in self.simulations:
            sim.SetModel(model)
        self.model = model
        
    def GetPath(self, basePath, path):
        return os.path.join(os.path.dirname(basePath), path)
