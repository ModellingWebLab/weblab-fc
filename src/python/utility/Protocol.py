
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

csp = CSP.CompactSyntaxParser

class Protocol(object):
    """Base class for protocols in the protocol language."""
    def __init__(self, protoFile):
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

    def SetOutputFolder(self, path):
        if os.path.isdir(path) and path.startswith('/tmp'):
            pass
#             fileList = os.listdir(path)
#             for fileName in fileList:
#                 os.remove(path+"/"+fileName)
        else:
            os.mkdir(path)
    
    def OutputsAndPlots(self):
        print 'plots', self.plots
        print 'outputs', self.outputs
        for output in self.outputs:
            if 'ref' in output:
                print 'ref', output['ref']
                self.outputEnv.DefineName(output['name'], self.postProcessingEnv.LookUp(output['ref']))
            else:
                print 'name', output['name']
                self.outputEnv.DefineName(output['name'], self.postProcessingEnv.LookUp(output['name']))
        class Output(IsDescription):
            x = Float32Col()
            y = Float32Col()
        filename = 'temp file'
        h5file = open_file(filename, mode='w', title=self.plots[0]['title'])
        group = h5file.create_group('/', 'detector', 'detector information')
        table = h5file.create_table(group, 'readout', Output, "readout example")
        output = table.row
#         output['x'] = self.outputEnv.LookUp(self.plots[0]['x'])
#         output['y'] = self.outputEnv.LookUp(self.plots[0]['y'])
#        import h5py
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.cm as cm
        
#         f = h5py.File('temp file','r')
#         arr = f["x"][:,:,:]
#         f.close()
        # Create some data to plot
        x_data = range(20)
        y_data = range(20)
        # Plot the data.
        #plt.plot(x_data, y_data)
        # Add some axis labels.
        #plt.set_xlabel("x")
        #plt.set_ylabel("y")
        # Produce an image.
        #fig.savefig("lineplot.png")    
        #plt.show()
        h5file.close()

#    3 # Define a user record to characterize some kind of particles
#    4 class Particle(IsDescription):
#    5     name      = StringCol(16)   # 16-character String
#    6     idnumber  = Int64Col()      # Signed 64-bit integer
#    7     ADCcount  = UInt16Col()     # Unsigned short integer
#    8     TDCcount  = UInt8Col()      # unsigned byte
#    9     grid_i    = Int32Col()      # integer
#   10     grid_j    = Int32Col()      # integer
#   11     pressure  = Float32Col()    # float  (single-precision)
#   12     energy    = FloatCol()      # double (double-precision)
#   13 
#   14 filename = "test.h5"
#   15 # Open a file in "w"rite mode
#   16 h5file = open_file(filename, mode = "w", title = "Test file")
#   17 # Create a new group under "/" (root)
#   18 group = h5file.create_group("/", 'detector', 'Detector information')
#   19 # Create one table on it
#   20 table = h5file.create_table(group, 'readout', Particle, "Readout example")
#   21 # Fill the table with 10 particles
#   22 particle = table.row
#   23 for i in xrange(10):
#   24     particle['name']  = 'Particle: %6d' % (i)
#   25     particle['TDCcount'] = i % 256
#   26     particle['ADCcount'] = (i * 256) % (1 << 16)
#   27     particle['grid_i'] = i
#   28     particle['grid_j'] = 10 - i
#   29     particle['pressure'] = float(i*i)
#   30     particle['energy'] = float(particle['pressure'] ** 4)
#   31     particle['idnumber'] = i * (2 ** 34)
#   32     # Insert a new particle record
#   33     particle.append()
#   34 # Close (and flush) the file
#   35 h5file.close()

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
        value = valueExpr.Evaluate(self.inputEnv)
        self.inputEnv.OverwriteDefinition(name, value)
        
    def SetModel(self, model):
        if isinstance(model, str):
            import tempfile, subprocess, imp, shutil, sys
            dir = tempfile.mkdtemp()
            xml_file = subprocess.check_output(['python', 'projects/FunctionalCuration/src/proto/parsing/CompactSyntaxParser.py', self.protoFile, dir])
            xml_file = xml_file.strip()
            model_py_file = os.path.join(dir, 'model.py')
            class_name = 'GeneratedModel'
            code = subprocess.check_output(['./python/pycml/translate.py', '-t', 'Python', '-p', '--Wu',
                                            '--protocol=' + xml_file,
                                            model, '-c', class_name, '-o', '-'])
#             sys.path.insert(0, dir)
#             import model as GeneratedModelModule
            module = imp.new_module(class_name)
            exec code in module.__dict__
            for name in module.__dict__.keys():
                if name.startswith(class_name):
                    model = getattr(module, name)()
                    model._module = module
                    model._code = code
            for sim in self.simulations:
                sim.SetModel(model)
#             try:
#                 shutil.rmtree(dir)  # delete directory
#             except OSError as exc:
#                 if exc.errno != 2:  # code 2 - no such file or directory
#                     raise  # re-raise exception
        else:
            for sim in self.simulations:
                sim.SetModel(model)
        
    def GetPath(self, basePath, path):
        return os.path.join(os.path.dirname(basePath), path)
