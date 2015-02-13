
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

import os
import shutil
import sys
import tables
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab

from . import environment as Env
from .error_handling import ProtocolError, ErrorRecorder
from .file_handling import OutputFolder
from .locatable import Locatable
from ..language import values as V

# NB: Do not import the CompactSyntaxParser here, or we'll get circular imports.
# Only import it within methods that use it.


class Protocol(object):
    """This class represents a protocol in the functional curation 'virtual experiment' language.
    
    It gives the central interface to functional curation, handling parsing a protocol description
    from file and running it on a given model.
    """
    def __init__(self, protoFile):
        """Construct a new protocol by parsing the description in the given file.
        
        The protocol must be specified using the textual syntax, as defined by the CompactSyntaxParser module.
        """
        self.outputFolder = None
        self.protoFile = protoFile
        self.protoName = os.path.basename(self.protoFile)
        self.timings = {}
        self.LogProgress('Constructing', self.protoName)
        # Main environments used when running the protocol
        self.env = Env.Environment()
        self.inputEnv = Env.Environment(allowOverwrite=True)
        self.libraryEnv = Env.Environment()
        self.outputEnv = Env.Environment()
        self.postProcessingEnv = Env.Environment(delegatee=self.libraryEnv)
        # The elements making up this protocol's definition
        self.imports = {}
        self.library = []
        self.simulations = []
        self.postProcessing = []
        self.outputs = []
        self.plots = []

        # Parse the protocol file and fill in the structures declared above
        start = time.time()
        import CompactSyntaxParser as CSP
        parser = self.parser = CSP.CompactSyntaxParser()
        CSP.Actions.source_file = protoFile
        generator = self.parsedProtocol = parser._Try(CSP.CompactSyntaxParser.protocol.parseFile, protoFile, parseAll=True)[0]
        assert isinstance(generator, CSP.Actions.Protocol)
        details = generator.expr()
        assert isinstance(details, dict)
        for prefix, path in details.get('imports', []):
            self.LogProgress('Importing', path, 'as', prefix, 'in', self.protoName)
            imported_proto = Protocol(self.GetPath(protoFile, path))
            # TODO: setting inputs for imported protocols!
            if prefix == "":
                # Make any prefixed imports of that protocol into our prefixed imports
                for imported_prefix, imported_import in imported_proto.imports.iteritems():
                    self.AddImportedProtocol(imported_import, imported_prefix)
                # Merge the other elements of its definition with our own
                self.library.extend(imported_proto.library)
                self.simulations.extend(imported_proto.simulations)
                self.postProcessing.extend(imported_proto.postProcessing)
                self.outputs.extend(imported_proto.outputs)
                self.plots.extend(imported_proto.plots)
            else:
                self.AddImportedProtocol(imported_proto, prefix)
        self.inputEnv.ExecuteStatements(details.get('inputs', []))
        self.libraryEnv.SetDelegateeEnv(self.inputEnv)
        self.library.extend(details.get('library', []))
        self.simulations.extend(details.get('simulations', []))
        self.postProcessing.extend(details.get('postprocessing', []))
        self.outputs.extend(details.get('outputs', []))
        self.plots.extend(details.get('plots', []))
        self.timings['parsing'] = time.time() - start

    def AddImportedProtocol(self, proto, prefix):
        """Add a protocol imported with a prefix to our collection.

        This also makes that protocol's library (if any) available as a delegatee of our own.
        """
        if prefix in self.imports:
            raise ProtocolError("The prefix '", prefix, "' has already been used for an imported protocol.")
        self.imports[prefix] = proto
        self.libraryEnv.SetDelegateeEnv(proto.libraryEnv, prefix)

    def Initialise(self):
        """(Re-)Initialise this protocol, ready to be run on a model."""
        self.LogProgress('Initialising', self.protoName)
        self.libraryEnv.Clear()
        self.postProcessingEnv.Clear()
        self.outputEnv.Clear()
        for sim in self.simulations:
            sim.Clear()
        for imported_proto in self.imports.values():
            imported_proto.Initialise()

    def SetOutputFolder(self, path):
        """Specify where the outputs from this protocol will be written to disk."""
        if isinstance(path, OutputFolder):
            self.outputFolder = path
        else:
            self.outputFolder = OutputFolder(path)
    
    def OutputsAndPlots(self, errors, verbose=True, writeOut=True):
        """Save the protocol outputs to disk, and generate the requested plots."""
        # Copy protocol outputs into the self.outputs environment,
        # and capture output descriptions needed by plots in the process.
        plot_vars = []
        plot_descriptions = {}
        for plot in self.plots:
            plot_vars.append(plot['x'])
            plot_vars.append(plot['y'])
        outputs_defined = []
        for output_spec in self.outputs:
            with errors:
                output_var = output_spec.get('ref', output_spec['name'])
                output = self.postProcessingEnv.LookUp(output_var)
                self.outputEnv.DefineName(output_spec['name'], output)
                outputs_defined.append(output_spec)
                if not 'description' in output_spec:
                    output_spec['description'] = output_spec['name']
                if output_spec['name'] in plot_vars:
                    plot_descriptions[output_spec['name']] = output_spec['description']
        if not self.outputFolder:
            self.LogWarning("Warning: protocol output folder not set, so not writing outputs to file")
            return

        if verbose:
            self.LogProgress('saving output data to h5 file...')
        start = time.time()
        if writeOut:
            with errors:
                filename = os.path.join(self.outputFolder.path, 'output.h5')
                h5file = tables.open_file(filename, mode='w', title=os.path.splitext(os.path.basename(self.protoFile))[0])
                group = h5file.create_group('/', 'output', 'output parent')
                for output_spec in outputs_defined:
                    h5file.create_array(group, output_spec['name'], self.outputEnv.unwrappedBindings[output_spec['name']], title=output_spec['description'])
                h5file.close()
        self.timings['save outputs'] = self.timings.get('output', 0.0) + (time.time() - start)

        # Plots
        start = time.time()
        for plot in self.plots:
            with errors:
                if verbose:
                    self.LogProgress('plotting', plot['title'], 'curve:', plot_descriptions[plot['y']], 'against', plot_descriptions[plot['x']])
                x_data = []
                y_data = []
                x_data.append(self.outputEnv.LookUp(plot['x']))
                y_data.append(self.outputEnv.LookUp(plot['y']))
                # Plot the data
                fig = plt.figure()
                for i, x in enumerate(x_data):
                    if y_data[i].array.ndim > 1:
                        for j in range(y_data[i].array.shape[0]):
                            plt.plot(x.array, y_data[i].array[j])
                    else:
                        plt.plot(x.array, y_data[i].array)
                    plt.title(plot['title'])
                    plt.xlabel(plot_descriptions[plot['x']])
                    plt.ylabel(plot_descriptions[plot['y']])
                plt.savefig(os.path.join(self.outputFolder.path, plot['title'] + '.png'))
                plt.close()
        self.timings['create plots'] = self.timings.get('plot', 0.0) + (time.time() - start)
    
    def ExecuteLibrary(self):
        """Run the statements in our library to build up the library environment.
        
        The libraries of any imported protocols will be executed first.
        """
        for imported_proto in self.imports.values():
            imported_proto.ExecuteLibrary()
        self.libraryEnv.ExecuteStatements(self.library)

    def Run(self, verbose=True, writeOut=True):
        """Run this protocol on the model already specified using SetModel."""
        Locatable.outputFolder = self.outputFolder
        self.Initialise()
        if verbose:
            self.LogProgress('running protocol', self.protoName, '...')
        errors = ErrorRecorder(self.protoName)
        with errors:
            for sim in self.simulations:
                sim.env.SetDelegateeEnv(self.libraryEnv)
                if sim.prefix:
                    if self.outputFolder:
                        sim.SetOutputFolder(self.outputFolder.CreateSubfolder('simulation_' + sim.prefix))
                    self.libraryEnv.SetDelegateeEnv(sim.results, sim.prefix)
            start = time.time()
            self.ExecuteLibrary()
            self.timings['run library'] = self.timings.get('library', 0.0) + (time.time() - start)
        with errors:
            start = time.time()
            self.RunSimulations(verbose)
            self.timings['simulations'] = self.timings.get('simulations', 0.0) + (time.time() - start)
        with errors:
            start = time.time()
            self.RunPostProcessing(verbose)
            self.timings['post-processing'] = self.timings.get('post-processing', 0.0) + (time.time() - start)
        self.OutputsAndPlots(errors, verbose, writeOut)
        # Summarise time spent in each protocol section
        if verbose:
            print 'Time spent running protocol (s): %.6f' % sum(self.timings.values())
            max_len = max(len(section) for section in self.timings)
            for section, duration in self.timings.iteritems():
                print '   ', section, ' ' * (max_len - len(section)), '%.6f' % duration
        if errors:
            # Report any errors that occurred
            raise errors

    def RunSimulations(self, verbose=True):
        """Run the model simulations specified in this protocol."""
        for sim in self.simulations:
            if verbose:
                self.LogProgress('running simulation', sim.prefix)
            sim.Initialise()
            sim.Run()
            # Reset trace folder in case a nested protocol changes it
            Locatable.outputFolder = self.outputFolder

    def RunPostProcessing(self, verbose=True):
        """Run the post-processing section of this protocol."""
        if verbose:
            self.LogProgress('running post processing for', self.protoName, '...')
        self.postProcessingEnv.ExecuteStatements(self.postProcessing)

    def SetInput(self, name, valueExpr):
        """Overwrite the value of a protocol input.
        
        The value may be given either as an actual value, or as an expression which will be evaluated in
        the context of the existing inputs.
        """
        if isinstance(valueExpr, V.AbstractValue):
            value = valueExpr
        else:
            value = valueExpr.Evaluate(self.inputEnv)
        self.inputEnv.OverwriteDefinition(name, value)

    def GetConversionCommand(self, model, xmlFile, className, tempDir,
                             useCython=True, useNumba=False):
        """Return the command to translate a modified CellML model to Python code, optionally optimised with numba or Cython."""
        if useCython:
            model_py_file = os.path.join(tempDir, 'model.pyx')
            target = 'Cython'
        else:
            model_py_file = os.path.join(tempDir, 'model.py')
            target = 'Python'
        code_gen_cmd = ['./python/pycml/translate.py', '-t', target, '-p', '--Wu',
                        '--protocol=' + xmlFile,  model, '-c', className, '-o', model_py_file]
        if not useCython and not useNumba:
            code_gen_cmd.append('--no-numba')
        return code_gen_cmd

    def SetModel(self, model, useNumba=True, useCython=True):
        """Specify the model this protocol is to be run on."""
        start = time.time()
        if isinstance(model, str):
            self.LogProgress('generating model code...')
            import tempfile, subprocess, imp, sys
            if self.outputFolder:
                temp_dir = tempfile.mkdtemp(dir=self.outputFolder.path)
            else:
                temp_dir = tempfile.mkdtemp()
            # Create an XML syntax version of the protocol, for PyCml's sake :(
            import CompactSyntaxParser as CSP
            CSP.DoXmlImports()
            xml_file = self.parser.ConvertProtocol(self.protoFile, temp_dir, xmlGenerator=self.parsedProtocol)
            # Generate the (protocol-modified) model code
            class_name = 'GeneratedModel'
            code_gen_cmd = self.GetConversionCommand(model, xml_file, class_name, temp_dir, useCython=useCython, useNumba=useNumba)
            print subprocess.check_output(code_gen_cmd, stderr=subprocess.STDOUT)
            if useCython:
                # Compile the extension module
                print subprocess.check_output(['python', 'setup.py', 'build_ext', '--inplace'], cwd=temp_dir, stderr=subprocess.STDOUT)
            # Create an instance of the model
            sys.path.insert(0, temp_dir)
            import model as module
            for name in module.__dict__.keys():
                if name.startswith(class_name):
                    model = getattr(module, name)()
                    model._module = module
            del sys.modules['model']
        self.model = model
        for sim in self.simulations:
            sim.SetModel(model)
        self.timings['load model'] = self.timings.get('load model', 0.0) + (time.time() - start)

    def GetPath(self, basePath, path):
        """Determine the full path of an imported protocol file.
        
        Relative paths are resolved relative to basePath (the path to this protocol) by default.
        If this does not yield an existing file, they are resolved relative to the built-in library folder instead.
        """
        new_path = os.path.join(os.path.dirname(basePath), path)
        if not os.path.isabs(path) and not os.path.exists(new_path):
            # Search in the library folder instead
            library = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir,
                                   'proto', 'library')
            new_path = os.path.join(library, path)
        return new_path
    
    def LogProgress(self, *args):
        """Print a progress line showing how far through the protocol we are.
        
        Arguments are converted to strings and space separated, as for the print builtin.
        """
        print ' '.join(map(str, args))
        sys.stdout.flush()

    def LogWarning(self, *args):
        """Print a warning message.
        
        Arguments are converted to strings and space separated, as for the print builtin.
        """
        print >>sys.stderr, ' '.join(map(str, args))
        sys.stderr.flush()
