
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

import operator
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
from ..language.statements import Assign

from ..simulations import simulations

# NB: Do not import the CompactSyntaxParser here, or we'll get circular imports.
# Only import it within methods that use it.

class Protocol(object):
    """This class represents a protocol in the functional curation 'virtual experiment' language.
    
    It gives the central interface to functional curation, handling parsing a protocol description
    from file and running it on a given model.
    """
    def __init__(self, protoFile, indentLevel=0):
        """Construct a new protocol by parsing the description in the given file.
        
        The protocol must be specified using the textual syntax, as defined by the CompactSyntaxParser module.
        """
        self.indentLevel = indentLevel
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
        self.inputs = []
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
        self.inputs = details.get('inputs', [])
        self.inputEnv.ExecuteStatements(self.inputs)
        for prefix, path, set_inputs in details.get('imports', []):
            self.LogProgress('Importing', path, 'as', prefix, 'in', self.protoName)
            imported_proto = Protocol(self.GetPath(protoFile, path))
            if prefix == "":
                # TODO: Ensure all environment delegatees are correct!
                # Merge inputs of the imported protocol into our own (duplicate names are an error here).
                # Override any values specified in the import statement itself.
                for stmt in imported_proto.inputs:
                    name = stmt.names[0]
                    if name in set_inputs:
                        stmt = Assign([name], set_inputs[name])
                    self.inputEnv.ExecuteStatements([stmt])
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
        self.libraryEnv.SetDelegateeEnv(self.inputEnv)
        self.library.extend(details.get('library', []))
        self.simulations.extend(details.get('simulations', []))
        self.postProcessing.extend(details.get('postprocessing', []))
        self.outputs.extend(details.get('outputs', []))
        self.plots.extend(details.get('plots', []))
        self.timings['parsing'] = time.time() - start


    # Override Object serialization methods to allow pickling with the dill module
    def __getstate__(self):
        # TODO: Original object unusable after serialization.
        # Should either maintain object state (i.e., remove reference to simulations
        # in copied dict and re-initialize in __setstate__) or dynamically restore
        # simulation model state at runtime.

        # Must remove Model class and regenerate during unpickling
        # (Pickling errors from nested class structure of ModelWrapperEnvironment)
        for sim in self.simulations:
            # Undo Simulation.SetModel()
            if sim.model is not None:
                modelenv = sim.model.GetEnvironmentMap()
                for prefix,env in modelenv.iteritems():
                    # Must clear references to model environment from nestedSim as well
                    if isinstance(sim,simulations.Nested):
                        sim.nestedSim.env.ClearDelegateeEnv(prefix)
                        # No need to clear the nestedSim.results, as the nesting simulation maintains
                        # a reference to it which is cleared two lines below this comment
                    sim.env.ClearDelegateeEnv(prefix)
                    sim.results.ClearDelegateeEnv(prefix)
                # If the protocol has been run, remove references to model environment
                # in the simulations
                if "" in sim.env.delegatees:
                    sim.env.ClearDelegateeEnv("")
                if sim.prefix and sim.prefix in self.libraryEnv.delegatees:
                    self.libraryEnv.ClearDelegateeEnv(sim.prefix)
                sim.model = None
                # ...and their nested simulations, if applicable
                if isinstance(sim,simulations.Nested):
                    if "" in sim.nestedSim.env.delegatees:
                        sim.nestedSim.env.ClearDelegateeEnv("")
                    if sim.nestedSim.prefix and sim.nestedSim.prefix in self.libraryEnv.delegatees:
                        self.libraryEnv.ClearDelegateeEnv(sim.nestedSim.prefix)
                    sim.nestedSim.model = None
        odict = self.__dict__.copy()
        # Remove Model and CSP from Protocol
        if 'model' in odict:
            del odict['model']
        if 'parser' in odict:
            del odict['parser']
            del odict['parsedProtocol']
        return odict
    def __setstate__(self,dict):
        self.__dict__.update(dict)
        # Re-import Model from temporary Python file
        sys.path.insert(0, self.modelPath)
        import model as module
        for name in module.__dict__.keys():
            if name.startswith('GeneratedModel'):
                model = getattr(module, name)()
                model._module = module
        del sys.modules['model']
        self.model = model
        for sim in self.simulations:
            sim.SetModel(model)


    def AddImportedProtocol(self, proto, prefix):
        """Add a protocol imported with a prefix to our collection.

        This also makes that protocol's library (if any) available as a delegatee of our own.
        """
        assert prefix
        if prefix in self.imports:
            raise ProtocolError("The prefix '", prefix, "' has already been used for an imported protocol.")
        self.imports[prefix] = proto
        self.libraryEnv.SetDelegateeEnv(proto.libraryEnv, prefix)

    def Initialise(self,verbose=True):
        """(Re-)Initialise this protocol, ready to be run on a model."""
        if verbose:
            self.LogProgress('Initialising', self.protoName)
        self.libraryEnv.Clear()
        self.postProcessingEnv.Clear()
        self.outputEnv.Clear()
        for sim in self.simulations:
            sim.Clear()
            sim.SetIndentLevel(self.indentLevel + 1)
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
        if not self.outputFolder and verbose and writeOut:
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
                x_data.append(self.outputEnv.LookUp(plot['x']).array)
                y_data.append(self.outputEnv.LookUp(plot['y']).array)
                if 'key' in plot:
                    key_data = self.outputEnv.LookUp(plot['key']).array
                    if key_data.ndim != 1:
                        raise ProtocolError('Plot key variables must be 1d vectors;', plot['key'], 'has', key_data.ndim, 'dimensions')
                # Check the x-axis data shape.  It must either be 1d, or be equivalent to a 1d vector (i.e. stacked copies of the same vector).
                for i, x in enumerate(x_data):
                    if x.ndim > 1:
                        num_repeats = reduce(operator.mul, x.shape[:-1])
                        x_2d = x.reshape((num_repeats,x.shape[-1])) # Flatten all extra dimensions as an array view
                        if x_2d.ptp(axis=0).any():
                            # There was non-zero difference between the min & max at some position in the 1d equivalent vector
                            raise ProtocolError('The X data for a plot must be (equivalent to) a 1d array, not', x.ndim, 'dimensions')
                        x_data[i] = x_2d[0] # Take just the first copy
                # Plot the data
                fig = plt.figure()
                for i, x in enumerate(x_data):
                    y = y_data[i]
                    if y.ndim > 1:
                        # Matplotlib can handle 2d data, but plots columns not rows, so we need to flatten & transpose
                        y_2d = y.reshape((reduce(operator.mul, y.shape[:-1]),y.shape[-1]))
                        plt.plot(x, y_2d.T)
                    else:
                        plt.plot(x, y)
                    plt.title(plot['title'])
                    plt.xlabel(plot_descriptions[plot['x']])
                    plt.ylabel(plot_descriptions[plot['y']])
                plt.savefig(os.path.join(self.outputFolder.path, self.SanitiseFileName(plot['title']) + '.png'))
                plt.close()
        self.timings['create plots'] = self.timings.get('plot', 0.0) + (time.time() - start)
    
    def SanitiseFileName(self, name):
        """Simply transform a name such as a graph title into a valid file name."""
        name = name.strip().replace(' ', '_')
        keep = ('.', '_')
        return ''.join(c for c in name if c.isalnum() or c in keep)

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
        self.Initialise(verbose)
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
        # Summarise time spent in each protocol section (if we're the main protocol)
        if verbose and self.indentLevel == 0:
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
            sim.Run(verbose)
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
                             useCython=True, useNumba=False, exposeNamedParameters=False):
        """Return the command to translate a modified CellML model to Python code, optionally optimised with numba or Cython."""
        if useCython:
            model_py_file = os.path.join(tempDir, 'model.pyx')
            target = 'Cython'
        else:
            model_py_file = os.path.join(tempDir, 'model.py')
            target = 'Python'
        code_gen_cmd = ['./python/pycml/translate.py', '-t', target, '-p', '--Wu',
                        '--protocol=' + xmlFile,  model, '-c', className, '-o', model_py_file]
        if exposeNamedParameters:
            # Allow the parameter fitting code to adjust the value of any annotated constant variable
            code_gen_cmd.append('--expose-named-parameters')
        if not useCython and not useNumba:
            code_gen_cmd.append('--no-numba')
        return code_gen_cmd

    def SetModel(self, model, useNumba=False, useCython=True, exposeNamedParameters=False):
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
            code_gen_cmd = self.GetConversionCommand(model, xml_file, class_name, temp_dir,
                                                     useCython=useCython, useNumba=useNumba, exposeNamedParameters=exposeNamedParameters)
            print subprocess.check_output(code_gen_cmd, stderr=subprocess.STDOUT)
            if useCython:
                # Compile the extension module
                print subprocess.check_output(['python', 'setup.py', 'build_ext', '--inplace'], cwd=temp_dir, stderr=subprocess.STDOUT)
            # Create an instance of the model
            self.modelPath = temp_dir
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
    
    def SetIndentLevel(self, indentLevel):
        """Set the level of indentation to use for progress output."""
        self.indentLevel = indentLevel

    def LogProgress(self, *args):
        """Print a progress line showing how far through the protocol we are.
        
        Arguments are converted to strings and space separated, as for the print builtin.
        """
        print '  ' * self.indentLevel + ' '.join(map(str, args))
        sys.stdout.flush()

    def LogWarning(self, *args):
        """Print a warning message.
        
        Arguments are converted to strings and space separated, as for the print builtin.
        """
        print >>sys.stderr, '  ' * self.indentLevel + ' '.join(map(str, args))
        sys.stderr.flush()
