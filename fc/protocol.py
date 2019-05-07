import operator
import os
import subprocess
import sys
import tables
import tempfile
import time
from functools import reduce

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.switch_backend('agg')  # on some machines this is required to avoid "Invalid DISPLAY variable" errors

import fc   # noqa: E402
from .environment import Environment    # noqa: E402
from .error_handling import ProtocolError, ErrorRecorder  # noqa: E402
from .file_handling import OutputFolder  # noqa: E402
from .language import values as V  # noqa: E402
from .language.statements import Assign  # noqa: E402
from .locatable import Locatable  # noqa: E402

# NB: Do not import the CompactSyntaxParser here, or we'll get circular imports.
# Only import it within methods that use it.


# Setup script
SETUP_PY = '''
import numpy

from setuptools import setup
from cython import inline
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension


SUNDIALS_MAJOR = inline(\'''
    cdef extern from *:
        """
        #include <sundials/sundials_config.h>

        #ifndef SUNDIALS_VERSION_MAJOR
            #define SUNDIALS_VERSION_MAJOR 2
        #endif
        """
        int SUNDIALS_VERSION_MAJOR

    return SUNDIALS_VERSION_MAJOR
    \''')

ext_modules=[
    Extension(
        '%(module_name)s',
        sources=['%(model_file)s'],
        include_dirs=[numpy.get_include(), '%(fcpath)s'],
        libraries=['sundials_cvode', 'sundials_nvecserial', 'm'],
        cython_compile_time_env={'FC_SUNDIALS_MAJOR': SUNDIALS_MAJOR},
    ),
]

setup(
    name='%(module_name)s',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
'''


class Protocol(object):
    """This class represents a protocol in the functional curation 'virtual experiment' language.

    It gives the central interface to functional curation, handling parsing a protocol description
    from file and running it on a given model.
    """

    def __init__(self, proto_file, indent_level=0):
        """Construct a new protocol by parsing the description in the given file.

        The protocol must be specified using the textual syntax, as defined by the CompactSyntaxParser module.
        """
        self.indent_level = indent_level
        self.output_folder = None
        self.proto_file = proto_file
        self.proto_name = os.path.basename(self.proto_file)
        self.timings = {}
        self.log_progress('Constructing', self.proto_name)
        # Main environments used when running the protocol
        self.env = Environment()
        self.input_env = Environment(allow_overwrite=True)
        self.input_env.define_name("load", V.LoadFunction(os.path.dirname(self.proto_file)))
        self.library_env = Environment()
        self.output_env = Environment()
        self.post_processing_env = Environment(delegatee=self.library_env)
        # The elements making up this protocol's definition
        self.inputs = []
        self.imports = {}
        self.library = []
        self.simulations = []
        self.post_processing = []
        self.outputs = []
        self.plots = []

        # Information from the model interface
        self.model_interface = []

        # A mapping of namespace names to uris, for namespaces used in the
        # protocol
        self.ns_map = {}

        # Parse the protocol file and fill in the structures declared above
        self.parser = None
        self.parsed_protocol = None

        start = time.time()

        import fc.parsing.CompactSyntaxParser as CSP

        parser = self.parser = CSP.CompactSyntaxParser()
        CSP.Actions.source_file = proto_file
        generator = self.parsed_protocol = parser._Try(
            CSP.CompactSyntaxParser.protocol.parseFile,
            proto_file,
            parseAll=True
        )[0]
        assert isinstance(generator, CSP.Actions.Protocol)

        details = generator.expr()
        assert isinstance(details, dict)

        self.inputs = details.get('inputs', [])
        self.input_env.execute_statements(self.inputs)
        for prefix, path, set_inputs in details.get('imports', []):
            self.log_progress('Importing', path, 'as', prefix, 'in', self.proto_name)
            imported_proto = Protocol(self.get_path(proto_file, path), self.indent_level + 1)

            if prefix:
                self.add_imported_protocol(imported_proto, prefix)
            else:
                # merge inputs of the imported protocol into our own (duplicate names are an error here).
                # Override any values specified in the import statement itself.
                for stmt in imported_proto.inputs:
                    name = stmt.names[0]
                    if name in set_inputs:
                        stmt = Assign([name], set_inputs[name])
                    self.input_env.execute_statements([stmt])
                # Make any prefixed imports of that protocol into our prefixed imports
                for imported_prefix, imported_import in imported_proto.imports.items():
                    self.add_imported_protocol(imported_import, imported_prefix)
                # merge the other elements of its definition with our own
                self.library.extend(imported_proto.library)
                self.simulations.extend(imported_proto.simulations)
                self.post_processing.extend(imported_proto.post_processing)
                self.outputs.extend(imported_proto.outputs)
                self.plots.extend(imported_proto.plots)
                self.model_interface.extend(imported_proto.model_interface)
                for ns_prefix, uri in imported_proto.ns_map.items():
                    existing_uri = self.ns_map.get(ns_prefix, None)
                    if existing_uri is None:
                        self.ns_map[ns_prefix] = uri
                    elif existing_uri != uri:
                        raise ProtocolError(
                            'Prefix ' + str(ns_prefix) + ' is used for'
                            ' multiple URIs in imported protocols.')

        self.library_env.set_delegatee_env(self.input_env)
        self.library.extend(details.get('library', []))
        self.simulations.extend(details.get('simulations', []))
        self.post_processing.extend(details.get('postprocessing', []))
        self.outputs.extend(details.get('outputs', []))
        self.plots.extend(details.get('plots', []))
        self.model_interface.extend(details.get('model_interface', []))
        for prefix, uri in details.get('ns_map', {}).items():
            existing_uri = self.ns_map.get(prefix, None)
            if existing_uri is None:
                self.ns_map[prefix] = uri
            elif existing_uri != uri:
                raise ProtocolError(
                    'Prefix ' + str(prefix) + ' is used for multiple URIs.')

        # Replace ns prefixes with uris in model interface
        for item in self.model_interface:
            if item['type'] == 'OutputVariable':
                item['ns'] = self.ns_map[item['ns']]

        # Benchmark
        self.timings['parsing'] = time.time() - start

    # Override Object serialization methods to allow pickling with the dill module
    def __getstate__(self):
        # TODO: Original object unusable after serialization.
        # Should either maintain object state (i.e., remove reference to simulations
        # in copied dict and re-initialize in __setstate__) or dynamically restore
        # simulation model state at runtime.

        # Must remove Model class and regenerate during unpickling
        # (Pickling errors from nested class structure of ModelWrapperEnvironment)

        # If the protocol has been run, remove references to model environment
        # in the simulations (and un-delegate from library_env)
        for sim in self.simulations:
            if sim.model is not None:
                if "" in sim.env.delegatees:
                    sim.env.clear_delegatee_env("")
                if sim.prefix and sim.prefix in self.library_env.delegatees:
                    self.library_env.clear_delegatee_env(sim.prefix)

        odict = self.__dict__.copy()
        # remove Model and CSP from Protocol
        if 'model' in odict:
            del odict['model']
        if 'parser' in odict:
            del odict['parser']
            del odict['parsed_protocol']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        # Re-import Model from temporary Python file (if provided)
        if hasattr(self, 'model_path'):
            sys.path.insert(0, self.model_path)
            import model as module
            for name in module.__dict__.keys():
                if name.startswith('GeneratedModel'):
                    model = getattr(module, name)()
                    model._module = module
            del sys.modules['model']
            self.model = model
            for sim in self.simulations:
                sim.set_model(model)

    def add_imported_protocol(self, proto, prefix):
        """
        Add a protocol imported with a prefix to our collection.

        This also makes that protocol's library (if any) available as a
        delegatee of our own.
        """
        assert prefix
        if prefix in self.imports:
            raise ProtocolError("The prefix '", prefix, "' has already been used for an imported protocol.")
        self.imports[prefix] = proto
        self.library_env.set_delegatee_env(proto.library_env, prefix)

    def initialise(self, verbose=True):
        """(Re-)initialise this protocol, ready to be run on a model."""
        if verbose:
            self.log_progress('Initialising', self.proto_name)
        self.library_env.clear()
        self.post_processing_env.clear()
        self.output_env.clear()
        for sim in self.simulations:
            sim.clear()
            sim.set_indent_level(self.indent_level + 1)
        for imported_proto in self.imports.values():
            imported_proto.initialise(verbose)

    def set_output_folder(self, path):
        """Specify where the outputs from this protocol will be written to disk."""
        if isinstance(path, OutputFolder):
            self.output_folder = path
        else:
            self.output_folder = OutputFolder(path)

    def outputs_and_plots(self, errors, verbose=True, write_out=True):
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
                try:
                    output = self.post_processing_env.look_up(output_var)
                except KeyError:
                    if output_spec['optional']:
                        self.log_warning("Ignoring missing optional output", output_spec['name'])
                        continue
                    else:
                        raise
                self.output_env.define_name(output_spec['name'], output)
                outputs_defined.append(output_spec)
                if 'description' not in output_spec:
                    output_spec['description'] = output_spec['name']
                if output_spec['name'] in plot_vars:
                    plot_descriptions[output_spec['name']] = output_spec['description']
        if not self.output_folder and verbose and write_out:
            self.log_warning("Warning: protocol output folder not set, so not writing outputs to file")
            return

        if verbose:
            self.log_progress('saving output data to h5 file...')
        start = time.time()
        if write_out:
            with errors:
                filename = os.path.join(self.output_folder.path, 'output.h5')
                h5file = tables.open_file(filename, mode='w', title=os.path.splitext(self.proto_name)[0])
                group = h5file.create_group('/', 'output', 'output parent')
                for output_spec in outputs_defined:
                    h5file.create_array(group,
                                        output_spec['name'],
                                        self.output_env.unwrapped_bindings[output_spec['name']],
                                        title=output_spec['description'])
                h5file.close()
        self.timings['save outputs'] = self.timings.get('output', 0.0) + (time.time() - start)

        # Plots
        if write_out:  # suppress plotting when performing fitting
            start = time.time()
            for plot in self.plots:
                with errors:
                    if verbose:
                        self.log_progress(
                            'plotting',
                            plot['title'],
                            'curve:',
                            plot_descriptions[plot['y']],
                            'against',
                            plot_descriptions[plot['x']]
                        )
                    x_data = []
                    y_data = []
                    x_data.append(self.output_env.look_up(plot['x']).array)
                    y_data.append(self.output_env.look_up(plot['y']).array)
                    if 'key' in plot:
                        key_data = self.output_env.look_up(plot['key']).array
                        if key_data.ndim != 1:
                            raise ProtocolError('Plot key variables must be 1d vectors;',
                                                plot['key'], 'has', key_data.ndim, 'dimensions')
                    # Check the x-axis data shape.  It must either be 1d, or be equivalent to
                    # a 1d vector (i.e. stacked copies of the same vector).
                    for i, x in enumerate(x_data):
                        if x.ndim > 1:
                            num_repeats = reduce(operator.mul, x.shape[:-1])
                            # flatten all extra dimensions as an array view
                            x_2d = x.reshape((num_repeats, x.shape[-1]))
                            if x_2d.ptp(axis=0).any():
                                # There was non-zero difference between the min & max at some position in
                                # the 1d equivalent vector
                                raise ProtocolError(
                                    'The X data for a plot must be (equivalent to) a 1d array, not',
                                    x.ndim, 'dimensions')
                            x_data[i] = x_2d[0]  # Take just the first copy
                    # Plot the data
                    plt.figure()
                    for i, x in enumerate(x_data):
                        y = y_data[i]
                        if y.ndim > 1:
                            # Matplotlib can handle 2d data, but plots columns not rows, so we need to
                            # flatten & transpose
                            y_2d = y.reshape((reduce(operator.mul, y.shape[:-1]), y.shape[-1]))
                            plt.plot(x, y_2d.T)
                        else:
                            plt.plot(x, y)
                        plt.title(plot['title'])
                        plt.xlabel(plot_descriptions[plot['x']])
                        plt.ylabel(plot_descriptions[plot['y']])
                    plt.savefig(os.path.join(self.output_folder.path, self.sanitise_file_name(plot['title']) + '.png'))
                    plt.close()
            self.timings['create plots'] = self.timings.get('plot', 0.0) + (time.time() - start)

    def sanitise_file_name(self, name):
        """Simply transform a name such as a graph title into a valid file name."""
        name = name.strip().replace(' ', '_')
        keep = ('.', '_')
        return ''.join(c for c in name if c.isalnum() or c in keep)

    def execute_library(self):
        """
        run the statements in our library to build up the library environment.

        The libraries of any imported protocols will be executed first.
        """
        for imported_proto in self.imports.values():
            imported_proto.execute_library()
        self.library_env.execute_statements(self.library)

    def run(self, verbose=True, write_out=True):
        """run this protocol on the model already specified using set_model."""
        Locatable.output_folder = self.output_folder
        self.initialise(verbose)
        if verbose:
            self.log_progress('running protocol', self.proto_name, '...')
        errors = ErrorRecorder(self.proto_name)
        with errors:
            for sim in self.simulations:
                sim.env.set_delegatee_env(self.library_env)
                if sim.prefix:
                    if self.output_folder:
                        sim.set_output_folder(self.output_folder.create_subfolder(
                            'simulation_' + sim.prefix))
                    self.library_env.set_delegatee_env(sim.results, sim.prefix)
            start = time.time()
            self.execute_library()
            self.timings['run library'] = self.timings.get('library', 0.0) + (time.time() - start)
        with errors:
            start = time.time()
            self.run_simulations(verbose)
            self.timings['simulations'] = self.timings.get('simulations', 0.0) + (time.time() - start)
        with errors:
            start = time.time()
            self.run_post_processing(verbose)
            self.timings['post-processing'] = self.timings.get('post-processing', 0.0) + (time.time() - start)
        self.outputs_and_plots(errors, verbose, write_out)
        # Summarise time spent in each protocol section (if we're the main protocol)
        if verbose and self.indent_level == 0:
            print('Time spent running protocol (s): %.6f' % sum(self.timings.values()))
            max_len = max(len(section) for section in self.timings)
            for section, duration in self.timings.items():
                print('   ', section, ' ' * (max_len - len(section)), '%.6f' % duration)
        if errors:
            # Report any errors that occurred
            raise errors

    def run_simulations(self, verbose=True):
        """run the model simulations specified in this protocol."""
        for sim in self.simulations:
            if verbose:
                self.log_progress('running simulation', sim.prefix)
            sim.initialise()
            sim.run(verbose)
            # Reset trace folder in case a nested protocol changes it
            Locatable.output_folder = self.output_folder

    def run_post_processing(self, verbose=True):
        """run the post-processing section of this protocol."""
        if verbose:
            self.log_progress('running post processing for', self.proto_name, '...')
        self.post_processing_env.execute_statements(self.post_processing)

    def set_input(self, name, value_expr):
        """Overwrite the value of a protocol input.

        The value may be given either as an actual value, or as an expression which will be evaluated in
        the context of the existing inputs.
        """
        if isinstance(value_expr, V.AbstractValue):
            value = value_expr
        else:
            value = value_expr.evaluate(self.input_env)
        self.input_env.overwrite_definition(name, value)

    def set_model(self, model, exposeNamedParameters=False):
        """
        Specify the model this protocol is to be run on.

        The ``model`` can be given as a Model object or a string.
        """

        # Benchmarking
        start = time.time()

        # compile model from CellML
        if isinstance(model, str) and model.endswith('.cellml'):

            if exposeNamedParameters:
                raise ValueError('I have no idea what this does.')

            self.log_progress('Generating model code...')

            # Create output folder
            if self.output_folder:
                temp_dir = tempfile.mkdtemp(dir=self.output_folder.path)
            else:
                temp_dir = tempfile.mkdtemp()

            # Select output path (in temporary dir)
            path = os.path.join(temp_dir, 'model.pyx')

            # Select class name
            # Note: Code further down relies on this starting with
            # 'GeneratedModel'
            class_name = 'GeneratedModel'

            # load cellml model
            import cellmlmanip
            model = cellmlmanip.load_model(model)

            # Select model outputs (as qualified names)
            outputs = []
            for item in self.model_interface:
                # TODO: Update this to use objects instead of a list of dicts
                if item['type'] == 'OutputVariable':
                    # TODO: Better handling for `state_variable`
                    if item['local_name'] == 'state_variable':
                        outputs.append('state_variable')
                    else:
                        outputs.append((item['ns'], item['local_name']))
                    # TODO: Handle units

            # Select model parameters (as qualified names)
            # TODO DO WHATEVER WE NEED TO DO HERE
            parameters = [
            ]

            # Create weblab model at path
            import weblab_cg as cg
            cg.create_weblab_model(
                path, class_name, model, outputs, parameters)

            self.log_progress('Compiling pyx model code...')

            # Get path to root dir of fc module
            fcpath = os.path.abspath(os.path.join(fc.MODULE_DIR, '..'))

            # Write setup.py
            setup_file = os.path.join(temp_dir, 'setup.py')
            template_strings = {
                'module_name': 'model',
                'model_file': 'model.pyx',
                'fcpath': fcpath,
            }
            with open(setup_file, 'w') as f:
                f.write(SETUP_PY % template_strings)

            # compile the extension module
            print(subprocess.check_output(
                ['python', 'setup.py', 'build_ext', '--inplace'],
                cwd=temp_dir,
                stderr=subprocess.STDOUT,
            ))

            # Create an instance of the model
            self.model_path = temp_dir
            sys.path.insert(0, temp_dir)
            import model as module
            for name in module.__dict__.keys():
                if name.startswith(class_name):
                    model = getattr(module, name)()
                    model._module = module
                    break
            del sys.modules['model']

        # Set model
        self.model = model
        for sim in self.simulations:
            sim.set_model(model)

        # Benchmarking
        self.timings['load model'] = (
            self.timings.get('load model', 0.0) + (time.time() - start))

    def get_path(self, base_path, path):
        """Determine the full path of an imported protocol file.

        Relative paths are resolved relative to base_path (the path to this
        protocol) by default.
        If this does not yield an existing file, they are resolved relative to
        the built-in library folder instead.
        """
        new_path = os.path.join(os.path.dirname(base_path), path)
        if not os.path.isabs(path) and not os.path.exists(new_path):
            # Search in the library folder instead
            library = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'library')
            new_path = os.path.join(library, path)
        return new_path

    def set_indent_level(self, indent_level):
        """Set the level of indentation to use for progress output."""
        self.indent_level = indent_level

    def log_progress(self, *args):
        """Print a progress line showing how far through the protocol we are.

        Arguments are converted to strings and space separated, as for the
        print builtin.
        """
        print('  ' * self.indent_level + ' '.join(map(str, args)))
        sys.stdout.flush()

    def log_warning(self, *args):
        """Print a warning message.

        Arguments are converted to strings and space separated, as for the
        print builtin.
        """
        print('  ' * self.indent_level + ' '.join(map(str, args)), file=sys.stderr)
        sys.stderr.flush()
