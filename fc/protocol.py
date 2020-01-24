#
# Contains the main classes representing an FC Protocol.
#
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
from .environment import Environment  # noqa: E402
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
        # if the filename passed as argument to CompactSyntaxParser is not found
        # it tries to read it before checking it exists
        # and you get a messy exit
        # file_contents = file_or_filename.read()
        #     E AttributeError: 'str' object has no attribute 'read'
        #
        #     During handling of the above exception, another exception occurred:
        #     E FileNotFoundError: [Errno 2] No such file or directory:
        # throw exception

        with open(proto_file) as f:
            f.close()

        self.proto_file = proto_file
        self.proto_name = os.path.basename(self.proto_file)

        # Indent level used for progress reporting
        self.indent_level = indent_level
        self.log_progress('Constructing', self.proto_name)

        # A dict with benchmarking information
        self.timings = {}

        # Path to store protocol output at
        self.output_folder = None

        # Main environment used when running the protocol
        self.env = Environment()

        # Environment containing the protocol inputs
        self.input_env = Environment(allow_overwrite=True)
        self.input_env.define_name("load", V.LoadFunction(os.path.dirname(self.proto_file)))

        # Environment for variables defined in the protocol library
        self.library_env = Environment()

        # Environment containing variables defined in post-processing
        self.post_processing_env = Environment(delegatee=self.library_env)

        # Environment containing the protocol outputs
        self.output_env = Environment()

        #
        # The elements making up this protocol's definition
        #

        # 1. The ``documentation`` section.
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Documentation

        # TODO Is this parsed / stored?

        # 2. Namespace bindings (no section, just a list of statements)
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Namespacebindings

        # Maps namespaces (prefixes) to URIs.
        self.ns_map = {}

        # 3. Parsed results from the ``inputs`` section.
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Protocolinputdeclarations
        # This contains inputs _to the protocol_, that can be used when this
        # protocol is used by another protocol.

        # A list of :class:`fc.language.statements.Assign` objects.
        # Each representing an assignment ``local var = expression``.
        self.inputs = []

        # 4. Any number of ``import`` statements (again, no section)
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Importsofotherprotocols

        # TODO: Not sure what this dict contains
        self.imports = {}

        # 5. The ``library`` section
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Library
        # Can contain assignment statements (``var = expr``), function
        # assignment statements (``var = lambda(...)``), or assertions
        # (``assert cond``).

        # TODO: Do all of these end up in this list?
        # TODO: Who checks the assertions?
        self.library = []

        # 6. The ``units`` section
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Physicalunitdefinitions
        # TODO: Where do these end up?

        # 7. The ``model interface`` section.
        # See :class:`ModelInterface`.

        # TODO: Only seems to contain model outputs at the moment
        self.model_interface = ModelInterface()

        # 8. The ``tasks`` section, which contains any number of simulation
        # tasks (possibly with nested ones).
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Simulationtasks

        # TODO: Contains?
        self.simulations = []

        # 9. The ``post-processing`` section, that contains post-processing
        # code
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Post-processing

        # TODO: Contains?
        self.post_processing = []

        # 10. The ``outputs`` section, listing outputs from the simulations or
        # from post-processing, that can be used in the ``plots`` section or by
        # other protocols.
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Protocoloutputs

        # A list of dictionaries, where each dict specifies a protocol output
        # TODO: dict format
        self.outputs = []

        # 11. The ``plots`` section
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Graphicalplots

        # TODO: Contains?
        self.plots = []

        # Parse, and fill section information objects defined above
        self._parse()

    def _parse(self):
        """
        Parses the protocol, and fills in this object's fields.
        """

        # Parse the protocol file and fill in the structures declared above
        self.parser = None
        self.parsed_protocol = None

        start = time.time()

        import fc.parsing.CompactSyntaxParser as CSP
        parser = self.parser = CSP.CompactSyntaxParser()
        CSP.Actions.source_file = self.proto_file
        generator = self.parsed_protocol = parser.try_parse(
            CSP.CompactSyntaxParser.protocol.parseFile,
            self.proto_file,
            parseAll=True
        )[0]
        assert isinstance(generator, CSP.Actions.Protocol)

        # A class:`fc.parsing.CompactSyntaxParser.Actions.Protocol` object,
        # containing parsed information about the protocol
        details = generator.expr()
        assert isinstance(details, dict)
        del(generator)

        # Store protocol inputs
        self.inputs = details.get('inputs', [])
        self.input_env.execute_statements(self.inputs)

        # Create model interface
        def process_interface(interface):
            """ Process a protocol's model interface. """

            for item in interface:
                if item['type'] == 'OutputVariable':
                    self.model_interface.outputs.append(
                        VariableReference(
                            item['local_name'],
                            item['ns'],
                            item['units'],
                        ))

                elif item['type'] == 'InputVariable':
                    var = VariableReference(
                        item['local_name'],
                        item['ns'],
                        item['units'],
                    )
                    self.model_interface.inputs.append(var)
                    if item['initial_value'] is not None:
                        self.model_interface.input_values[var] = item['initial_value']

                elif item['type'] == 'ModelEquation':
                    self.model_interface.defines.append(
                        DefineStatement(
                            VariableReference.from_string(item['var']),
                            item['rhs'],
                            item['ode'],
                            VariableReference.from_string(item['bvar']),
                        ))

        # Update namespace map
        def process_ns_map(ns_map):
            """ Merge the items from ``ns_map`` with this protocol's prefix to namespace mapping. """

            for ns_prefix, uri in ns_map.items():
                existing_uri = self.ns_map.get(ns_prefix, None)
                if existing_uri is None:
                    self.ns_map[ns_prefix] = uri
                elif existing_uri != uri:
                    raise ProtocolError('Prefix ' + str(ns_prefix) + ' is used for multiple URIs.')

        # Store information from imported protocols
        for prefix, path, set_inputs in details.get('imports', []):
            self.log_progress('Importing', path, 'as', prefix, 'in', self.proto_name)
            imported_proto = Protocol(self.get_path(self.proto_file, path), self.indent_level + 1)
            if prefix:
                self.add_imported_protocol(imported_proto, prefix)
            else:
                # Merge inputs of the imported protocol into our own (duplicate names are an error here).
                # Override any values specified in the import statement itself.
                for stmt in imported_proto.inputs:
                    name = stmt.names[0]
                    if name in set_inputs:
                        stmt = Assign([name], set_inputs[name])
                    self.input_env.execute_statements([stmt])

                # Merge any prefixed imports of that protocol into our prefixed imports
                for imported_prefix, imported_import in imported_proto.imports.items():
                    self.add_imported_protocol(imported_import, imported_prefix)

                # Merge various elements with our own
                self.library.extend(imported_proto.library)
                self.simulations.extend(imported_proto.simulations)
                self.post_processing.extend(imported_proto.post_processing)
                self.outputs.extend(imported_proto.outputs)
                self.plots.extend(imported_proto.plots)

                # Process interface
                process_interface(imported_proto.model_interface)

                # Process namespace mapping
                process_ns_map(imported_proto.ns_map)

        # Store information from this protocol (potentially overriding info from imported protocols)
        self.library_env.set_delegatee_env(self.input_env)
        self.library.extend(details.get('library', []))
        self.simulations.extend(details.get('simulations', []))
        self.post_processing.extend(details.get('postprocessing', []))
        self.outputs.extend(details.get('outputs', []))
        self.plots.extend(details.get('plots', []))

        # Store information from the model interface section
        process_interface(details.get('model_interface', []))

        # Store namespace map
        process_ns_map(details.get('ns_map', {}))

        # Replace ns prefixes with uris in model interface
        self.model_interface.resolve_namespaces(self.ns_map)

        # Store benchmarking information
        self.timings['parsing'] = time.time() - start

    def __getstate__(self):
        """
        Override Object serialization methods to allow pickling with the dill module
        """
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
        # Remove Model and CSP from Protocol
        if 'model' in odict:
            del odict['model']
        if 'parser' in odict:
            del odict['parser']
            del odict['parsed_protocol']
        return odict

    def __setstate__(self, dict):
        """ Set fields of a protocol after unpickling. """
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
        """(Re-)Initialise this protocol, ready to be run on a model."""
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

    def get_output_folder(self):
        """ Return the output folder used to save data """
        return self.output_folder.path

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
                            'plotting', plot['title'],
                            'curve:', plot_descriptions[plot['y']],
                            'against', plot_descriptions[plot['x']]
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
                            # Flatten all extra dimensions as an array view
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
                            # Flatten & transpose
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
        Run the statements in our library to build up the library environment.

        The libraries of any imported protocols will be executed first.
        """
        for imported_proto in self.imports.values():
            imported_proto.execute_library()
        self.library_env.execute_statements(self.library)

    def run(self, verbose=True, write_out=True):
        """
        Run this protocol on the model already specified using set_model.
        """

        # Initialise
        Locatable.output_folder = self.output_folder
        self.initialise(verbose)
        if verbose:
            self.log_progress('running protocol', self.proto_name, '...')

        # Run the statements in our library to build up the library environment
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

        # Run the simulation block
        with errors:
            start = time.time()
            self.run_simulations(verbose)
            self.timings['simulations'] = self.timings.get('simulations', 0.0) + (time.time() - start)

        # Run the post-processing block
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

        # Report any errors that occurred
        if errors:
            raise errors

    def run_simulations(self, verbose=True):
        """Run the model simulations specified in this protocol."""
        for sim in self.simulations:
            if verbose:
                self.log_progress('running simulation', sim.prefix)
            sim.initialise()
            sim.run(verbose)
            # Reset trace folder in case a nested protocol changes it
            Locatable.output_folder = self.output_folder

    def run_post_processing(self, verbose=True):
        """Run the post-processing section of this protocol."""
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

        # Benchmark model creation time
        start = time.time()

        # Compile model from CellML
        if isinstance(model, str) and model.endswith('.cellml'):

            if exposeNamedParameters:
                raise ValueError('Michael has no idea what this does.')

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

            # Load cellml model
            import cellmlmanip
            model = cellmlmanip.load_model(model)

            # Create protocol unit store
            # TODO

            # Add input variables in correct units
            for var in self.model_interface.inputs:
                # TODO: Check if inputs are of allowed types, i.e. states or parameters
                # TODO Add input variables to cellmlmanip model
                pass

            # Add output variables in correct units
            for var in self.model_interface.outputs:
                # TODO: Ask cellmlmanip to check if var has one of the allowed types: states, parameters, or derived
                #       quantities.
                # TODO: Handle units
                pass

            # Create a list of model outputs (as ontology terms) for code generation
            outputs = []
            for var in self.model_interface.outputs:
                if var.name == 'state_variable':
                    outputs.append('state_variable')
                else:
                    outputs.append((var.ns, var.name))

            # Create a list of parameters (as ontology terms). Parameters are inputs that are constants.
            parameters = []
            for var in self.model_interface.inputs:
                # TODO: Check if a constant, if so add to parameter list
                pass

            # Handle define statements
            for define in self.model_interface.defines:
                # TODO Convert parse tree to sympy expression with correct variables
                pass

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

            # Compile the extension module
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
                if name.startswith(class_name):     # Note: This bit!
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


class ModelInterface(object):
    """
    Holds information about a model interface.

    Properties:

    ``inputs``
        A list of :class:`VariableReference` objects indicating model inputs
        (model variables that can be changed by the protocol).
    ``input_values``
        A mapping from variable references to the initial values they take.
    ``outputs``
        A list of :class:`VariableReference` objects indicating model outputs
        (model variables that can be read by the protocol).
    ``defines``
       A list of define statements

    See: https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Modelinterface
    """
    def __init__(self):

        # self.independent_var = None
        self.inputs = []
        self.input_values = {}
        self.outputs = []
        self.defines = []

        '''
        ``clamps``
            A list of model variables to be clamped
        ``vars``
        ``converts``
        ``independent_var``
            Specifies the units that the indepdendent variable (time) should be in.

        self.clamps = []
        self.defines = []
        self.vars = []
        self.converts = []
        '''

    def resolve_namespaces(self, ns_map):
        """ Resolve the variable references in this interface, changing all prefixes to the full namespaces. """

        for ref in self.inputs:
            ref.ns = ns_map[ref.ns]

        for ref in self.outputs:
            ref.ns = ns_map[ref.ns]

        for item in self.defines:
            item.var.ns = ns_map[item.var.ns]
            if item.ode:
                item.bvar.ns = ns_map[item.bvar.ns]
            # TODO Update references in equation

    def resolve_units(self):
        """ Resolve unit names in this interface. """

        raise NotImplementedError


class VariableReference(object):
    """
    Reference to a variable with a ``name`` and an optional namespace ``ns``.
    """
    def __init__(self, name, ns=None, units=None):
        self.ns = ns
        self.name = name
        self.units = units
        # TODO: Initial value? Not sure if this goes here

    def __str__(self):
        return self.name if self.ns is None else self.ns + ':' + self.name

    @staticmethod
    def from_string(name):
        ns, name = name.split(':', 2)
        return VariableReference(name, ns)


class DefineStatement(object):
    """
    A statement that defines or replaces a model variable.

    Arguments:

    ``var``
        A :class:`VariableReference` indicating the variable this statement defines or modifies.
    ``rhs``
        The new RHS ????????????????????????????? AS A WHAT ??????????????????????????????
    ``ode``
        True if the variable should be a state.
    ``bvar``
        ``None`` if this variable isn't a state, but a :class:`VariableReference` if it is.

    """
    def __init__(self, var, rhs, ode, bvar=None):
        self.var = var
        self.rhs = rhs
        self.ode = bool(ode)
        self.bvar = bvar

    def __str__(self):
        lhs = ('diff(' + str(self.var) + ', ' + str(self.bvar) + ')') if self.ode else str(self.var)
        rhs = '???'
        return 'Define[' + lhs + ' = ' + rhs + ']'

