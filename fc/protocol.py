"""
Contains the main classes representing an FC Protocol.
"""
import itertools
import os
import subprocess
import sys
import tables
import tempfile
import time

from cellmlmanip.units import UnitStore

import fc
from .code_generation import create_weblab_model
from .environment import Environment
from .error_handling import ProtocolError, ErrorRecorder
from .file_handling import OutputFolder, sanitise_file_name
from .language import values as V
from .language.statements import Assign
from .locatable import Locatable
from .parsing import actions
from .parsing.rdf import get_used_annotations
from .plotting import create_plot

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
        # This is currently not stored in this object.

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
        # Maps an import 'name' prefix to a :class:`Protocol` instance.
        self.imports = {}

        # 5. The ``library`` section
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Library
        # Can contain assignment statements (``var = expr``), function
        # assignment statements (``var = lambda(...)``), or assertions
        # (``assert cond``).
        # These statements will be executed when the protocol is run.
        self.library = []

        # 6. The ``units`` section
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Physicalunitdefinitions
        # We store the definitions as well as the resolved units to allow for merging definitions from
        # imported protocols or nested protocols without the need to reconcile unit registries and exact
        # unit names.
        self.unit_definitions = {}
        self.units = UnitStore()

        # 7. The ``model interface`` section.
        # See :class:`fc.parsing.actions.ModelInterface`.
        self.model_interface = actions.ModelInterface()

        # 8. The ``tasks`` section, which contains any number of simulation
        # tasks (possibly with nested ones).
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Simulationtasks
        # Contains instances of :class:`fc.simulations.simulations.AbstractSimulation` subclasses.
        self.simulations = []

        # 9. The ``post-processing`` section, that contains post-processing
        # code
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Post-processing
        # Contains statement instances, as with the library.
        self.post_processing = []

        # 10. The ``outputs`` section, listing outputs from the simulations or
        # from post-processing, that can be used in the ``plots`` section or by
        # other protocols.
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Protocoloutputs

        # A list of dictionaries, where each dict specifies a protocol output. They can have keys:
        # - name: name to give the output; should be a valid simple identifier.
        # - ref (optional): the (prefixed or simple) variable name giving the output's value.
        #   Will be looked up in the post-processing results if no prefix. Defaults to name.
        # - description (optional): human-readable description of the output; defaults to name.
        #   Used for plot labels.
        # - optional: whether it is an error if the variable referenced doesn't exist.
        # - units: name of the units this output is measured in, from self.units
        self.outputs = []

        # 11. The ``plots`` section
        # https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Graphicalplots
        # A list of dictionaries with keys:
        # - title: title for the plot.
        # - x: name of the x-variable; should be a protocol output.
        # - y: name of the y-variable; should be a protocol output.
        # - key (optional): name of the key variable; should be a protocol output.
        self.plots = []

        # Parse, and fill section information objects defined above
        self._parse()

    def _parse(self):
        """
        Parses the protocol, and fills in this object's fields.
        """
        start = time.time()

        import fc.parsing.CompactSyntaxParser as CSP
        parser = CSP.CompactSyntaxParser()
        generator = parser.parse_file(self.proto_file)
        # A :class:`fc.parsing.actions.Protocol` object,
        # containing parsed information about the protocol
        assert isinstance(generator, actions.Protocol)

        with actions.set_reference_source(self.proto_file):
            details = generator.expr()
        assert isinstance(details, dict)
        del(generator)

        # Store protocol inputs
        self.inputs = details.get('inputs', [])
        self.input_env.execute_statements(self.inputs)

        # Store unit definitions and add these to protocol
        for udef in details.get('units', []):
            self.unit_definitions[udef.name] = udef
            self.units.add_unit(udef.name, udef.pint_expression)

        def merge_unit_definitions(unit_definitions):
            """Merge ``unit_definitions`` from a nested/imported protocol with ours.

            Duplicate names are OK if the definitions are the same.
            """
            for name, udef in unit_definitions.items():
                if name in self.unit_definitions:
                    # Just check the definitions match, and complain if not
                    our_udef = self.unit_definitions[name]
                    if our_udef.pint_expression != udef.pint_expression:
                        raise ProtocolError(
                            f'Imported or nested protocol redefines units {name} as '
                            f'{udef.pint_expression} not {our_udef.pint_expression}')
                    if our_udef.description != udef.description:
                        raise ProtocolError(
                            f'Imported or nested protocol redefines units {name} with '
                            f'description "{udef.description}" not "{our_udef.description}"')
                else:
                    # Include this new definition
                    self.unit_definitions[name] = udef
                    self.units.add_unit(name, udef.pint_expression)

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

                # Merge model interface
                self.model_interface.merge(imported_proto.model_interface)

                # Add units
                merge_unit_definitions(imported_proto.unit_definitions)

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
        interface = details.get('model_interface', None)
        if interface is not None:
            self.model_interface.merge(interface)

        # Process information from nested simulations
        for simulation in details.get('simulations', []):
            try:
                nested_proto = simulation.nested_sim.model.proto
                nested_interface = nested_proto.model_interface
            except AttributeError:
                continue

            # Merge interface
            self.model_interface.merge(nested_interface)

            # Add units
            merge_unit_definitions(nested_proto.unit_definitions)

        # Store namespace map
        process_ns_map(details.get('ns_map', {}))

        # Replace ns prefixes with uris in model interface
        self.model_interface.resolve_namespaces(self.ns_map)

        # Store benchmarking information
        self.timings['parsing'] = time.time() - start

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
        # Copy protocol outputs into the self.output_env environment,
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
                    path = os.path.join(self.output_folder.path, sanitise_file_name(plot['title']) + '.png')
                    x_label = plot_descriptions[plot['x']]
                    y_label = plot_descriptions[plot['y']]
                    x_data = [self.output_env.look_up(plot['x']).array]
                    y_data = [self.output_env.look_up(plot['y']).array]

                    if verbose:
                        self.log_progress(
                            'plotting', plot['title'],
                            'curve:', plot_descriptions[plot['y']],
                            'against', plot_descriptions[plot['x']]
                        )

                    if 'key' in plot:
                        key_data = self.output_env.look_up(plot['key']).array
                        if key_data.ndim != 1:
                            raise ProtocolError('Plot key variables must be 1d vectors;',
                                                plot['key'], 'has', key_data.ndim, 'dimensions')

                    create_plot(path, x_data, y_data, x_label, y_label, plot['title'])

            self.timings['create plots'] = self.timings.get('plot', 0.0) + (time.time() - start)

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

        The value may be given either as an actual value (any object extending ``AbstractValue``), or as an expression
        (extending ``AbstractExpression``) which will be evaluated in the context of the existing inputs.
        """
        if isinstance(value_expr, V.AbstractValue):
            value = value_expr
        else:
            value = value_expr.evaluate(self.input_env)
        self.input_env.overwrite_definition(name, value)

    def set_model(self, model):
        """
        Specify the model this protocol is to be run on.

        The ``model`` can be given as a Model object or a string.
        """

        # Benchmark model creation time
        start = time.time()

        # Compile model from CellML
        if isinstance(model, str) and model.endswith('.cellml'):
            self.log_progress('Generating model code...')

            # Create output folder
            if self.output_folder:
                temp_dir = tempfile.mkdtemp(dir=self.output_folder.path)
            else:
                temp_dir = tempfile.mkdtemp()

            # Select output path (in temporary dir)
            path = os.path.join(temp_dir, 'model.pyx')

            # Select class name
            class_name = 'GeneratedModel'

            # Load cellml model
            import cellmlmanip
            model = cellmlmanip.load_model(model, self.units)

            # Check whether the model has a time variable. If not, create one
            try:
                time_variable = model.get_free_variable()
            except ValueError:
                time_variable = model.create_unique_name('time')
                time_variable = model.add_variable(time, self.units.get('seconds'))

            # Do all the transformations specified by the protocol
            time_variable = self.model_interface.modify_model(model, time_variable, self.units)

            # Create weblab model at path
            create_weblab_model(
                path,
                self.output_folder.path if self.output_folder else temp_dir,
                class_name,
                model,
                time_variable,
                ns_map=self.ns_map,
                protocol_variables=self.model_interface.protocol_variables,
            )

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
            result = subprocess.run(
                ['python', 'setup.py', 'build_ext', '--inplace'],
                cwd=temp_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            print(result.stdout.decode())
            if result.returncode != 0:
                raise ProtocolError('Failed to generate executable model code; see output above for details')

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

    def get_protocol_interface(self):
        """Get the names and units of the inputs to and outputs from this protocol.

        This is used by the Web Lab front-end as part of its protocol analysis phase.

        Returns a list of dictionaries, each with keys 'name', 'units' and 'kind':
        * ``kind`` is either 'input' or 'output'
        * ``name`` is the name of the input or output, and is guaranteed unique for a given kind
        * ``units`` is a pint definition string for the units, as determined by ``self.units``
        """
        interface = []
        for output in self.outputs:
            units = output.get('units')
            if units is None:
                # Units are read from the model, and so may be determined by our model interface
                units = ''  # Default in case we can't figure it out
                ref = output.get('ref', '')
                if ':' in ref:
                    local_name = ref.split(':')[-1]
                    for model_output in self.model_interface.outputs:
                        if model_output.local_name == local_name:
                            units = model_output.units
                            break
            if not self.units.is_defined(units):
                # We don't currently require units taken from the store
                units = ''
            if units != '':
                units = self.units.format(self.units.get_unit(units), base_units=True)
            interface.append({
                'kind': 'output',
                'name': output['name'],
                'units': units,
            })
        return interface

    def get_required_model_annotations(self):
        """Get the set of ontology terms referenced by this protocol, recursively processing imports.

        This looks at the combined model interface, and analyses as follows:
        * Inputs, outputs and clamps all define a single variable that is part of the interface
        * The tokens tree for equations is walked recursively and any :class:`actions.Variable` instances that reference
          ontology terms are included
        * Variables defined directly by an equation are optional; similarly for inputs that give units and initial value
        * Optional declarations are processed to determine which of the terms found are explicitly optional
        * Variable references in units conversion rules are considered optional unless they appear elsewhere as required

        :return: a pair (required terms, optional terms) of sets of ontology terms, as full URI strings
        """
        iface = self.model_interface
        maybe_required = set()
        maybe_optional = set()
        optional = set()

        def process_var_ref(name, result_set):
            """Add the given variable to the result set if it is an ontology term reference.

            If the ``name`` has the form ``prefix:local_name`` then look up the prefix in our namespace map to construct
            the full URI.
            """
            if ':' in name:
                prefix, local_name = name.split(':', 1)
                ns_uri = iface.ns_map.get(prefix)
                if ns_uri:
                    result_set.add(ns_uri + local_name)

        def walk_equation(token, result_set):
            """Recursively walk an equation token tree looking for variable references, and add them to the set."""
            if isinstance(token, actions.Variable):
                process_var_ref(token.name(), result_set)
            elif hasattr(token, 'tokens'):
                for t in token.tokens:
                    walk_equation(t, result_set)
            elif isinstance(token, actions.pyparsing.ParseResults):
                for t in token:
                    walk_equation(t, result_set)

        # Unit conversion rules: maybe optional
        for rule in iface.unit_conversion_rules:
            walk_equation(rule.tokens[-1], maybe_optional)

        # Simple references
        for var in itertools.chain(iface.outputs, iface.clamps):
            maybe_required.add(var.ns_uri + var.local_name)
        for var in iface.inputs:
            if var.units is not None and var.initial_value is not None:
                optional.add(var.ns_uri + var.local_name)
            else:
                maybe_required.add(var.ns_uri + var.local_name)

        # Equation references
        for eq in iface.equations:
            walk_equation(eq.rhs, maybe_required)
            if eq.is_ode:
                process_var_ref(eq.var.name(), maybe_required)
                process_var_ref(eq.bvar.name(), maybe_required)
            else:
                process_var_ref(eq.var.name(), optional)

        # Optional variables
        for var in iface.optional_decls:
            optional.add(var.ns_uri + var.local_name)
            if var.default_expr is not None:
                walk_equation(var.default_expr, optional)

        # Figure out which way the 'maybe' references go
        optional = optional | (maybe_optional - maybe_required)
        return maybe_required - optional, optional

    def check_model_compatibility(self, model_path):
        """Determine whether the given model is compatible with this protocol.

        This checks whether the ontology terms accessed by the protocol are present in the model.
        It returns lists of terms required but not present, and the pair are compatible if the first
        of these is empty. The second list gives optional terms that aren't in the model, so the
        protocol should still run but not all outputs may be available.

        :param model_path: the CellML file to check
        :return: a pair of sorted lists (missing required terms, missing optional terms) as full URI strings
        """
        required_terms, optional_terms = self.get_required_model_annotations()
        import cellmlmanip
        model = cellmlmanip.load_model(model_path)
        model_terms = get_used_annotations(model)
        # Return the mismatch, if any, as sorted lists
        missing_terms = list(required_terms - model_terms)
        missing_terms.sort()
        missing_optional_terms = list(optional_terms - model_terms)
        missing_optional_terms.sort()
        return missing_terms, missing_optional_terms

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
