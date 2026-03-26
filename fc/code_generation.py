"""
Methods for code generation (using jinja2 templates).
"""
import os
import posixpath
import time

import jinja2
import sympy
from cellmlmanip.model import Quantity
from cellmlmanip.parser import SYMPY_SYMBOL_DELIMITER, Transpiler
from cellmlmanip.printer import Printer

# Tell cellmlmanip to create _exp objects instead of exp objects, etc. This prevents Sympy doing simplification (or
# canonicalisation) resulting in weird errors with exps in some cardiac models.
Transpiler.set_mathml_handler('exp', sympy.Function('_exp'))
Transpiler.set_mathml_handler('abs', sympy.Function('_abs'))
Transpiler.set_mathml_handler('sqrt', sympy.Function('_sqrt'))
Transpiler.set_mathml_handler('sin', sympy.Function('_sin'))
Transpiler.set_mathml_handler('cos', sympy.Function('_cos'))
Transpiler.set_mathml_handler('acos', sympy.Function('_acos'))


# Shared Jinja environment
_environment = None


def _jinja_environment():
    """
    Returns a shared Jinja environment to create templates from.
    """

    global _environment
    if _environment is None:
        _environment = jinja2.Environment(
            # Automatic loading of templates stored in the module
            # This also enables template inheritance
            loader=jinja2.PackageLoader('fc', 'templates'),

            # Keep a single trailing newline, if present
            keep_trailing_newline=True,

            # Don't replace undefined template variables by an empty string
            # but raise a jinja2.UndefinedError instead.
            undefined=jinja2.StrictUndefined,
        )
    return _environment


def load_template(*name):
    """
    Loads a template from the local template directory.

    Templates can be specified as a single filename, e.g.
    ``load_template('temp.txt')``, or loaded from subdirectories using e.g.
    ``load_template('subdir_1', 'subdir_2', 'file.txt')``.

    """
    # Due to a Jinja2 convention, posixpaths must be used, regardless of the
    # user's operating system!
    path = posixpath.join(*name)

    env = _jinja_environment()
    return env.get_template(path)


class DataInterpolation(Quantity):
    """Represents linear interpolation on data loaded from file, for code generation."""
    _next_id = 0  # Used to generate unique table IDs in code

    # Sympy annoyingly overwrites __new__
    def __new__(cls, name, data, index_variable, units, index_name=None, id=None, *args, **kwargs):
        obj = super().__new__(cls, name, real=True)

        # Record a unique ID for this table
        if id is None:
            obj._id = cls._next_id
            cls._next_id += 1
        else:
            obj._id = id

        # Ensure we depend on the index variable in the model's graph
        obj._args = [index_variable]

        return obj

    def __init__(self, name, data, index_variable, units, index_name=None, id=None, *args, **kwargs):
        """Create a new interpolation construct.

        :param name: an identifier for the table, e.g. the data file base name. Will be used for documenting the
            generated code.
        :param data: 2d numpy array containing the column-wise data. The first column is the index data (values to look
            up) and the second column contains the corresponding result values.
        :param index_variable: the :class:`cellmlmanip.model.Variable` used to index the data
        :param units: the units of the interpolated values (a :class:`~cellmlmanip.units.UnitStore.Unit`)
        """
        super().__init__(name, units)

        self._input_data = data
        self.index_variable = index_variable
        self.initial_index = '%.17g' % data[0, 0]
        self.final_index = '%.17g' % data[0, -1]

        self.table_name = '_data_table_' + str(self._id)
        self.step_inverse = '%.17g' % (1.0 / (data[0, 1] - data[0, 0]))

        self.index_name = index_name

    def _eval_evalf(self, prec):
        """We don't want the model's graph to try replacing this with a number!"""
        return self

    def set_index_name(self, variable_name_generator):
        """Set the name used for our index variable in code.

        :param variable_name_generator: function turning our index variable into its name in the code
        """
        self.index_name = variable_name_generator(self.index_variable)

    @property
    def data(self):
        return self._input_data[1]

    @property
    def func(self):
        return lambda index_variable: self.__class__(self.name, self._input_data, index_variable, self.units,
                                                     index_name=self.index_name, id=self._id)

    @property
    def lookup_call(self):
        """The code for a function call looking up this interpolation table."""
        assert self.index_name is not None
        return 'lookup{}({})'.format(self.table_name, self.index_name)

    @property
    def data_code(self):
        """The code creating the data array to interpolate over."""
        return '[{}]'.format(', '.join(map(lambda f: '%.17g' % f, self.data)))


class WebLabPrinter(Printer):
    """
    Cellmlmanip ``Printer`` subclass for the Web Lab.
    """

    def __init__(self, symbol_function=None, derivative_function=None):
        super().__init__(symbol_function, derivative_function)

        # Deal with functions introduced to avoid simplification
        self._function_names['_exp'] = 'math.exp'
        self._function_names['_abs'] = 'math.fabs'
        self._function_names['_sqrt'] = 'math.sqrt'
        self._function_names['_sin'] = 'math.sin'
        self._function_names['_cos'] = 'math.cos'
        self._function_names['_acos'] = 'math.acos'


def get_unique_names(model):
    """
    Creates unique names for all variables in a CellML model.
    """
    # Component variable separator
    # Note that variables are free to use __ in their names too, it makes the
    # produced code less readable but doesn't break anything.
    sep = '__'

    # Create a variable => name mapping, and a reverse name => variable mapping
    variables = {}
    reverse = {}

    def uname(name):
        """ Add an increasing number to a name until it's unique """
        root = name + '_'
        i = 0
        while name in reverse:
            i += 1
            name = root + str(i)
        return name

    # Get sorted list of variables for consistent output
    sorted_variables = [v for v in model.graph_with_sympy_numbers]
    sorted_variables.sort(key=str)

    # Generate names
    for v in sorted_variables:
        if isinstance(v, sympy.Derivative):
            continue

        # Split off component name (if present, which it might not be for FC created variables)
        name = v.name.split(SYMPY_SYMBOL_DELIMITER)[-1]

        # If already taken, rename _both_ variables using component name
        if name in reverse:

            # Get existing variable
            other = reverse[name]

            # Check it hasn't been renamed already
            if variables[other] == name:
                # Try adding component name, and ensure uniqueness with uname()
                oname = uname(other.name.replace(SYMPY_SYMBOL_DELIMITER, sep))

                variables[other] = oname
                reverse[oname] = other

            # Try adding component name, and ensure uniqueness with uname()
            name = uname(v.name.replace(SYMPY_SYMBOL_DELIMITER, sep))

        # Store variable name
        variables[v] = name
        reverse[name] = v

    return variables


def create_weblab_model(path, output_dir, class_name, model, time_variable, ns_map, protocol_variables):
    """
    Takes a :class:`cellmlmanip.Model`, generates a ``.pyx`` model for use with
    the Web Lab, and stores it at ``path``.

    Arguments

    ``path``
        The path to store the generated model code at.
    ``output_dir``
        The path to store any extra output at.
    ``class_name``
        A name for the generated class.
    ``model``
        A :class:`cellmlmanip.model.Model` object.
    ``time_variable``
        A :class:`cellmlmanip.model.Variable` object representing time.
    ``ns_map``
        A dict mapping namespace prefixes to namespace URIs.
    ``protocol_variables``
        A list of :class:`ProtocolVariable` objects representing variables used by the protocol.

    """
    # TODO: About the outputs:
    # WL1 uses just the local names here, without the base URI part. What we should do eventually is update the
    # ModelWrapperEnvironment so we can use a separate instance for each namespace defined by the protocol, and then we
    # can use longer names here and let each environment wrap its respective subset. But until that happens, users just
    # have to make sure not to use the same local name in different namespaces.

    # Get unique names for all variables
    unames = get_unique_names(model)

    # Variable naming function
    def variable_name(variable):
        if isinstance(variable, DataInterpolation):
            return variable.lookup_call
        return 'var_' + unames[variable]

    # Derivative naming function
    def derivative_name(deriv):
        var = deriv.expr if isinstance(deriv, sympy.Derivative) else deriv
        return 'd_dt_' + unames[var]

    # Create expression printer
    printer = WebLabPrinter(variable_name, derivative_name)

    # Create free variable name
    free_name = variable_name(time_variable)

    # Create state information dicts
    state_info = []
    for i, state in enumerate(model.get_state_variables()):
        state_info.append({
            'index': i,
            'var_name': variable_name(state),
            'deriv_name': derivative_name(state),
            'initial_value': state.initial_value,
            'var_names': model.get_ontology_terms_by_variable(state),
        })

    # Create parameter information dicts, and map of parameter variables to their indices.
    # Parameters are all inputs that are constant w.r.t. time
    parameter_info = []
    parameter_variables = {}
    todo_use_qualified_names = set()    # TODO: Remove this. See above.
    for pvar in protocol_variables:
        if pvar.is_input and pvar.model_variable is not None and model.is_constant(pvar.model_variable):

            # TODO: Remove this. See above.
            if pvar.short_name in todo_use_qualified_names:
                raise NotImplementedError('Need to convert parameter maps to use qualified instead of local names.')
            todo_use_qualified_names.add(pvar.short_name)

            i = len(parameter_info)
            parameter_info.append({
                'index': i,
                'local_name': pvar.short_name,
                'var_name': variable_name(pvar.model_variable),
                'initial_value': model.get_value(pvar.model_variable),
            })
            parameter_variables[pvar.model_variable] = i

    # Create output information dicts
    # Each output is associated either with a variable or a list thereof.
    output_info = []
    output_variables = set()
    todo_use_qualified_names = set()    # TODO: Remove this. See above.
    for pvar in protocol_variables:
        if not pvar.is_output:
            continue
        if pvar.model_variable is not None:
            # Single variable output
            length = None
            var_name = variable_name(pvar.model_variable)
            output_variables.add(pvar.model_variable)
        elif pvar.is_vector:
            # Vector output
            length = len(pvar.vector_variables)
            var_name = [{'index': i, 'var_name': variable_name(v)} for i, v in enumerate(pvar.vector_variables)]
            output_variables.update(pvar.vector_variables)
        else:
            # Optional, unresolved output
            assert pvar.is_optional, 'Unresolved non-optional variable ' + pvar.long_name
            continue

        # TODO: Add an output for each rdf term pointing to the same variable.

        # TODO: Remove this. See above.
        if pvar.short_name in todo_use_qualified_names:
            raise NotImplementedError('Need to convert output maps to use qualified instead of local names.')
        todo_use_qualified_names.add(pvar.short_name)

        output_info.append({
            'index': len(output_info),
            'local_name': pvar.short_name,
            'var_name': var_name,
            'length': length,
        })

    # Track any data interpolations appearing in equations used
    data_tables = set()

    # Create RHS equation information dicts
    rhs_equations = []
    for eq in model.get_equations_for(model.get_derivatives()):
        for table in eq.atoms(DataInterpolation):
            table.set_index_name(variable_name)
            data_tables.add(table)
        # TODO: Parameters should never appear as the left-hand side of an
        # equation (cellmlmanip should already have filtered these out).
        rhs_equations.append({
            'lhs': printer.doprint(eq.lhs),
            'rhs': printer.doprint(eq.rhs),
            'parameter_index': parameter_variables.get(eq.lhs, None),
        })

    # Create output equation information dicts
    output_equations = []
    for eq in model.get_equations_for(output_variables):
        for table in eq.atoms(DataInterpolation):
            table.set_index_name(variable_name)
            data_tables.add(table)
        output_equations.append({
            'lhs': printer.doprint(eq.lhs),
            'rhs': printer.doprint(eq.rhs),
            'parameter_index': parameter_variables.get(eq.lhs, None),
        })

    # Write debug output about the created model
    with open(os.path.join(output_dir, 'model-debug-info.txt'), 'w') as f:
        f.write('=== STATES ' + '=' * 68 + '\n')
        for i in sorted(state_info, key=lambda x: x['index']):
            f.write(f"{i['index']} {i['var_name']}, {i['deriv_name']}, init {i['initial_value']}\n")
            for name in i['var_names']:
                f.write(f"  {name}\n")

        f.write('=== PARAMETERS ' + '=' * 64 + '\n')
        for i in sorted(parameter_info, key=lambda x: x['index']):
            f.write(f"{i['index']} {i['var_name']}, init {i['initial_value']}\n")
            f.write(f"  {i['local_name']}\n")

        f.write('=== OUTPUTS ' + '=' * 67 + '\n')
        for i in sorted(output_info, key=lambda x: x['index']):
            if i['length'] is None:
                f.write(f"{i['index']} {i['var_name']}\n")
                f.write(f"  {i['local_name']}\n")
            else:
                f.write(f"{i['index']} vector: \n")
                for var in sorted(i['var_name'], key=lambda x: x['index']):
                    f.write(f"    {var['index']} {var['var_name']}\n")

                f.write(f"  {i['local_name']}\n")

        f.write('=== OUTPUT EQUATIONS ' + '=' * 58 + '\n')
        for e in output_equations:
            f.write(f"{e['lhs']} = {e['rhs']}\n")
            if e['parameter_index'] is not None:
                f.write(f"  Parameter index {e['parameter_index']}\n")

        f.write('=== RHS EQUATIONS ' + '=' * 61 + '\n')
        for e in rhs_equations:
            f.write(f"{e['lhs']} = {e['rhs']}\n")
            if e['parameter_index'] is not None:
                f.write(f"  Parameter index {e['parameter_index']}\n")

    # Generate model
    template = load_template('weblab_model.pyx')
    with open(path, 'w') as f:
        f.write(template.render({
            'class_name': class_name,
            'free_variable': free_name,
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model.name,
            'ns_map': ns_map,
            'outputs': output_info,
            'output_equations': output_equations,
            'parameters': parameter_info,
            'rhs_equations': rhs_equations,
            'states': state_info,
            'data_tables': data_tables,
        }))
