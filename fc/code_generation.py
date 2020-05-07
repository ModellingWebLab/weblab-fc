"""
Methods for code generation (using jinja2 templates).
"""
import os
import posixpath
import time

import jinja2
import sympy

from cellmlmanip.parser import SYMPY_SYMBOL_DELIMITER, Transpiler
from cellmlmanip.printer import Printer

# Tell cellmlmanip to create _exp objects instead of exp objects. This prevents Sympy doing simplification (or
# canonicalisation) resulting in weird errors with exps in some cardiac models.
Transpiler.set_mathml_handler('exp', sympy.Function('_exp'))


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


class WebLabPrinter(Printer):
    """
    Cellmlmanip ``Printer`` subclass for the Web Lab.
    """

    def __init__(self, symbol_function=None, derivative_function=None):
        super().__init__(symbol_function, derivative_function)

        # Deal with _exp function introduced to avoid simplification
        self._function_names['_exp'] = 'math.exp'


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

    # Create RHS equation information dicts
    rhs_equations = []
    for eq in model.get_equations_for(model.get_derivatives()):
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
        }))
