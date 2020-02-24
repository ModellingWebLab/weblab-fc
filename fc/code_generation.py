"""
Methods for code generation (using jinja2 templates).
"""
import posixpath
import time

import jinja2
import sympy

from cellmlmanip import transpiler
from cellmlmanip.printer import Printer

from .parsing.rdf import OXMETA_NS, get_variables_transitively

# Add an `_exp` method to sympy, and tell cellmlmanip to create _exp objects instead of exp objects.
# This prevents Sympy doing simplification (or canonicalisation) resulting in weird errors with exps in some cardiac
# models.
setattr(sympy, '_exp', sympy.Function('_exp'))
transpiler.SIMPLE_MATHML_TO_SYMPY_NAMES['exp'] = '_exp'


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
    Creates unique names for all symbols in a CellML model.
    """
    # Component variable separator
    # Note that variables are free to use __ in their names too, it makes the
    # produced code less readable but doesn't break anything.
    sep = '__'

    # Create a symbol => name mapping, and a reverse name => symbol mapping
    symbols = {}
    reverse = {}

    def uname(name):
        """ Add an increasing number to a name until it's unique """
        root = name + '_'
        i = 0
        while name in reverse:
            i += 1
            name = root + str(i)
        return name

    # Get sorted list of symbols for consistent output
    sorted_symbols = [v for v in model.graph_with_sympy_numbers]
    sorted_symbols.sort(key=str)

    # Generate names
    for v in sorted_symbols:
        if isinstance(v, sympy.Derivative):
            continue

        # Try simple name
        parts = v.name.split('$')
        assert len(parts) == 2
        name = parts[-1]

        # If already taken, rename _both_ variables using component name
        if name in reverse:

            # Get existing variable
            other = reverse[name]

            # Check it hasn't been renamed already
            if symbols[other] == name:
                oparts = other.name.split('$')
                assert len(oparts) == 2
                oname = uname(oparts[0] + sep + oparts[1])
                symbols[other] = oname
                reverse[oname] = other

            # Get new name for v
            name = uname(parts[0] + sep + parts[1])

        # Store symbol name
        symbols[v] = name
        reverse[name] = v

    return symbols


def create_weblab_model(path, class_name, model, outputs, parameters, vector_orderings={}):
    """
    Takes a :class:`cellmlmanip.Model`, generates a ``.pyx`` model for use with
    the Web Lab, and stores it at ``path``.

    Arguments

    ``path``
        The path to store the generated model code at.
    ``class_name``
        A name for the generated class.
    ``model``
        A :class:`cellmlmanip.Model` object.
    ``outputs``
        An ordered list of :class:`VariableReference`s for the variables to use as model outputs.
    ``parameters``
        An ordered list of :class:`VariableReference`s for the variables to use as model parameters.
        All variables used as parameters must be literal constants.
    ``vector_orderings``
        An optional mapping defining custom orderings for vector outputs, instead of the default
        ``symbol.order_added`` ordering. Keys are annotations (RDF nodes), and values are mappings
        from ``rdf_identity`` to order index.

    """
    # TODO: About the outputs:
    # WL1 uses just the local names here, without the base URI part. What we
    # should do eventually is update the ModelWrapperEnvironment so we can use
    # a separate instance for each namespace defined by the protocol, and then
    # we can use longer names here and let each environment wrap its respective
    # subset. But until that happens, users just have to make sure not to use
    # the same local name in different namespaces.

    # Get unique names for all symbols
    unames = get_unique_names(model)

    # Symbol naming function
    def symbol_name(symbol):
        return 'var_' + unames[symbol]

    # Derivative naming function
    def derivative_name(deriv):
        var = deriv.expr if isinstance(deriv, sympy.Derivative) else deriv
        return 'd_dt_' + unames[var]

    # Create expression printer
    printer = WebLabPrinter(symbol_name, derivative_name)

    # Create free variable name
    free_name = symbol_name(model.get_free_variable_symbol())

    # Create state information dicts
    state_info = []
    for i, state in enumerate(model.get_state_symbols()):
        state_info.append({
            'index': i,
            'var_name': symbol_name(state),
            'deriv_name': derivative_name(state),
            'initial_value': state.initial_value,
            'var_names': model.get_ontology_terms_by_symbol(state, OXMETA_NS),
        })

    # Create parameter information dicts, and map of parameter symbols to their indices
    parameter_info = []
    parameter_symbols = {}
    for i, parameter in enumerate(parameters):
        symbol = model.get_symbol_by_ontology_term(parameter.rdf_term)
        parameter_info.append({
            'index': i,
            'local_name': parameter.local_name,
            'var_name': symbol_name(symbol),
            'initial_value': model.get_value(symbol),
        })
        parameter_symbols[symbol] = i

    # Create output information dicts
    # Each output is associated either with a symbol or a list thereof.
    output_info = []
    output_symbols = set()
    for i, output in enumerate(outputs):
        term = output.rdf_term
        symbols = get_variables_transitively(model, term)
        output_symbols.update(symbols)
        if len(symbols) == 0:
            raise ValueError('No variable annotated as {} found'.format(term))
        elif len(symbols) == 1:
            length = None  # Not a vector output
            var_name = symbol_name(symbols[0])
        else:
            # Vector output
            if term in vector_orderings:
                order = vector_orderings[term]
                symbols.sort(key=lambda s: order[s.rdf_identity])
            length = len(symbols)
            var_name = [{'index': i, 'var_name': symbol_name(s)} for i, s in enumerate(symbols)]

        output_info.append({
            'index': i,
            'local_name': output.local_name,
            'var_name': var_name,
            'length': length,
        })

    # Create RHS equation information dicts
    rhs_equations = []
    for eq in model.get_equations_for(model.get_derivative_symbols()):
        # TODO: Parameters should never appear as the left-hand side of an
        # equation (cellmlmanip should already have filtered these out).
        rhs_equations.append({
            'lhs': printer.doprint(eq.lhs),
            'rhs': printer.doprint(eq.rhs),
            'parameter_index': parameter_symbols.get(eq.lhs, None),
        })

    # Create output equation information dicts
    output_equations = []
    for eq in model.get_equations_for(output_symbols):
        output_equations.append({
            'lhs': printer.doprint(eq.lhs),
            'rhs': printer.doprint(eq.rhs),
            'parameter_index': parameter_symbols.get(eq.lhs, None),
        })

    # Generate model
    template = load_template('weblab_model.pyx')
    with open(path, 'w') as f:
        f.write(template.render({
            'class_name': class_name,
            'free_variable': free_name,
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model.name,
            'outputs': output_info,
            'output_equations': output_equations,
            'parameters': parameter_info,
            'rhs_equations': rhs_equations,
            'states': state_info,
        }))
