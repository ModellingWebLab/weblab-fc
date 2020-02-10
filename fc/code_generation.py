"""
Methods for code generation (using jinja2 templates).
"""
import pkg_resources
import posixpath
import time

import jinja2
import rdflib
import sympy

from cellmlmanip import transpiler
from cellmlmanip.printer import Printer
from cellmlmanip.rdf import create_rdf_node

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


_ONTOLOGY = None


def get_variables_transitively(model, term):
    global _ONTOLOGY

    if _ONTOLOGY is None:
        # Load oxmeta ontology
        g = _ONTOLOGY = rdflib.Graph()
        oxmeta_ttl = pkg_resources.resource_stream('fc', 'ontologies/oxford-metadata.ttl')
        g.parse(oxmeta_ttl, format='turtle')

    term = create_rdf_node(term)
    pred_is = create_rdf_node(('http://biomodels.net/biology-qualifiers/', 'is'))
    pred_is_ver = create_rdf_node(('http://biomodels.net/biology-qualifiers/', 'isVersionOf'))

    cmeta_ids = set()
    for annotation in _ONTOLOGY.transitive_subjects(rdflib.RDF.type, term):
        cmeta_ids.update(model.rdf.subjects(pred_is, annotation))
        cmeta_ids.update(model.rdf.subjects(pred_is_ver, annotation))
    symbols = []
    for cmeta_id in cmeta_ids:
        assert isinstance(cmeta_id, rdflib.URIRef), 'Non-resource {} annotated.'.format(cmeta_id)
        cmeta_id = str(cmeta_id)
        if cmeta_id[0] != '#':
            # TODO This should eventually be implemented
            raise NotImplementedError(
                'Non-local annotations are not supported.')
        symbols.append(model.get_symbol_by_cmeta_id(cmeta_id[1:]))
    return sorted(symbols, key=lambda sym: sym.order_added)


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
        An ordered list of annotations ``(namespace_uri, local_name)`` for the
        variables to use as model outputs.
    ``parameters``
        An ordered list of annotations ``(namespace_uri, local_name)`` for the
        variables to use as model parameters. All variables used as parameters
        must be literal constants.
    ``vector_orderings``
        An optional mapping defining custom orderings for vector outputs, instead
        of the default symbol.order_added ordering. Should be a map from annotation
        ``(namespace_uri, local_name)`` to a mapping from cmeta_id to order index.

    """
    # TODO: Jon's comment on the outputs/parameters being annotations:
    # IIRC the pycml code basically says you can use anything that's a valid
    # input to create_rdf_node. So we might eventually want to avoid all the
    # *parameter unpacking when passing around, but I don't think it's urgent.

    # TODO: About the outputs:
    # WL1 uses just the local names here, without the base URI part. What we
    # should do eventually is update the ModelWrapperEnvironment so we can use
    # a separate instance for each namespace defined by the protocol, and then
    # we can use longer names here and let each environment wrap its respective
    # subset. But until that happens, users just have to make sure not to use
    # the same local name in different namespaces.

    # Oxmeta namespace
    oxmeta = 'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata'

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
            'var_names': model.get_ontology_terms_by_symbol(state, oxmeta),
        })

    # Create parameter information dicts, and map of parameter symbols to their indices
    parameter_info = []
    parameter_symbols = {}
    for i, parameter in enumerate(parameters):
        symbol = model.get_symbol_by_ontology_term(*parameter)
        parameter_info.append({
            'index': i,
            'annotation': parameter,
            'var_name': symbol_name(symbol),
            'initial_value': model.get_value(symbol),
        })
        parameter_symbols[symbol] = i

    # Create output information dicts
    print(vector_orderings)
    print('state vector:', [(i, s.cmeta_id) for i, s in enumerate(model.get_state_symbols())])
    # Each output is associated either with a symbol, a parameter, or a list thereof.
    output_info = []
    output_symbols = set()
    for i, output in enumerate(outputs):
        symbols = get_variables_transitively(model, output)
        output_symbols.update(symbols)
        if len(symbols) == 0:
            raise ValueError('No variable annotated as {{{}}}{} found'.format(*output))
        elif len(symbols) == 1:
            length = None  # Not a vector output
            var_name = symbol_name(symbols[0])
            parameter_index = parameter_symbols.get(symbols[0], None)
        else:
            # Vector output
            print(output, output in vector_orderings)
            if output in vector_orderings:
                order = vector_orderings[output]
                symbols.sort(key=lambda s: order[s.cmeta_id])
            length = len(symbols)
            var_name = [{'index': i, 'var_name': symbol_name(s)} for i, s in enumerate(symbols)]
            parameter_index = [parameter_symbols.get(s, None) for s in symbols]

        output_info.append({
            'index': i,
            'annotation': output,
            'var_name': var_name,
            'parameter_index': parameter_index,
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
