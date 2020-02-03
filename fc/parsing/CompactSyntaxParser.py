
import sys

import pyparsing as p

from . import actions


__all__ = ['CompactSyntaxParser']

# Necessary for reasonable speed when using infixNotation
p.ParserElement.enablePackrat()


################################################################################
# Helper methods for defining parsers
################################################################################
def make_kw(keyword, suppress=True):
    """Helper function to create a parser for the given keyword."""
    kw = p.Keyword(keyword)
    if suppress:
        kw = kw.suppress()
    return kw


def adjacent(parser):
    """Create a copy of the given parser that doesn't permit whitespace to occur before it."""
    adj = parser.copy()
    adj.setWhitespaceChars('')
    return adj


class Optional(p.Optional):
    """An Optional pattern that doesn't consume whitespace if the contents don't match."""

    def __init__(self, *args, **kwargs):
        super(Optional, self).__init__(*args, **kwargs)
        self.callPreparse = False
        self._optionalNotMatched = p.Optional(p.Empty()).defaultValue

    def parseImpl(self, instring, loc, doActions=True):
        try:
            loc, tokens = self.expr._parse(instring, loc, doActions, callPreParse=True)
        except (p.ParseException, IndexError):
            if self.defaultValue is not self._optionalNotMatched:
                if self.expr.resultsName:
                    tokens = p.ParseResults([self.defaultValue])
                    tokens[self.expr.resultsName] = self.defaultValue
                else:
                    tokens = [self.defaultValue]
            else:
                tokens = []
        return loc, tokens


def optional_delimited_list(expr, delim):
    """Like delimitedList, but the list may be empty."""
    return p.delimitedList(expr, delim) | p.Empty()


def delimited_multi_list(elements, delimiter):
    """Like delimitedList, but allows for a sequence of constituent element expressions.

    elements should be a sequence of tuples (expr, unbounded), where expr is a ParserElement,
    and unbounded is True iff zero or more occurrences are allowed; otherwise the expr is
    considered to be optional (i.e. 0 or 1 occurrences).  The delimiter parameter must occur
    in between each matched token, and is suppressed from the output.
    """
    if len(elements) == 0:
        return p.Empty()
    # If we have an optional expr, we need (expr + delimiter + rest) | expr | rest
    # If we have an unbounded expr, we need (expr + delimiter + this) | expr | rest, i.e. allow expr to recur
    expr, unbounded = elements[0]
    if not isinstance(delimiter, p.Suppress):
        delimiter = p.Suppress(delimiter)
    rest = delimited_multi_list(elements[1:], delimiter)
    if unbounded:
        result = p.Forward()
        result << ((expr + delimiter + result) | expr | rest)
    else:
        if isinstance(rest, p.Empty):
            result = expr | rest
        else:
            result = (expr + delimiter + rest) | expr | rest
    return result


def unignore(parser):
    """Stop ignoring things in the given parser (and its children)."""
    for child in getattr(parser, 'exprs', []):
        unignore(child)
    if hasattr(parser, 'expr'):
        unignore(parser.expr)
    parser.ignoreExprs = []


def monkey_patch_pyparsing():
    """Monkey-patch some pyparsing methods to behave slightly differently."""

    def ignore(self, other):
        """Improved ignore that avoids ignoring self by accident."""
        if isinstance(other, p.Suppress):
            if other not in self.ignoreExprs and other != self:
                self.ignoreExprs.append(other.copy())
        else:
            self.ignoreExprs.append(p.Suppress(other.copy()))
        return self
    p.ParserElement.ignore = ignore

    def err_str(self):
        """Extended exception reporting that also prints the offending line with an error marker underneath."""
        return "%s (at char %d), (line:%d, col:%d):\n%s\n%s" % (self.msg, self.loc, self.lineno, self.column, self.line,
                                                                ' ' * (self.column - 1) + '^')
    p.ParseException.__str__ = err_str


monkey_patch_pyparsing()


class CompactSyntaxParser(object):
    """A parser for the compact textual syntax for protocols."""

    # Newlines are significant most of the time for us
    p.ParserElement.setDefaultWhitespaceChars(' \t\r')

    # Single-line Python-style comments
    comment = p.Regex(r'#.*').suppress().setName('Comment')

    # Punctuation etc.
    eq = p.Suppress('=')
    colon = p.Suppress(':')
    comma = p.Suppress(',')
    oparen = p.Suppress('(')
    cparen = p.Suppress(')')
    osquare = p.Suppress('[')
    csquare = p.Suppress(']')
    dollar = p.Suppress('$')
    nl = p.Suppress(p.OneOrMore(Optional(comment) + p.LineEnd())
                    ).setName('Newline(s)')  # Any line can end with a comment
    obrace = (Optional(nl) + p.Suppress('{') + Optional(nl)).setName('{')
    cbrace = (Optional(nl) + p.Suppress('}') + Optional(nl)).setName('}')
    embedded_cbrace = (Optional(nl) + p.Suppress('}')).setName('}')

    # Identifiers
    nc_ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*').setName('ncIdent')
    c_ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*:[_a-zA-Z][_0-9a-zA-Z]*').setName('cIdent')
    ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*(:[_a-zA-Z][_0-9a-zA-Z]*)*').setName('Ident')
    nc_ident_as_var = nc_ident.copy().setParseAction(actions.Variable)
    ident_as_var = ident.copy().setParseAction(actions.Variable)

    # Numbers can be given in scientific notation, with an optional leading minus sign.
    # Within expressions they may also have units specified, e.g. in the model interface.
    units_ident = p.originalTextFor(p.Literal('units_of(') - adjacent(ident) + adjacent(p.Literal(')'))) | nc_ident
    units_annotation = p.Suppress('::') - units_ident("units")
    plain_number = p.Regex(r'-?[0-9]+((\.[0-9]+)?(e[-+]?[0-9]+)?)?').setName('Number')
    number = (plain_number + Optional(units_annotation)).setName('Number')

    # Used for descriptive text
    quoted_string = (p.QuotedString('"', escChar="\\") | p.QuotedString("'", escChar="\\")).setName('QuotedString')
    # This may become more specific in future
    quoted_uri = quoted_string.copy().setName('QuotedUri')

    # Expressions from the "post-processing" language
    #################################################

    # Expressions and statements must be constructed recursively
    expr = p.Forward().setName('Expression')
    stmt_list = p.Forward().setName('StatementList')

    # A vector written like 1:2:5 or 1:5 or A:B:C
    numeric_range = p.Group(expr + colon - expr + Optional(colon - expr))

    # Creating arrays
    dim_spec = Optional(expr + adjacent(dollar)) + nc_ident
    comprehension = p.Group(make_kw('for') - dim_spec + make_kw('in') - numeric_range).setParseAction(actions.Comprehension)
    array = p.Group(osquare - expr + (p.OneOrMore(comprehension) | p.ZeroOrMore(comma - expr)) + csquare
                    ).setName('Array').setParseAction(actions.Array)

    # Array views
    opt_expr = Optional(expr, default='')
    view_spec = p.Group(adjacent(osquare) - Optional(('*' | expr) + adjacent(dollar))('dimspec') +
                       opt_expr + Optional(colon - opt_expr + Optional(colon - opt_expr)) + csquare).setName('ViewSpec')

    # If-then-else
    if_expr = p.Group(make_kw('if') - expr + make_kw('then') - expr +
                     make_kw('else') - expr).setName('IfThenElse').setParseAction(actions.Piecewise)

    # Lambda definitions
    param_decl = p.Group(nc_ident_as_var + Optional(eq + expr))
    param_list = p.Group(optional_delimited_list(param_decl, comma))
    lambda_expr = p.Group(make_kw('lambda') - param_list + ((colon - expr) | (obrace - stmt_list + embedded_cbrace))
                         ).setName('Lambda').setParseAction(actions.Lambda)

    # Function calls
    # TODO: Allow lambdas, not just ident?
    arg_list = p.Group(optional_delimited_list(expr, comma))
    function_call = p.Group(ident_as_var + adjacent(oparen) - arg_list +
                           cparen).setName('FnCall').setParseAction(actions.FunctionCall)

    # Tuples
    tuple = p.Group(oparen + expr + comma - optional_delimited_list(expr, comma) +
                    cparen).setName('Tuple').setParseAction(actions.Tuple)

    # Accessors
    accessor = p.Combine(adjacent(p.Suppress('.')) -
                         p.oneOf('IS_SIMPLE_VALUE IS_ARRAY IS_STRING IS_TUPLE IS_FUNCTION IS_NULL IS_DEFAULT '
                                 'NUM_DIMS NUM_ELEMENTS SHAPE')).setName('Accessor')

    # Indexing
    pad = (make_kw('pad') + adjacent(colon) - expr + eq + expr).setResultsName('pad')
    shrink = (make_kw('shrink') + adjacent(colon) - expr).setResultsName('shrink')
    index_dim = expr.setResultsName('dim')
    index = p.Group(adjacent(p.Suppress('{')) - expr +
                    p.ZeroOrMore(comma - (pad | shrink | index_dim)) + p.Suppress('}')).setName('Index')

    # Special values
    null_value = p.Group(make_kw('null')).setName('Null').setParseAction(actions.Symbol('null'))
    default_value = p.Group(make_kw('default')).setName('Default').setParseAction(actions.Symbol('defaultParameter'))
    string_value = quoted_string.copy().setName('String').setParseAction(actions.Symbol('string'))

    # Recognised MathML operators
    mathml_operators = set('''
        quotient rem max min root xor abs floor ceiling exp ln log
        sin cos tan
        sec csc cot
        sinh cosh tanh
        sech csch coth
        arcsin arccos arctan
        arccosh arccot arccoth
        arccsc arccsch arcsec
        arcsech arcsinh arctanh
        '''.split())

    # Wrapping MathML operators into lambdas
    mathml_operator = (
        p.oneOf('^ * / + - not == != <= >= < > && ||') |
        p.Combine('MathML:' + p.oneOf(' '.join(mathml_operators))))
    wrap = p.Group(
            p.Suppress('@') - adjacent(p.Word(p.nums)) + adjacent(colon) + mathml_operator
        ).setName('WrapMathML').setParseAction(actions.Wrap)

    # Turning on tracing for debugging protocols
    trace = adjacent(p.Suppress('?'))

    # The main expression grammar.  Atoms are ordered according to rough speed of detecting mis-match.
    atom = (array | wrap | number.copy().setParseAction(actions.Number) | string_value |
            if_expr | null_value | default_value | lambda_expr | function_call | ident_as_var | tuple).setName('Atom')
    expr <<= p.infixNotation(atom, [(accessor, 1, p.opAssoc.LEFT, actions.Accessor),
                                    (view_spec, 1, p.opAssoc.LEFT, actions.View),
                                    (index, 1, p.opAssoc.LEFT, actions.Index),
                                    (trace, 1, p.opAssoc.LEFT, actions.Trace),
                                    ('^', 2, p.opAssoc.LEFT, actions.Operator),
                                    ('-', 1, p.opAssoc.RIGHT,
                                        lambda *args: actions.Operator(*args, rightAssoc=True)),
                                    (p.oneOf('* /'), 2, p.opAssoc.LEFT, actions.Operator),
                                    (p.oneOf('+ -'), 2, p.opAssoc.LEFT, actions.Operator),
                                    (p.Keyword('not'), 1, p.opAssoc.RIGHT,
                                     lambda *args: actions.Operator(*args, rightAssoc=True)),
                                    (p.oneOf('== != <= >= < >'), 2, p.opAssoc.LEFT, actions.Operator),
                                    (p.oneOf('&& ||'), 2, p.opAssoc.LEFT, actions.Operator)
                                    ])

    # Simpler expressions containing no arrays, functions, etc. Used in the model interface.
    simple_expr = p.Forward().setName('SimpleExpression')
    simple_if_expr = p.Group(make_kw('if') - simple_expr + make_kw('then') - simple_expr +
                           make_kw('else') - simple_expr).setName('SimpleIfThenElse').setParseAction(actions.Piecewise)
    simple_arg_list = p.Group(optional_delimited_list(simple_expr, comma))
    simple_function_call = p.Group(ident_as_var + adjacent(oparen) - simple_arg_list +
                                 cparen).setName('SimpleFnCall').setParseAction(actions.FunctionCall)
    simple_expr <<= p.infixNotation(number.copy().setParseAction(actions.Number) |
                                   simple_if_expr | simple_function_call | ident_as_var,
                                   [('^', 2, p.opAssoc.LEFT, actions.Operator),
                                    ('-', 1, p.opAssoc.RIGHT,
                                        lambda *args: actions.Operator(*args, rightAssoc=True)),
                                    (p.oneOf('* /'), 2, p.opAssoc.LEFT, actions.Operator),
                                    (p.oneOf('+ -'), 2, p.opAssoc.LEFT, actions.Operator),
                                    (p.Keyword('not'), 1, p.opAssoc.RIGHT,
                                     lambda *args: actions.Operator(*args, rightAssoc=True)),
                                    (p.oneOf('== != <= >= < >'), 2, p.opAssoc.LEFT, actions.Operator),
                                    (p.oneOf('&& ||'), 2, p.opAssoc.LEFT, actions.Operator)
                                    ])
    simple_param_list = p.Group(optional_delimited_list(p.Group(nc_ident_as_var), comma))
    simple_lambda_expr = p.Group(make_kw('lambda') - simple_param_list + colon -
                               expr).setName('SimpleLambda').setParseAction(actions.Lambda)

    # Newlines in expressions may be escaped with a backslash
    expr.ignore('\\' + p.LineEnd())
    simple_expr.ignore('\\' + p.LineEnd())
    # Bare newlines are OK provided we started with a bracket.
    # However, it's quite hard to enforce that restriction.
    expr.ignore(p.Literal('\n'))
    simple_expr.ignore(p.Literal('\n'))
    # Embedded comments are also OK
    expr.ignore(comment)
    simple_expr.ignore(comment)
    # Avoid mayhem
    unignore(nl)

    # Statements from the "post-processing" language
    ################################################

    # Simple assignment (i.e. not to a tuple)
    simple_assign = p.Group(nc_ident_as_var + eq - expr).setName('SimpleAssign').setParseAction(actions.Assignment)
    simple_assign_list = p.Group(optional_delimited_list(simple_assign, nl)).setParseAction(actions.StatementList)

    # Assertions and function returns
    assert_stmt = p.Group(make_kw('assert') - expr).setName('AssertStmt').setParseAction(actions.Assert)
    return_stmt = p.Group(make_kw('return') - p.delimitedList(expr)).setName('ReturnStmt').setParseAction(actions.Return)

    # Full assignment, to a tuple of names or single name
    _idents = p.Group(p.delimitedList(nc_ident_as_var)).setParseAction(actions.MaybeTuple)
    assign_stmt = p.Group(((make_kw('optional', suppress=False)("optional") + _idents) | _idents) + eq -
                         p.Group(p.delimitedList(expr)).setParseAction(actions.MaybeTuple))   \
        .setName('AssignStmt').setParseAction(actions.Assignment)

    # Function definition
    function_defn = p.Group(make_kw('def') - nc_ident_as_var + oparen + param_list + cparen -
                           ((colon - expr) | (obrace - stmt_list + Optional(nl) + p.Suppress('}')))
                           ).setName('FunctionDef').setParseAction(actions.FunctionDef)

    stmt_list << p.Group(p.delimitedList(assert_stmt | return_stmt | function_defn | assign_stmt, nl))
    stmt_list.setParseAction(actions.StatementList)

    # Miscellaneous constructs making up protocols
    ##############################################

    # Documentation (Markdown)
    documentation = p.Group(make_kw('documentation') - obrace - p.Regex("[^}]*") + cbrace)("dox")

    # Namespace declarations
    ns_decl = p.Group(make_kw('namespace') - nc_ident("prefix") + eq + quoted_uri("uri")).setName('NamespaceDecl')
    ns_decls = optional_delimited_list(ns_decl("namespace*"), nl)

    # Protocol input declarations, with default values
    inputs = (make_kw('inputs') - obrace - simple_assign_list + cbrace).setName('Inputs').setParseAction(actions.Inputs)

    # Import statements
    import_stmt = p.Group(
        make_kw('import') -
        Optional(
            nc_ident +
            eq,
            default='') +
        quoted_uri +
        Optional(
            obrace -
            simple_assign_list +
            embedded_cbrace)).setName('Import').setParseAction(
                actions.Import)
    imports = optional_delimited_list(import_stmt, nl).setName('Imports')

    # Library, globals defined using post-processing language.
    # Strictly speaking returns aren't allowed, but that gets picked up later.
    library = (make_kw('library') - obrace - Optional(stmt_list) +
               cbrace).setName('Library').setParseAction(actions.Library)

    # Post-processing
    post_processing = (make_kw('post-processing') + obrace -
                      optional_delimited_list(assert_stmt | return_stmt | function_defn | assign_stmt, nl) +
                      cbrace).setName('PostProc').setParseAction(actions.PostProcessing)

    # Units definitions
    si_prefix = p.oneOf('deka hecto kilo mega giga tera peta exa zetta yotta'
                       'deci centi milli micro nano pico femto atto zepto yocto')
    _num_or_expr = p.originalTextFor(plain_number | (oparen + expr + cparen))
    unit_ref = p.Group(Optional(_num_or_expr)("multiplier") + Optional(si_prefix)("prefix") + nc_ident("units") +
                      Optional(p.Suppress('^') + plain_number)("exponent") +
                      Optional(p.Group(p.oneOf('- +') + _num_or_expr))("offset")).setParseAction(actions.UnitRef)
    units_def = p.Group(nc_ident + eq + p.delimitedList(unit_ref, '.') + Optional(quoted_string)("description")
                       ).setName('UnitsDefinition').setParseAction(actions.UnitsDef)
    units = (make_kw('units') - obrace - optional_delimited_list(units_def, nl) + cbrace
             ).setName('Units').setParseAction(actions.Units)

    # Model interface section
    #########################
    units_ref = make_kw('units') - nc_ident

    # Setting the units for the independent variable
    set_time_units = (make_kw('independent') - make_kw('var') - units_ref("units")).setParseAction(actions.SetTimeUnits)

    # Input variables, with optional units and initial value
    input_variable = p.Group(
        make_kw('input') -
        c_ident('name') +
        Optional(units_ref)('units') +
        Optional(eq + plain_number)('initial_value')
    ).setName('InputVariable').setParseAction(actions.InputVariable)

    # Model outputs of interest, with optional units
    output_variable = p.Group(
        make_kw('output') -
        c_ident("name") +
        Optional(units_ref("units"))
    ).setName('OutputVariable').setParseAction(actions.OutputVariable)

    # Model variables (inputs, outputs, or just used in equations) that are allowed to be missing
    locator = p.Empty().leaveWhitespace().setParseAction(lambda s, l, t: l)
    var_default = make_kw('default') - locator("default_start") + simple_expr("default")
    optional_variable = p.Group(make_kw('optional') - c_ident("name") + Optional(var_default) + locator("default_end")
                               ).setName('OptionalVar').setParseAction(actions.OptionalVariable)

    # New variables added to the model, with optional initial value
    new_variable = p.Group(
        make_kw('var') -
        nc_ident("name") +
        units_ref("units") +
        Optional(
            eq +
            plain_number)("initial_value")).setName('NewVariable').setParseAction(
        actions.DeclareVariable)
    # Adding or replacing equations in the model
    clamp_variable = p.Group(make_kw('clamp') - ident_as_var + Optional(make_kw('to') - simple_expr)
                            ).setName('ClampVariable').setParseAction(actions.ClampVariable)
    interpolate = p.Group(
        make_kw('interpolate') -
        oparen -
        quoted_string -
        comma -
        ident_as_var -
        comma -
        nc_ident -
        comma -
        nc_ident -
        cparen).setName('Interpolate').setParseAction(
        actions.Interpolate)
    model_equation = p.Group(make_kw('define') -
                            (p.Group(make_kw('diff') +
                                     adjacent(oparen) -
                                     ident_as_var +
                                     p.Suppress(';') +
                                     ident_as_var +
                                     cparen) | ident_as_var) +
                            eq +
                            (interpolate | simple_expr)
                            ).setName('AddOrReplaceEquation').setParseAction(actions.ModelEquation)
    # Units conversion rules
    units_conversion = p.Group(
        make_kw('convert') -
        nc_ident("actualDimensions") +
        make_kw('to') +
        nc_ident("desiredDimensions") +
        make_kw('by') -
        simple_lambda_expr).setName('UnitsConversion').setParseAction(
        actions.UnitsConversion)

    model_interface = p.Group(make_kw('model') - make_kw('interface') - obrace -
                             Optional(set_time_units - nl) +
                             optional_delimited_list((input_variable | output_variable | optional_variable | new_variable |
                                                    clamp_variable | model_equation | units_conversion), nl) +
                             cbrace).setName('ModelInterface').setParseAction(actions.ModelInterface)

    # Simulation definitions
    ########################

    # Ranges
    uniform_range = make_kw('uniform') + numeric_range
    vector_range = make_kw('vector') + expr
    while_range = make_kw('while') + expr
    range = p.Group(make_kw('range') + nc_ident("name") + units_ref("units") +
                    (uniform_range("uniform") | vector_range("vector") | while_range("while"))
                    ).setName('Range').setParseAction(actions.Range)

    # Modifiers
    modifier_when = make_kw('at') - (make_kw('start', False) |
                                   (make_kw('each', False) - make_kw('loop')) |
                                   make_kw('end', False)).setParseAction(actions.ModifierWhen)
    set_variable = make_kw('set') - ident + eq + expr
    save_state = make_kw('save') - make_kw('as') - nc_ident
    reset_state = make_kw('reset') - Optional(make_kw('to') + nc_ident)
    modifier = p.Group(modifier_when + p.Group(set_variable("set") | save_state("save") | reset_state("reset"))
                       ).setName('Modifier').setParseAction(actions.Modifier)
    modifiers = p.Group(make_kw('modifiers') + obrace - optional_delimited_list(modifier, nl) + cbrace
                        ).setName('Modifiers').setParseAction(actions.Modifiers)

    # The simulations themselves
    simulation = p.Forward().setName('Simulation')
    _select_output = p.Group(make_kw('select') - Optional(make_kw('optional', suppress=False)) -
                            make_kw('output') - nc_ident).setName('SelectOutput')
    nested_protocol = p.Group(make_kw('protocol') - quoted_uri + obrace +
                             simple_assign_list + Optional(nl) + optional_delimited_list(_select_output, nl) +
                             cbrace + Optional('?')).setName('NestedProtocol').setParseAction(actions.NestedProtocol)
    timecourse_sim = p.Group(make_kw('timecourse') - obrace - range + Optional(nl + modifiers) + cbrace
                            ).setName('TimecourseSim').setParseAction(actions.TimecourseSimulation)
    nested_sim = p.Group(make_kw('nested') - obrace - range + nl + Optional(modifiers) +
                        p.Group(make_kw('nests') + (simulation | nested_protocol | ident)) +
                        cbrace).setName('NestedSim').setParseAction(actions.NestedSimulation)
    one_step_sim = p.Group(make_kw('oneStep') - Optional(p.originalTextFor(expr))("step") +
                         Optional(obrace - modifiers + cbrace)("modifiers")).setParseAction(actions.OneStepSimulation)
    simulation << p.Group(make_kw('simulation') - Optional(nc_ident + eq, default='') +
                          (timecourse_sim | nested_sim | one_step_sim) -
                          Optional('?' + nl)).setParseAction(actions.Simulation)

    tasks = p.Group(make_kw('tasks') + obrace - p.ZeroOrMore(simulation) +
                    cbrace).setName('Tasks').setParseAction(actions.Tasks)

    # Output specifications
    #######################

    output_desc = Optional(quoted_string)("description")
    output_spec = p.Group(Optional(make_kw('optional', suppress=False))("optional") +
                         nc_ident("name") +
                         ((units_ref("units") +
                           output_desc) | (eq +
                                          ident("ref") +
                                          Optional(units_ref)("units") +
                                          output_desc))).setName('Output').setParseAction(actions.Output)
    outputs = p.Group(make_kw('outputs') + obrace - optional_delimited_list(output_spec, nl) +
                      cbrace).setName('Outputs').setParseAction(actions.Outputs)

    # Plot specifications
    #####################

    plot_curve = p.Group(p.delimitedList(nc_ident, ',') +
                        make_kw('against') - nc_ident +
                        Optional(make_kw('key') - nc_ident("key"))).setName('Curve')
    plot_using = (make_kw('using') - (make_kw('lines', suppress=False) |
                                    make_kw('points', suppress=False) |
                                    make_kw('linespoints', suppress=False)))("using")
    plot_spec = p.Group(make_kw('plot') - quoted_string + Optional(plot_using) - obrace +
                       plot_curve + p.ZeroOrMore(nl + plot_curve) + cbrace).setName('Plot').setParseAction(actions.Plot)
    plots = p.Group(make_kw('plots') + obrace - p.ZeroOrMore(plot_spec) +
                    cbrace).setName('Plots').setParseAction(actions.Plots)

    # Parsing a full protocol
    #########################

    protocol = p.And(
        list(map(Optional, [
            nl,
            documentation,
            ns_decls + nl,
            inputs,
            imports + nl,
            library,
            units,
            model_interface,
            tasks,
            post_processing,
            outputs,
            plots,
        ]))).setName('Protocol').setParseAction(actions.Protocol)

    def __init__(self):
        """Initialise the parser."""
        # We just store the original stack limit here, so we can increase
        # it for the lifetime of this object if needed for parsing, on the
        # basis that if one expression needs to, several are likely to.
        self._stack_depth_factor = 1
        self._original_stack_limit = sys.getrecursionlimit()
        self.increase_stack_depth_limit()

    def __del__(self):
        """Reset the stack limit if it changed."""
        sys.setrecursionlimit(self._original_stack_limit)

    def increase_stack_depth_limit(self, step=0.5):
        """Increase the limit by the given factor of the original."""
        self._stack_depth_factor += 0.5
        new_limit = int(
            self._stack_depth_factor * self._original_stack_limit)
        print('Increasing recursion limit to', new_limit,
              file=sys.stderr)
        sys.setrecursionlimit(new_limit)

    def try_parse(self, callable, source_file, *args, **kwargs):
        """
        Try calling the given parse command, increasing the stack depth limit
        if needed.
        """
        orig_source_file = actions.source_file
        actions.source_file = source_file
        r = None  # Result
        while self._stack_depth_factor < 3:
            try:
                r = callable(source_file, *args, **kwargs)
            except RuntimeError as msg:
                print('Got RuntimeError:', msg, file=sys.stderr)
                self.increase_stack_depth_limit()
            else:
                break  # Parsed OK
        if not r:
            raise RuntimeError("Failed to parse expression even with a recursion limit of %d; giving up!"
                               % (int(self._stack_depth_factor * self._original_stack_limit),))
        actions.source_file = orig_source_file
        return r

################################################################################
# Parser debugging support
################################################################################

def get_named_grammars(obj=CompactSyntaxParser):
    """Get a list of all the named grammars in the given object."""
    grammars = []
    for parser in dir(obj):
        parser = getattr(obj, parser)
        if isinstance(parser, p.ParserElement):
            grammars.append(parser)
    return grammars


def enable_debug(grammars=None):
    """Enable debugging of our (named) grammars."""
    def display_loc(instring, loc):
        return " at loc " + str(loc) + "(%d,%d)" % (p.lineno(loc, instring), p.col(loc, instring))

    def success_debug_action(instring, startloc, endloc, expr, toks):
        print("Matched " + str(expr) + " -> " + str(toks.asList()) + display_loc(instring, endloc))

    def exception_debug_action(instring, loc, expr, exc):
        print("Exception raised:" + str(exc) + display_loc(instring, loc))

    for parser in grammars or get_named_grammars():
        parser.setDebugActions(None, success_debug_action, exception_debug_action)


def disable_debug(grammars=None):
    """Stop debugging our (named) grammars."""
    for parser in grammars or get_named_grammars():
        parser.setDebug(False)


class Debug(object):
    """A Python 2.6+ context manager that enables debugging just for the enclosed block."""

    def __init__(self, grammars=None):
        self._grammars = list(grammars or get_named_grammars())

    def __enter__(self):
        enable_debug(self._grammars)

    def __exit__(self, type, value, traceback):
        disable_debug(self._grammars)

