"""
Parsing methods for the protocol language.

The resulting data structure is defined in the :mod:`.actions` module.
"""
import os
import pickle
import sys
import time

import pyparsing as p

from . import actions


__all__ = ['CompactSyntaxParser']

# Necessary for reasonable speed when using infixNotation
p.ParserElement.enable_packrat()


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
    adj.set_whitespace_chars('')
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
    """Like DelimitedList, but the list may be empty."""
    return p.DelimitedList(expr, delim) | p.Empty()


def delimited_multi_list(elements, delimiter):
    """Like DelimitedList, but allows for a sequence of constituent element expressions.

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
    p.ParserElement.set_default_whitespace_chars(' \t\r')

    # Single-line Python-style comments
    comment = p.Regex(r'#.*').suppress().set_name('comment')

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
                    ).set_name('newline(s)')  # Any line can end with a comment
    obrace = (Optional(nl) + p.Suppress('{') + Optional(nl)).set_name('{')
    cbrace = (Optional(nl) + p.Suppress('}') + Optional(nl)).set_name('}')
    embedded_cbrace = (Optional(nl) + p.Suppress('}')).set_name('}')

    # Identifiers
    nc_ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*').set_name('non-prefixed identifier')
    c_ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*:[_a-zA-Z][_0-9a-zA-Z]*').set_name('prefixed identifier')
    ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*(:[_a-zA-Z][_0-9a-zA-Z]*)*').set_name('identifier (with or without prefix)')
    nc_ident_as_var = nc_ident.copy().set_parse_action(actions.Variable)
    ident_as_var = ident.copy().set_parse_action(actions.Variable)

    # Numbers can be given in scientific notation, with an optional leading minus sign.
    # Within expressions they may also have units specified, e.g. in the model interface.
    units_ident = p.original_text_for(p.Literal('units_of(') - adjacent(ident) + adjacent(p.Literal(')'))) | nc_ident
    units_annotation = p.Suppress('::') - units_ident("units")
    plain_number = p.Regex(r'-?[0-9]+((\.[0-9]+)?(e[-+]?[0-9]+)?)?').set_name('number')
    number = (plain_number + Optional(units_annotation)).set_name('number or quantity')

    # Used for descriptive text
    quoted_string = (p.QuotedString('"', esc_char="\\") | p.QuotedString("'", esc_char="\\")).set_name('quoted string')
    # This may become more specific in future
    quoted_uri = quoted_string.copy().set_name('quoted uri')

    # Expressions from the "post-processing" language
    #################################################

    # Expressions and statements must be constructed recursively
    expr = p.Forward().set_name('expression')
    stmt_list = p.Forward().set_name('statement list')

    # A vector written like 1:2:5 or 1:5 or A:B:C
    numeric_range = p.Group(expr + colon - expr + Optional(colon - expr))

    # Creating arrays
    dim_spec = Optional(expr + adjacent(dollar)) + nc_ident
    comprehension = p.Group(
        make_kw('for') - dim_spec + make_kw('in') - numeric_range).set_parse_action(actions.Comprehension)
    array = p.Group(osquare - expr + (p.OneOrMore(comprehension) | p.ZeroOrMore(comma - expr)) + csquare
                    ).set_name('array').set_parse_action(actions.Array)

    # Array views
    opt_expr = Optional(expr, default='')
    view_spec = p.Group(
        adjacent(osquare) - Optional(('*' | expr) + adjacent(dollar))('dimspec') +
        opt_expr + Optional(colon - opt_expr + Optional(colon - opt_expr)) + csquare
    ).set_name('view specification on an array')

    # If-then-else
    if_expr = p.Group(make_kw('if') - expr + make_kw('then') - expr +
                      make_kw('else') - expr).set_name('if-then-else expression').set_parse_action(actions.Piecewise)

    # Lambda definitions
    param_decl = p.Group(nc_ident_as_var + Optional(eq + expr))
    param_list = p.Group(optional_delimited_list(param_decl, comma))
    lambda_expr = p.Group(make_kw('lambda') - param_list + ((colon - expr) | (obrace - stmt_list + embedded_cbrace))
                          ).set_name('lambda function').set_parse_action(actions.Lambda)

    # Function calls
    # TODO: Allow lambdas, not just ident?
    arg_list = p.Group(optional_delimited_list(expr, comma))
    function_call = p.Group(ident_as_var + adjacent(oparen) - arg_list +
                            cparen).set_name('function call').set_parse_action(actions.FunctionCall)

    # Tuples
    tuple = p.Group(oparen + expr + comma - optional_delimited_list(expr, comma) +
                    cparen).set_name('tuple').set_parse_action(actions.Tuple)

    # Accessors
    accessor = p.Combine(adjacent(p.Suppress('.')) -
                         p.one_of('IS_SIMPLE_VALUE IS_ARRAY IS_STRING IS_TUPLE IS_FUNCTION IS_NULL IS_DEFAULT '
                                 'NUM_DIMS NUM_ELEMENTS SHAPE')).set_name('.accessor (e.g. .IS_ARRAY)')

    # Indexing
    pad = (make_kw('pad') + adjacent(colon) - expr + eq + expr).set_results_name('pad')
    shrink = (make_kw('shrink') + adjacent(colon) - expr).set_results_name('shrink')
    index_dim = expr.set_results_name('dim')
    index = p.Group(adjacent(p.Suppress('{')) - expr +
                    p.ZeroOrMore(comma - (pad | shrink | index_dim)) + p.Suppress('}')).set_name('index expression')

    # Special values
    null_value = p.Group(make_kw('null')).set_name('null').set_parse_action(actions.Symbol('null'))
    default_value = p.Group(make_kw('default')).set_name('default').set_parse_action(actions.Symbol('defaultParameter'))
    string_value = quoted_string.copy().set_name('string').set_parse_action(actions.Symbol('string'))

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
        p.one_of('^ * / + - not == != <= >= < > && ||') |
        p.Combine('MathML:' + p.one_of(' '.join(mathml_operators))))
    wrap = p.Group(
        p.Suppress('@') - adjacent(p.Word(p.nums)) + adjacent(colon) + mathml_operator
    ).set_name('MathML lambda "@" syntax').set_parse_action(actions.Wrap)

    # Turning on tracing for debugging protocols
    trace = adjacent(p.Suppress('?'))

    # The main expression grammar.  Atoms are ordered according to rough speed of detecting mis-match.
    atom = (
        array | wrap | number.copy().set_parse_action(actions.Number) | string_value |
        if_expr | null_value | default_value | lambda_expr | function_call | ident_as_var | tuple
    ).set_name('atomic expression')
    expr <<= p.infix_notation(atom, [(accessor, 1, p.opAssoc.LEFT, actions.Accessor),
                                    (view_spec, 1, p.opAssoc.LEFT, actions.View),
                                    (index, 1, p.opAssoc.LEFT, actions.Index),
                                    (trace, 1, p.opAssoc.LEFT, actions.Trace),
                                    ('^', 2, p.opAssoc.LEFT, actions.Operator),
                                    ('-', 1, p.opAssoc.RIGHT,
                                        lambda *args: actions.Operator(*args, rightAssoc=True)),
                                    (p.one_of('* /'), 2, p.opAssoc.LEFT, actions.Operator),
                                    (p.one_of('+ -'), 2, p.opAssoc.LEFT, actions.Operator),
                                    (p.Keyword('not'), 1, p.opAssoc.RIGHT,
                                     lambda *args: actions.Operator(*args, rightAssoc=True)),
                                    (p.one_of('== != <= >= < >'), 2, p.opAssoc.LEFT, actions.Operator),
                                    (p.one_of('&& ||'), 2, p.opAssoc.LEFT, actions.Operator)
                                    ])

    # Simpler expressions containing no arrays, functions, etc. Used in the model interface.
    simple_expr = p.Forward().set_name('simple expression')
    simple_if_expr = p.Group(
        make_kw('if') - simple_expr + make_kw('then') - simple_expr + make_kw('else') - simple_expr
    ).set_name('simple if-then-else').set_parse_action(actions.Piecewise)
    simple_arg_list = p.Group(optional_delimited_list(simple_expr, comma))
    simple_function_call = p.Group(ident_as_var + adjacent(oparen) - simple_arg_list +
                                   cparen).set_name('simple function call').set_parse_action(actions.FunctionCall)
    simple_expr <<= p.infix_notation(
        number.copy().set_parse_action(actions.Number) | simple_if_expr | simple_function_call | ident_as_var,
        [
            ('^', 2, p.opAssoc.LEFT, actions.Operator),
            ('-', 1, p.opAssoc.RIGHT, lambda *args: actions.Operator(*args, rightAssoc=True)),
            (p.one_of('* /'), 2, p.opAssoc.LEFT, actions.Operator),
            (p.one_of('+ -'), 2, p.opAssoc.LEFT, actions.Operator),
            (p.Keyword('not'), 1, p.opAssoc.RIGHT, lambda *args: actions.Operator(*args, rightAssoc=True)),
            (p.one_of('== != <= >= < >'), 2, p.opAssoc.LEFT, actions.Operator),
            (p.one_of('&& ||'), 2, p.opAssoc.LEFT, actions.Operator)
        ])
    simple_param_list = p.Group(optional_delimited_list(p.Group(nc_ident_as_var), comma))
    simple_lambda_expr = p.Group(make_kw('lambda') - simple_param_list + colon -
                                 simple_expr).set_name('simple lambda function').set_parse_action(actions.Lambda)

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
    simple_assign = p.Group(
        nc_ident_as_var + eq - expr).set_name('simple assignment').set_parse_action(actions.Assignment)
    simple_assign_list = p.Group(optional_delimited_list(simple_assign, nl)).set_parse_action(actions.StatementList)

    # Assertions and function returns
    assert_stmt = p.Group(make_kw('assert') - expr).set_name('assert statement').set_parse_action(actions.Assert)
    return_stmt = p.Group(
        make_kw('return') - p.DelimitedList(expr)).set_name('return statement').set_parse_action(actions.Return)

    # Full assignment, to a tuple of names or single name
    _idents = p.Group(p.DelimitedList(nc_ident_as_var)).set_parse_action(actions.MaybeTuple)
    assign_stmt = p.Group(
        ((make_kw('optional', suppress=False)("optional") + _idents) | _idents) + eq -
        p.Group(p.DelimitedList(expr)).set_parse_action(actions.MaybeTuple)
    ).set_name('assignment statement').set_parse_action(actions.Assignment)

    # Function definition
    function_defn = p.Group(make_kw('def') - nc_ident_as_var + oparen + param_list + cparen -
                            ((colon - expr) | (obrace - stmt_list + Optional(nl) + p.Suppress('}')))
                            ).set_name('function definition').set_parse_action(actions.FunctionDef)

    stmt_list << p.Group(p.DelimitedList(assert_stmt | return_stmt | function_defn | assign_stmt, nl))
    stmt_list.set_parse_action(actions.StatementList)

    # Miscellaneous constructs making up protocols
    ##############################################

    # Documentation (Markdown)
    documentation = p.Group(make_kw('documentation') - obrace - p.Regex("[^}]*") + cbrace).set_results_name("dox")

    # Namespace declarations
    ns_decl = p.Group(
        make_kw('namespace') - nc_ident("prefix") + eq + quoted_uri("uri")).set_name('namespace declaration')
    ns_decls = optional_delimited_list(ns_decl("namespace*"), nl)

    # Protocol input declarations, with default values
    inputs = (
        make_kw('inputs') - obrace - simple_assign_list + cbrace
    ).set_results_name("inputs").set_name('protocol inputs').set_parse_action(actions.Inputs)

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
            embedded_cbrace)).set_name('protocol import').set_parse_action(
                actions.Import)
    imports = optional_delimited_list(import_stmt, nl).set_results_name('imports').set_name('protocol imports')

    # Library, globals defined using post-processing language.
    # Strictly speaking returns aren't allowed, but that gets picked up later.
    library = (make_kw('library') - obrace - Optional(stmt_list) +
               cbrace).set_results_name("library").set_name('library section').set_parse_action(actions.Library)

    # Post-processing
    post_processing = (
        make_kw('post-processing') + obrace -
        optional_delimited_list(assert_stmt | return_stmt | function_defn | assign_stmt, nl) +
        cbrace
    ).set_results_name("postprocessing").set_name('post-processing section').set_parse_action(actions.PostProcessing)

    # Units definitions
    si_prefix = p.one_of('deka hecto kilo mega giga tera peta exa zetta yotta'
                        'deci centi milli micro nano pico femto atto zepto yocto')
    _num_or_expr = p.original_text_for(plain_number | (oparen + expr + cparen))
    unit_ref = p.Group(Optional(_num_or_expr)("multiplier") + Optional(si_prefix)("prefix") + nc_ident("units") +
                       Optional(p.Suppress('^') + plain_number)("exponent") +
                       Optional(p.Group(p.one_of('- +') + _num_or_expr))("offset")).set_parse_action(actions.UnitRef)
    units_def = p.Group(nc_ident + eq + p.DelimitedList(unit_ref, '.') + Optional(quoted_string)("description")
                        ).set_name('units definition').set_parse_action(actions.UnitsDef)
    units = (make_kw('units') - obrace - optional_delimited_list(units_def, nl) + cbrace
             ).set_results_name("units").set_name('units section').set_parse_action(actions.Units)

    # Model interface section
    #########################
    units_ref = make_kw('units') - nc_ident

    # Setting the units for the independent variable
    set_time_units = (make_kw('independent') - make_kw('var') - units_ref("units")).set_parse_action(actions.SetTimeUnits)

    # Input variables, with optional units and initial value
    input_variable = p.Group(
        make_kw('input') -
        c_ident('name') +
        Optional(units_ref)('units') +
        Optional(eq + plain_number)('initial_value')
    ).set_name('input variable declaration').set_parse_action(actions.InputVariable)

    # Model outputs of interest, with optional units
    output_variable = p.Group(
        make_kw('output') -
        c_ident("name") +
        Optional(units_ref("units"))
    ).set_name('output variable declaration').set_parse_action(actions.OutputVariable)

    # Model variables (inputs, outputs, or just used in equations) that are allowed to be missing
    locator = p.Empty().leave_whitespace().set_parse_action(lambda s, loc, tokens: loc)
    var_default = make_kw('default') - locator("default_start") + simple_expr("default")
    optional_variable = p.Group(
        make_kw('optional') - c_ident("name") + Optional(var_default) + locator("default_end")
    ).set_name('optional variable declaration').set_parse_action(actions.OptionalVariable)

    # New variables added to the model, with optional initial value
    new_variable = p.Group(
        make_kw('var') -
        nc_ident("name") +
        units_ref("units") +
        Optional(
            eq +
            plain_number)("initial_value")
    ).set_name('new variable declaration').set_parse_action(actions.DeclareVariable)

    # Adding or replacing equations in the model
    clamp_variable = p.Group(
        make_kw('clamp') - ident_as_var + Optional(make_kw('to') - simple_expr)
    ).set_name('clamp variable declaration').set_parse_action(actions.ClampVariable)
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
        cparen).set_name('interpolate').set_parse_action(
        actions.Interpolate)
    model_equation = p.Group(
        make_kw('define') - (
            p.Group(make_kw('diff') + adjacent(oparen) - ident_as_var + p.Suppress(';') + ident_as_var + cparen)
            | ident_as_var
        ) + eq + (interpolate | simple_expr)
    ).set_name('model equation definition').set_parse_action(actions.ModelEquation)

    # Units conversion rules
    units_conversion = p.Group(
        make_kw('convert') - nc_ident("actualDimensions") +
        make_kw('to') + nc_ident("desiredDimensions") +
        make_kw('by') - simple_lambda_expr
    ).set_name('units conversion rule').set_parse_action(actions.UnitsConversion)

    model_interface = p.Group(
        make_kw('model') - make_kw('interface') - obrace - Optional(set_time_units - nl) +
        optional_delimited_list((
            input_variable | output_variable | optional_variable | new_variable | clamp_variable | model_equation
            | units_conversion
        ), nl) + cbrace
    ).set_results_name("model_interface").set_name('model interface section').set_parse_action(actions.ModelInterface)

    # Simulation definitions
    ########################

    # Ranges
    uniform_range = make_kw('uniform') + numeric_range
    vector_range = make_kw('vector') + expr
    while_range = make_kw('while') + expr
    range = p.Group(make_kw('range') + nc_ident("name") + units_ref("units") +
                    (uniform_range("uniform") | vector_range("vector") | while_range("while"))
                    ).set_name('range').set_parse_action(actions.Range)

    # Modifiers
    modifier_when = make_kw('at') - (make_kw('start', False) |
                                     (make_kw('each', False) - make_kw('loop')) |
                                     make_kw('end', False)).set_parse_action(actions.ModifierWhen)
    set_variable = make_kw('set') - ident + eq + expr
    save_state = make_kw('save') - make_kw('as') - nc_ident
    reset_state = make_kw('reset') - Optional(make_kw('to') + nc_ident)
    modifier = p.Group(modifier_when + p.Group(set_variable("set") | save_state("save") | reset_state("reset"))
                       ).set_name('modifier').set_parse_action(actions.Modifier)
    modifiers = p.Group(make_kw('modifiers') + obrace - optional_delimited_list(modifier, nl) + cbrace
                        ).set_name('modifiers').set_parse_action(actions.Modifiers)

    # The simulations themselves
    simulation = p.Forward().set_name('simulation')
    _select_output = p.Group(
        make_kw('select') - Optional(make_kw('optional', suppress=False)) - make_kw('output') - nc_ident
    ).set_name('SelectOutput')
    nested_protocol = p.Group(
        make_kw('protocol') - quoted_uri + obrace + simple_assign_list + Optional(nl) +
        optional_delimited_list(_select_output, nl) + cbrace + Optional('?')
    ).set_name('nested protocol').set_parse_action(actions.NestedProtocol)
    timecourse_sim = p.Group(
        make_kw('timecourse') - obrace - range + Optional(nl + modifiers) + cbrace
    ).set_name('timecourse simulation').set_parse_action(actions.TimecourseSimulation)
    nested_sim = p.Group(
        make_kw('nested') - obrace - range + nl + Optional(modifiers) +
        p.Group(make_kw('nests') + (simulation | nested_protocol | ident)) + cbrace
    ).set_name('nested simulation').set_parse_action(actions.NestedSimulation)
    one_step_sim = p.Group(
        make_kw('oneStep') - Optional(p.original_text_for(expr))("step") +
        Optional(obrace - modifiers + cbrace)("modifiers")
    ).set_parse_action(actions.OneStepSimulation)
    simulation << p.Group(make_kw('simulation') - Optional(nc_ident + eq, default='') +
                          (timecourse_sim | nested_sim | one_step_sim) -
                          Optional('?' + nl)).set_parse_action(actions.Simulation)

    tasks = p.Group(make_kw('tasks') + obrace - p.ZeroOrMore(simulation) +
                    cbrace).set_results_name("tasks").set_name('tasks section').set_parse_action(actions.Tasks)

    # Output specifications
    #######################

    output_desc = Optional(quoted_string)("description")
    output_spec = p.Group(
        Optional(make_kw('optional', suppress=False))("optional") +
        nc_ident("name") +
        ((units_ref("units") + output_desc) | (eq + ident("ref") + Optional(units_ref)("units") + output_desc))
    ).set_name('protocol output specification').set_parse_action(actions.Output)
    outputs = p.Group(make_kw('outputs') + obrace - optional_delimited_list(output_spec, nl) +
                      cbrace).set_results_name("outputs").set_name('outputs section').set_parse_action(actions.Outputs)

    # Plot specifications
    #####################

    plot_curve = p.Group(
        p.DelimitedList(nc_ident, ',') +
        make_kw('against') - nc_ident +
        Optional(make_kw('key') - nc_ident("key"))
    ).set_name('Curve')
    plot_using = (make_kw('using') - (make_kw('lines', suppress=False) |
                                      make_kw('points', suppress=False) |
                                      make_kw('linespoints', suppress=False)))("using")
    plot_spec = p.Group(
        make_kw('plot') - quoted_string + Optional(plot_using) - obrace +
        plot_curve + p.ZeroOrMore(nl + plot_curve) + cbrace
    ).set_name('plot specification').set_parse_action(actions.Plot)
    plots = p.Group(make_kw('plots') + obrace - p.ZeroOrMore(plot_spec) +
                    cbrace).set_results_name("plots").set_name('plots section').set_parse_action(actions.Plots)

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
        ]))).set_name('Protocol').set_parse_action(actions.Protocol)

    # Caching of parsed files
    # This maps source file names to a tuple (date_read, result)
    _cache = {}

    def __init__(self):
        """Initialise the parser."""
        # We just store the original stack limit here, so we can increase
        # it for the lifetime of this object if needed for parsing, on the
        # basis that if one expression needs to, several are likely to.
        self._stack_depth_factor = 1
        self._original_stack_limit = sys.getrecursionlimit()

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

    def parse_file(self, source_file):
        """Main entry point to parse a protocol file.

        :param source_file: path to the file to parse
        :return: a :class:`fc.parsing.actions.Protocol` object, containing parsed information about the protocol
        """
        return self.try_parse(self.protocol.parseFile, source_file, parse_all=True)[0]

    def try_parse(self, callable, source_file, *args, **kwargs):
        """
        Try calling the given parse command, increasing the stack depth limit if needed.
        """
        # Try returning in-memory cached file
        now = time.time()
        try:
            date_read, r = self._cache[source_file]
            if os.path.getmtime(source_file) < date_read:
                print('Using mem-cached protocol for ' + source_file)
                return r
        except KeyError:
            pass

        # Try returning disk-cached file
        cache_file = os.path.split(source_file)
        cache_file = os.path.join(cache_file[0], '.' + cache_file[1] + '.cache')
        if os.path.exists(cache_file):
            if os.path.getmtime(source_file) < os.path.getmtime(cache_file):
                with open(cache_file, 'rb') as f:
                    try:
                        print('Reading disk-cached protocol from ' + cache_file)
                        r = pickle.load(f)

                        # Store parse result in memory cache and return
                        self._cache[source_file] = (now, r)
                        return r
                    except Exception:
                        pass
            try:
                os.remove(cache_file)
            except OSError:
                pass

        # Read file
        r = None  # Result
        with actions.set_reference_source(source_file):
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

        # Store parse result in memory cache
        self._cache[source_file] = (now, r)

        # Store parse result in disk cache
        with open(cache_file, 'wb') as f:
            print('Caching protocol to ' + cache_file)
            pickle.dump(r, f)

        # Return
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
