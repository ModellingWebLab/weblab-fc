import pytest
import sys

# Import the module to test
# The default for this module now is to assume the Python implementation,
# so we have to override that!
sys._fc_csp_no_pyimpl = True
import fc.parsing.CompactSyntaxParser as CSP  # noqa: E402

csp = CSP.CompactSyntaxParser
# An end-of-string match that doesn't allow trailing whitespace
strict_string_end = CSP.p.StringEnd().leaveWhitespace()


def check_parse_results(actual, expected):
    """Compare parse results to expected strings.

    The expected results may be given as a (nested) list or a dictionary, depending
    on whether you want to compare against matched tokens by order or results name.
    """
    def check_result(actual, expected):
        if isinstance(expected, str):
            assert actual == expected
        else:
            check_parse_results(actual, expected)

    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for i, result in enumerate(expected):
            check_result(actual[i], result)
    elif isinstance(expected, dict):
        assert len(actual) == len(expected)
        for key, value in expected.items():
            check_result(actual[key], value)


def assert_parses(grammar, input, results):
    """Utility method to test that a given grammar parses an input as expected."""
    actual_results = grammar.parseString(input, parseAll=True)
    check_parse_results(actual_results, results)


def assert_does_not_parse(grammar, input):
    """Utility method to test that a given grammar fails to parse an input."""
    strict_grammar = grammar + strict_string_end
    with pytest.raises(CSP.p.ParseBaseException):
        strict_grammar.parseString(input)


def test_parsing_identifiers():
    assert_does_not_parse(csp.nc_ident, 'abc:def')
    assert_does_not_parse(csp.nc_ident, '123')

    assert_parses(csp.ident_as_var, 'abc', ['abc'])
    assert_parses(csp.ident, 'abc_def', ['abc_def'])
    assert_parses(csp.ident, 'abc123', ['abc123'])
    assert_parses(csp.ident, '_abc', ['_abc'])
    assert_parses(csp.ident_as_var, 'abc:def', ['abc:def'])
    assert_does_not_parse(csp.ident, '123')


def test_parsing_numbers():
    assert_parses(csp.number, '123', ['123'])
    assert_parses(csp.number, '1.23', ['1.23'])
    assert_parses(csp.number, '-12.3', ['-12.3'])
    assert_parses(csp.number, '12e3', ['12e3'])
    assert_parses(csp.number, '-1.2e3', ['-1.2e3'])
    assert_parses(csp.number, '1.2e3', ['1.2e3'])
    assert_parses(csp.number, '1.2e+3', ['1.2e+3'])
    assert_parses(csp.number, '12e-3', ['12e-3'])
    assert_parses(csp.number, '-1e-2', ['-1e-2'])
    assert_parses(csp.number, '0.1', ['0.1'])
    assert_does_not_parse(csp.number, '.123')
    assert_does_not_parse(csp.number, '123.')
    assert_does_not_parse(csp.number, '+123')
    assert_does_not_parse(csp.number, '1E3')


def test_parsing_numbers_with_units():
    assert_parses(csp.number, '123 :: dimensionless', ['123', 'dimensionless'])
    assert_parses(csp.number, '-4e5::mA', ['-4e5', 'mA'])
    assert_parses(csp.expr, '123 :: dimensionless', ['123'])
    assert_parses(csp.expr, '4.6e5::mA', ['4.6e5'])
    assert_does_not_parse(csp.number, '123 :: ')
    assert_does_not_parse(csp.number, '123 :: prefix:units')


def test_parsing_comments():
    assert_parses(csp.comment, '# blah blah', [])
    assert_does_not_parse(csp.comment, '# blah blah\n')


def test_parsing_simple_expressions():
    assert_parses(csp.expr, '1', ['1'])
    assert_parses(csp.expr, '(A)', ['A'])
    assert_parses(
        csp.expr,
        '1 - 2 + 3 * 4 / 5 ^ 6',
        [['1', '-', '2', '+', ['3', '*', '4', '/', ['5', '^', '6']]]])
    assert_parses(csp.expr, '-(a + -3)', [['-', ['a', '+', ['-', '3']]]])
    assert_parses(csp.expr, '1 ^ 2 ^ 3', [['1', '^', '2', '^', '3']])
    assert_parses(csp.expr, 'not 1 < 2', [[['not', '1'], '<', '2']])
    assert_parses(csp.expr, 'not (1 < 2)', [['not', ['1', '<', '2']]])
    assert_parses(csp.expr, 'not not my:var', [['not', ['not', 'my:var']]])
    assert_parses(
        csp.expr,
        '1 < 2 <= (3 >= 4) > 5 == 6 != 7',
        [['1', '<', '2', '<=', ['3', '>=', '4'], '>', '5', '==', '6', '!=', '7']])
    assert_parses(csp.expr, '1 < 2 + 4 > 3', [['1', '<', ['2', '+', '4'], '>', '3']])
    assert_parses(csp.expr, '1 < 2 && 4 > 3', [[['1', '<', '2'], '&&', ['4', '>', '3']]])
    assert_parses(csp.expr, '0 || 1 && 1', [['0', '||', '1', '&&', '1']])
    assert_parses(csp.expr, 'A + B || C * D', [[['A', '+', 'B'], '||', ['C', '*', 'D']]])
    assert_parses(csp.expr, 'if 1 then 2 else 3', [['1', '2', '3']])
    assert_parses(csp.expr, 'if 1 < 2 then 3 + 4 else 5 * 6',
                  [[['1', '<', '2'], ['3', '+', '4'], ['5', '*', '6']]])


def test_parsing_trace():
    assert_parses(csp.expr, '1?', [['1']])
    assert_parses(csp.expr, 'var?', [['var']])
    assert_parses(csp.expr, '(1 + a)?', [[['1', '+', 'a']]])
    assert_parses(csp.expr, '1 + a?', [['1', '+', ['a']]])

    action = csp.expr.parseString('var?', parseAll=True)
    assert action[0].expr().trace

    action = csp.expr.parseString('var', parseAll=True)
    assert not action[0].expr().trace


def test_parsing_multi_line_expressions():
    assert_parses(csp.expr, '(1 + 2) * 3', [[['1', '+', '2'], '*', '3']])
    assert_parses(csp.expr, '((1 + 2)\\\n * 3)', [[['1', '+', '2'], '*', '3']])
    assert_parses(csp.expr, '(1 + 2)\\\n * 3', [[['1', '+', '2'], '*', '3']])
    assert_parses(csp.expr, '((1 + 2)\n  * 3)', [[['1', '+', '2'], '*', '3']])
    assert_parses(csp.expr, '((1 + 2)#embedded comment\n  * 3)', [[['1', '+', '2'], '*', '3']])
    # This one doesn't match Python behaviour, but it's hard to stop it parsing while allowing
    # the above to work.
    assert_parses(csp.expr, '(1 + 2)\n * 3', [[['1', '+', '2'], '*', '3']])


def test_parsing_simple_assignments():
    assert_parses(csp.simple_assign, 'var = value', [['var', 'value']])
    assert_parses(csp.simple_assign, 'var = pre:value', [['var', 'pre:value']])
    assert_does_not_parse(csp.simple_assign, 'pre:var = value')
    assert_parses(csp.simple_assign, 'var = 1 + 2', [['var', ['1', '+', '2']]])
    assert_parses(csp.simple_assign_list, 'v1 = 1\nv2=2', [[['v1', '1'], ['v2', '2']]])
    assert_parses(csp.simple_assign_list, '', [[]])


def test_parsing_namespaces():
    assert_parses(csp.ns_decl, 'namespace prefix = "urn:test"', [{'prefix': 'prefix', 'uri': 'urn:test'}])
    assert_parses(csp.ns_decl, "namespace prefix='urn:test'", [{'prefix': 'prefix', 'uri': 'urn:test'}])
    assert_parses(csp.ns_decls, 'namespace n1="urn:t1"#test ns\nnamespace n2 = "urn:t2"',
                  [{'prefix': 'n1', 'uri': 'urn:t1'}, {'prefix': 'n2', 'uri': 'urn:t2'}])
    assert_parses(csp.ns_decls, '', [])
    assert_does_not_parse(csp.ns_decls, 'namespace n="uri"\n')


def test_parsing_inputs():
    mls = """inputs {
    label_duration = 300  # How long to label for
    unlabel_time = 600000 # When to stop labelling
    use_tapasin = 1       # Whether to include Tapasin
}
"""
    out = [[[['label_duration', '300'], ['unlabel_time', '600000'], ['use_tapasin', '1']]]]
    assert_parses(csp.inputs, mls, out)
    assert_parses(csp.inputs, 'inputs {}', [[[]]])
    assert_parses(csp.inputs, 'inputs\n{\n}\n', [[[]]])
    assert_parses(csp.inputs, 'inputs{X=1}', [[[['X', '1']]]])


def test_parsing_imports():
    assert_parses(
        csp.import_stmt,
        'import std = "../library/BasicLibrary.xml"',
        [['std', '../library/BasicLibrary.xml']]
    )
    assert_parses(
        csp.import_stmt,
        "import 'TestS1S2.xml'",
        [['', 'TestS1S2.xml']])
    assert_parses(
        csp.imports,
        'import l1="file1"#blah\nimport "file2"',
        [['l1', 'file1'], ['', 'file2']])
    assert_parses(csp.imports, '', [])
    assert_does_not_parse(csp.imports, 'import "file"\n')


def test_parsing_imports_with_set_input():
    mls = """import "S1S2.txt" {
    steady_state_beats = 10
    timecourse_duration = 2000
}"""
    assert_parses(
        csp.import_stmt,
        mls,
        [['', 'S1S2.txt', [['steady_state_beats', '10'], ['timecourse_duration', '2000']]]]
    )
    assert_parses(csp.import_stmt, 'import "file.txt" { }', [['', 'file.txt', []]])
    assert_does_not_parse(csp.import_stmt, 'import "file.txt" { } \n')


def test_parsing_model_interface():
    assert_parses(csp.set_time_units, 'independent var units u', [['u']])
    assert_parses(csp.input_variable, 'input test:var units u = 1.2', [['test:var', 'u', '1.2']])
    assert_parses(csp.input_variable, 'input test:var units u', [['test:var', 'u']])
    assert_parses(csp.input_variable, 'input test:var = -1e2', [['test:var', '-1e2']])
    assert_parses(csp.input_variable, 'input test:var', [['test:var']])
    assert_does_not_parse(csp.input_variable, 'input no_prefix')

    assert_parses(csp.output_variable, 'output test:var', [['test:var']])
    assert_parses(csp.output_variable, 'output test:var units uname', [['test:var', 'uname']])
    assert_does_not_parse(csp.output_variable, 'output no_prefix')

    assert_parses(
        csp.optional_variable, 'optional prefix:name', [['prefix:name', 20]])
    assert_parses(
        csp.optional_variable,
        'optional p:n default p:v + local * 2 :: u',
        [['p:n', 12, ['p:v', '+', ['local', '*', '2']], 41]])
    assert_does_not_parse(csp.optional_variable, 'optional no_prefix')

    assert_parses(csp.new_variable, 'var varname units uname = 0', [['varname', 'uname', '0']])
    assert_parses(csp.new_variable, 'var varname units uname', [['varname', 'uname']])
    assert_does_not_parse(csp.new_variable, 'var prefix:varname units uname = 0')
    assert_does_not_parse(csp.new_variable, 'var varname = 0')
    assert_does_not_parse(csp.new_variable, 'var varname')

    assert_parses(csp.clamp_variable, 'clamp p:v', [['p:v']])
    assert_parses(csp.clamp_variable, 'clamp p:v to 0 :: U', [['p:v', '0']])

    assert_parses(
        csp.model_equation,
        'define local_var = 1::U + model:var',
        [['local_var', ['1', '+', 'model:var']]])
    assert_parses(
        csp.model_equation,
        'define model:var = 2.5 :: units / local_var',
        [['model:var', ['2.5', '/', 'local_var']]])
    assert_parses(
        csp.model_equation,
        'define diff(oxmeta:membrane_voltage; oxmeta:time) = 1 :: mV_per_ms',
        [[['oxmeta:membrane_voltage', 'oxmeta:time'], '1']])

    assert_parses(
        csp.units_conversion,
        'convert uname1 to uname2 by lambda u: u / model:var',
        [['uname1', 'uname2', [[['u']], ['u', '/', 'model:var']]]])

    mls = """model interface {  # Comments can go here
    independent var units t

    input test:v1 = 0  # a comment
    var early_local units u = 12.34 # order of elements doesn't matter
    input test:v2 units u
    output test:time
    # comments are always ignored
    output test:v3 units u

    optional test:opt

    var local units dimensionless = 5
    define test:v3 = test:v2 * local
    convert u1 to u2 by lambda u: u * test:v3
}"""
    out = [[
        ['t'],
        ['test:v1', '0'],
        ['early_local', 'u', '12.34'],
        ['test:v2', 'u'],
        ['test:time'],
        ['test:v3', 'u'],
        ['test:opt', 315],
        ['local', 'dimensionless', '5'],
        ['test:v3', ['test:v2', '*', 'local']],
        ['u1', 'u2', [[['u']], ['u', '*', 'test:v3']]]
    ]]
    assert_parses(csp.model_interface, mls, out)
    assert_parses(csp.model_interface, 'model interface {}', [[]])
    assert_parses(csp.model_interface, 'model interface#comment\n{output test:time\n}', [[['test:time']]])
    assert_parses(csp.model_interface, 'model interface {output test:time }', [[['test:time']]])


def test_parsing_uniform_range():
    assert_parses(csp.range, 'range time units ms uniform 0:1:1000', [['time', 'ms', ['0', '1', '1000']]])
    assert_parses(csp.range, 'range time units ms uniform 0:1000', [['time', 'ms', ['0', '1000']]])
    assert_parses(csp.range, 'range t units s uniform 0:end', [['t', 's', ['0', 'end']]])

    # Spaces or brackets are required in this case to avoid 'start:step:end' parsing as an ident
    assert_parses(
        csp.range,
        'range t units s uniform start : step : end',
        [['t', 's', ['start', 'step', 'end']]])
    assert_parses(csp.range, 'range t units s uniform start:(step):end', [['t', 's', ['start', 'step', 'end']]])
    assert_does_not_parse(csp.range, 'range t units s uniform start:step:end')


def test_parsing_vector_range():
    assert_parses(
        csp.range,
        'range run units dimensionless vector [1, 2, 3, 4]',
        [['run', 'dimensionless', ['1', '2', '3', '4']]])


def test_parsing_while_range():
    assert_parses(
        csp.range, 'range rpt units dimensionless while rpt < 5',
        [['rpt', 'dimensionless', ['rpt', '<', '5']]])


def test_parsing_modifiers():
    assert_parses(csp.modifier_when, 'at start', ['start'])
    assert_parses(csp.modifier_when, 'at each loop', ['each'])
    assert_parses(csp.modifier_when, 'at end', ['end'])

    assert_parses(csp.set_variable, 'set model:V = 5.0', ['model:V', '5.0'])
    assert_parses(csp.set_variable, 'set model:t = time + 10.0', ['model:t', ['time', '+', '10.0']])

    assert_parses(csp.save_state, 'save as state_name', ['state_name'])
    assert_does_not_parse(csp.save_state, 'save as state:name')

    assert_parses(csp.reset_state, 'reset', [])
    assert_parses(csp.reset_state, 'reset to state_name', ['state_name'])
    assert_does_not_parse(csp.reset_state, 'reset to state:name')

    mls = """modifiers {
        # Multiple
        # comment lines
        # are OK
        at start reset
        at each loop set model:input = loopVariable

        # Blank lines OK too
        at end save as savedState
} # Trailing comments are fine"""
    assert_parses(
        csp.modifiers, mls,
        [[['start', []], ['each', ['model:input', 'loopVariable']], ['end', ['savedState']]]])
    assert_parses(csp.modifiers, 'modifiers {at start reset}', [[['start', []]]])


def test_parsing_timecourse_simulations():
    assert_parses(
        csp.simulation,
        'simulation sim = timecourse { range time units ms uniform 1:10 }',
        [['sim', [['time', 'ms', ['1', '10']]]]])
    assert_parses(
        csp.simulation,
        'simulation timecourse #c\n#c\n{\n range time units ms uniform 1:10\n\n }#c',
        [['', [['time', 'ms', ['1', '10']]]]])

    mls = """simulation sim = timecourse {
range time units U while time < 100
modifiers { at end save as prelim }
}"""
    assert_parses(csp.simulation, mls,
                  [['sim', [['time', 'U', ['time', '<', '100']], [['end', ['prelim']]]]]])
    assert_does_not_parse(csp.simulation, 'simulation sim = timecourse {}')


def test_parsing_one_step_simulations():
    assert_parses(csp.simulation, 'simulation oneStep', [['', []]])
    assert_parses(csp.simulation, 'simulation sim = oneStep', [['sim', []]])
    assert_parses(csp.simulation, 'simulation oneStep 1.0', [['', ['1.0']]])
    assert_parses(csp.simulation, 'simulation sim = oneStep step', [['sim', ['step']]])
    assert_parses(
        csp.simulation,
        'simulation oneStep { modifiers { at start set a = 1 } }',
        [['', [[['start', ['a', '1']]]]]])


def test_parsing_nested_simulations():
    assert_parses(
        csp.simulation,
        'simulation rpt = nested { range run units U while not rpt:result\n nests sim }',
        [['rpt', [['run', 'U', ['not', 'rpt:result']], ['sim']]]])

    assert_parses(
        csp.simulation,
        """simulation nested {
range R units U uniform 3:5
modifiers { at each loop reset to prelim }
nests sim
}""",
        [['', [['R', 'U', ['3', '5']], [['each', ['prelim']]], ['sim']]]])

    assert_parses(
        csp.simulation,
        """simulation nested { range R units U uniform 1:2
nests simulation timecourse { range t units u uniform 1:100 } }""",
        [['', [['R', 'U', ['1', '2']], [['', [['t', 'u', ['1', '100']]]]]]]])

    assert_does_not_parse(
        csp.simulation,
        'simulation rpt = nested { range run units U while 1 }')


def test_parsing_nested_protocol():
    assert_parses(
        csp.simulation,
        'simulation nested { range iter units D vector [0, 1]\n nests protocol "P" { } }',
        [['', [['iter', 'D', ['0', '1']], [['P', []]]]]])

    assert_parses(
        csp.simulation,
        """simulation nested {
    range iter units D vector [0, 1]
    nests protocol "../proto.xml" {
        input1 = 1
        input2 = 2
    }
}""",
        [['', [['iter', 'D', ['0', '1']], [['../proto.xml', [['input1', '1'], ['input2', '2']]]]]]])

    assert_parses(
        csp.simulation,
        """simulation nested {
    range iter units D vector [0, 1]
    nests protocol "proto.txt" {
        input = iter
        select output oname
        select optional output opt
    }
}""",
        [['', [['iter', 'D', ['0', '1']], [['proto.txt', [['input', 'iter']], ['oname'], ['optional', 'opt']]]]]])

    # Tracing a nested protocol
    assert_parses(
        csp.simulation,
        'simulation nested { range iter units D vector [0, 1]\n nests protocol "P" { }? }',
        [['', [['iter', 'D', ['0', '1']], [['P', []]]]]])


def test_parsing_tasks():
    assert_parses(
        csp.tasks,
        """tasks {
    simulation timecourse { range time units second uniform 1:1000 }
    simulation main = nested { range n units dimensionless vector [i*2 for i in 1:4]
                               nests inner }
}
""",
        [[
            ['', [['time', 'second', ['1', '1000']]]],
            [
                'main',
                [
                    ['n', 'dimensionless', [['i', '*', '2'], ['i', ['1', '4']]]],
                    ['inner']
                ]
            ]
        ]])


def test_parsing_output_specifications():
    assert_parses(csp.output_spec, 'name = model:var "Description"', [['name', 'model:var', 'Description']])
    assert_parses(csp.output_spec, r'name = ref:var units U "Description \"quotes\""',
                  [['name', 'ref:var', 'U', 'Description "quotes"']])
    assert_parses(csp.output_spec, "name = ref:var units U 'Description \\'quotes\\' \"too\"'",
                  [['name', 'ref:var', 'U', 'Description \'quotes\' "too"']])
    assert_parses(csp.output_spec, 'varname units UU', [['varname', 'UU']])
    assert_parses(csp.output_spec, 'varname units UU "desc"', [['varname', 'UU', 'desc']])
    assert_parses(csp.output_spec, 'optional varname units UU', [['optional', 'varname', 'UU']])
    assert_parses(csp.output_spec, 'optional varname = ref:var', [['optional', 'varname', 'ref:var']])
    assert_does_not_parse(csp.output_spec, 'varname_no_units')

    assert_parses(csp.outputs, """outputs #cccc
{ #cdc
        n1 = n2 units u1
        n3 = p:m 'd1'
        n4 units u2 "d2"
        optional n5 units u3
} #cpc
""", [[['n1', 'n2', 'u1'], ['n3', 'p:m', 'd1'], ['n4', 'u2', 'd2'], ['optional', 'n5', 'u3']]])
    assert_parses(csp.outputs, "outputs {}", [[]])


def test_parsing_plot_specifications():
    assert_parses(csp.plot_curve, 'y against x', [['y', 'x']])
    assert_parses(csp.plot_curve, 'y, y2 against x', [['y', 'y2', 'x']])
    assert_does_not_parse(csp.plot_curve, 'm:y against x')
    assert_does_not_parse(csp.plot_curve, 'y against m:x')
    assert_parses(
        csp.plot_spec,
        'plot "A title\'s good" { y1, y2 against x1\n y3 against x2 }',
        [["A title's good", ['y1', 'y2', 'x1'], ['y3', 'x2']]])
    assert_parses(csp.plot_spec, 'plot "Keys" { y against x key k }', [["Keys", ['y', 'x', 'k']]])
    assert_does_not_parse(csp.plot_spec, 'plot "only title" {}')
    assert_does_not_parse(csp.plot_spec, 'plot "only title"')

    assert_parses(csp.plots, """plots { plot "t1" { v1 against v2 key vk }
    plot "t1" { v3, v4 against v5 }
}""", [[['t1', ['v1', 'v2', 'vk']], ['t1', ['v3', 'v4', 'v5']]]])
    assert_parses(csp.plots, 'plots {}', [[]])


def test_parsing_function_calls():
    assert_parses(csp.function_call, 'noargs()', [['noargs', []]])
    assert_parses(csp.function_call, 'swap(a, b)', [['swap', ['a', 'b']]])
    assert_parses(csp.function_call, 'double(33)', [['double', ['33']]])
    assert_parses(csp.function_call, 'double(a + b)', [['double', [['a', '+', 'b']]]])
    assert_parses(csp.function_call, 'std:max(A)', [['std:max', ['A']]])
    assert_does_not_parse(csp.function_call, 'spaced (param)')
    assert_parses(csp.expr, 'func(a,b, 3)', [['func', ['a', 'b', '3']]])


def test_parsing_mathml_operators():
    # MathML that doesn't have a special operator is represented as a normal function call,
    # with the 'magic' MathML: prefix.
    assert len(csp.mathml_operators) == 12 + 3 * 8
    for trigbase in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']:
        assert trigbase in csp.mathml_operators
        assert trigbase + 'h' in csp.mathml_operators
        assert 'arc' + trigbase in csp.mathml_operators
        assert 'arc' + trigbase + 'h' in csp.mathml_operators
    for op in 'quotient rem max min root xor abs floor ceiling exp ln log'.split():
        assert op in csp.mathml_operators
    assert_parses(csp.expr, 'MathML:exp(MathML:floor(MathML:exponentiale))',
                  [['MathML:exp', [['MathML:floor', ['MathML:exponentiale']]]]])


def test_parsing_assign_statements():
    assert_parses(csp.assign_stmt, 'var = value', [[['var'], ['value']]])
    assert_parses(csp.assign_stmt, 'var = pre:value', [[['var'], ['pre:value']]])
    assert_does_not_parse(csp.assign_stmt, 'pre:var = value')
    assert_parses(csp.assign_stmt, 'var = 1 + 2', [[['var'], [['1', '+', '2']]]])

    assert_parses(csp.assign_stmt, 'a, b = tuple', [[['a', 'b'], ['tuple']]])
    assert_parses(csp.assign_stmt, 'a, b = b, a', [[['a', 'b'], ['b', 'a']]])
    assert_parses(csp.assign_stmt, 'a, b = (b, a)', [[['a', 'b'], [['b', 'a']]]])
    assert_does_not_parse(csp.assign_stmt, 'p:a, p:b = e')
    assert_does_not_parse(csp.assign_stmt, '')


def test_parsing_optional_assignments():
    assert_parses(
        csp.assign_stmt, 'optional var = value', [[['var'], ['value']]])

    # The following should parse as a non-optional assignment!
    assert_parses(
        csp.assign_stmt, 'optional = expr', [[['optional'], ['expr']]])


def test_parsing_return_statements():
    assert_parses(csp.return_stmt, 'return 2 * a', [[['2', '*', 'a']]])
    assert_parses(csp.return_stmt, 'return (3 - 4)', [[['3', '-', '4']]])
    assert_parses(csp.return_stmt, 'return a, b', [['a', 'b']])
    assert_parses(csp.return_stmt, 'return a + 1, b - 1', [[['a', '+', '1'], ['b', '-', '1']]])
    assert_parses(csp.return_stmt, 'return (a, b)', [[['a', 'b']]])


def test_parsing_assert_statements():
    assert_parses(csp.assert_stmt, 'assert a + b', [[['a', '+', 'b']]])
    assert_parses(csp.assert_stmt, 'assert (a + b)', [[['a', '+', 'b']]])
    assert_parses(csp.assert_stmt, 'assert 1', [['1']])


def test_parsing_statement_lists():
    assert_parses(csp.stmt_list, "b=-a\nassert 1", [[[['b'], [['-', 'a']]], ['1']]])

    mls = """assert a < 0 # comments are ok

# as are blank lines
b = -a
assert b > 0
c, d = a * 2, b + 1
return c, d"""
    assert_parses(
        csp.stmt_list,
        mls,
        [[
            [['a', '<', '0']],
            [['b'], [['-', 'a']]],
            [['b', '>', '0']],
            [['c', 'd'], [['a', '*', '2'], ['b', '+', '1']]],
            ['c', 'd']
        ]])
    assert_does_not_parse(csp.stmt_list, '')


def test_parsing_lambda_expressions():
    assert_parses(csp.lambda_expr, 'lambda a: a + 1', [[[['a']], ['a', '+', '1']]])
    assert_parses(csp.lambda_expr, 'lambda a, b: a + b', [[[['a'], ['b']], ['a', '+', 'b']]])
    assert_parses(csp.lambda_expr, 'lambda a, b=2: a - b', [[[['a'], ['b', '2']], ['a', '-', 'b']]])
    assert_parses(csp.expr, 'lambda a=c, b: a * b', [[[['a', 'c'], ['b']], ['a', '*', 'b']]])
    assert_parses(csp.lambda_expr, 'lambda a=p:c, b: a * b', [[[['a', 'p:c'], ['b']], ['a', '*', 'b']]])
    assert_does_not_parse(csp.lambda_expr, 'lambda p:a: 5')

    mls = """lambda a, b {
assert a > b
c = a - b
return c
}
"""
    assert_parses(
        csp.lambda_expr,
        mls,
        [[
            [['a'], ['b']],
            [[['a', '>', 'b']], [['c'], [['a', '-', 'b']]], ['c']]
        ]])
    assert_parses(csp.expr, "lambda a, b { return b, a }", [[[['a'], ['b']], [['b', 'a']]]])
    assert_parses(csp.expr, "lambda a { return a }", [[[['a']], [['a']]]])
    assert_parses(csp.expr, 'lambda { return 1 }', [[[], [['1']]]])
    assert_parses(csp.expr, 'lambda: 1', [[[], '1']])


def test_parsing_function_definitions():
    assert_parses(
        csp.function_defn,
        'def double(a)\n {\n return a * 2\n }',
        [['double', [['a']], [[['a', '*', '2']]]]])

    assert_parses(
        csp.function_defn,
        'def double(a): a * 2',
        [['double', [['a']], ['a', '*', '2']]])

    # A function definition is just sugar for an assignment of a lambda expression
    assert_parses(
        csp.stmt_list,
        'def double(a) {\n    return a * 2}',
        [[['double', [['a']], [[['a', '*', '2']]]]]])
    assert_parses(
        csp.function_defn, 'def noargs(): 1', [['noargs', [], '1']])


def test_parsing_nested_functions():
    assert_parses(
        csp.function_defn,
        """def outer()
{
    def inner1(a): a/2
    inner2 = lambda { return 5 }
    def inner3(b) {
        return b*2
    }
    return inner1(1) + inner2() + inner3(2)
}""",
        [[
            'outer',
            [],
            [
                ['inner1', [['a']], ['a', '/', '2']],
                [['inner2'], [[[], [['5']]]]],
                ['inner3', [['b']], [[['b', '*', '2']]]],
                [[['inner1', ['1']], '+', ['inner2', []], '+', ['inner3', ['2']]]]
            ]
        ]])


def test_parsing_tuples():
    assert_parses(csp.tuple, '(1,2)', [['1', '2']])
    assert_parses(csp.tuple, '(1+a,2*b)', [[['1', '+', 'a'], ['2', '*', 'b']]])
    assert_parses(csp.tuple, '(singleton,)', [['singleton']])
    assert_does_not_parse(csp.tuple, '(1)')  # You need a Python-style comma as above
    assert_parses(csp.expr, '(1,2)', [['1', '2']])
    assert_parses(csp.expr, '(1,a,3,c)', [['1', 'a', '3', 'c']])
    assert_parses(csp.assign_stmt, 't = (1,2)', [[['t'], [['1', '2']]]])
    assert_parses(csp.assign_stmt, 'a, b = (1,2)', [[['a', 'b'], [['1', '2']]]])


def test_parsing_arrays():
    assert_parses(csp.expr, '[1, 2, 3]', [['1', '2', '3']])
    assert_parses(csp.array, '[[a, b], [c, d]]', [[['a', 'b'], ['c', 'd']]])
    assert_parses(
        csp.array,
        '[ [ [1+2,a,b]],[[3/4,c,d] ]]',
        [[[[['1', '+', '2'], 'a', 'b']], [[['3', '/', '4'], 'c', 'd']]]])


def test_parsing_array_comprehensions():
    assert_parses(csp.array, '[i for i in 0:N]', [['i', ['i', ['0', 'N']]]])

    assert_parses(csp.expr, '[i*2 for i in 0:2:4]', [[['i', '*', '2'], ['i', ['0', '2', '4']]]])

    assert_parses(csp.array, '[i+j*5 for i in 1:3 for j in 2:4]',
                  [[['i', '+', ['j', '*', '5']], ['i', ['1', '3']], ['j', ['2', '4']]]])

    assert_parses(csp.array, '[block for 1$i in 2:10]', [['block', ['1', 'i', ['2', '10']]]])
    assert_parses(
        csp.array,
        '[i^j for i in 1:3 for 2$j in 4:-1:2]',
        [[['i', '^', 'j'], ['i', ['1', '3']], ['2', 'j', ['4', ['-', '1'], '2']]]])

    # Dimension specifiers can be expressions too...
    assert_parses(
        csp.expr,
        '[i for (1+2)$i in 2:(3+5)]',
        [['i', [['1', '+', '2'], 'i', ['2', ['3', '+', '5']]]]])
    assert_parses(
        csp.expr,
        '[i for 1+2$i in 2:4]',
        [['i', [['1', '+', '2'], 'i', ['2', '4']]]])
    assert_does_not_parse(csp.expr, '[i for 1 $i in 2:4]')


def test_parsing_views():
    assert_parses(csp.expr, 'A[1:3:7]', [['A', ['1', '3', '7']]])
    assert_parses(csp.expr, 'A[2$6:-2:4]', [['A', ['2', '6', ['-', '2'], '4']]])
    assert_parses(csp.expr, 'sim:res[1$2]', [['sim:res', ['1', '2']]])
    assert_parses(csp.expr, 'func(A)[5]', [[['func', ['A']], ['5']]])
    assert_parses(csp.expr, 'arr[:]', [['arr', ['', '']]])
    assert_parses(csp.expr, 'arr[2:]', [['arr', ['2', '']]])
    assert_parses(csp.expr, 'arr[:2:]', [['arr', ['', '2', '']]])
    assert_parses(csp.expr, 'arr[:-alpha]', [['arr', ['', ['-', 'alpha']]]])
    assert_parses(csp.expr, 'arr[-3:-1:]', [['arr', [['-', '3'], ['-', '1'], '']]])
    assert_parses(csp.expr, 'genericity[*$:]', [['genericity', ['*', '', '']]])
    assert_parses(csp.expr, 'genericity[*$0]', [['genericity', ['*', '0']]])
    assert_parses(csp.expr, 'genericity[*$0:5]', [['genericity', ['*', '0', '5']]])
    assert_parses(csp.expr, 'genericity[*$0:5:50]', [['genericity', ['*', '0', '5', '50']]],)
    assert_parses(csp.expr, 'genericity[*$:5:]', [['genericity', ['*', '', '5', '']]])
    assert_parses(csp.expr, 'genericity[*$0:]', [['genericity', ['*', '0', '']]])
    assert_parses(csp.expr, 'multiples[3][4]', [['multiples', ['3'], ['4']]])
    assert_parses(
        csp.expr,
        'multiples[1$3][0$:-step:0][*$0]',
        [[
            'multiples',
            ['1', '3'],
            ['0', '', ['-', 'step'], '0'],
            ['*', '0']
        ]]
    )
    assert_parses(csp.expr, 'dimspec[dim$0:2]', [['dimspec', ['dim', '0', '2']]])
    assert_parses(csp.expr, 'okspace[ 0$ (1+2) : a+b : 50 ]',
                  [['okspace', ['0', ['1', '+', '2'], ['a', '+', 'b'], '50']]])

    # Some spaces aren't allowed
    assert_does_not_parse(csp.expr, 'arr [1]')
    assert_does_not_parse(csp.expr, 'arr[1] [3]')
    assert_does_not_parse(csp.expr, 'arr[1 $ 2]')


def test_parsing_find_and_index():
    # Curly braces represent index, with optional pad or shrink argument.  Find is a function call.
    assert_parses(csp.expr, 'find(arr)', [['find', ['arr']]])
    assert_does_not_parse(csp.expr, 'find (arr)')
    # assert_does_not_parse(csp.expr, 'find(arr, extra)') # Needs special support e.g. from parse actions

    assert_parses(csp.expr, 'arr{idxs}', [['arr', ['idxs']]])
    assert_does_not_parse(csp.expr, 'arr {spaced}')
    assert_parses(csp.expr, 'arr{idxs, shrink:1}', [['arr', ['idxs', '1']]])
    assert_parses(csp.expr, 'arr{idxs, dim, shrink:-1}', [['arr', ['idxs', 'dim', ['-', '1']]]])
    assert_parses(csp.expr, 'arr{idxs, dim, pad:1=value}', [['arr', ['idxs', 'dim', '1', 'value']]])
    assert_parses(csp.expr, 'arr{idxs, shrink:0, pad:1=value}', [['arr', ['idxs', '0', '1', 'value']]])
    assert_parses(
        csp.expr,
        'f(1,2){find(blah), 0, shrink:1}',
        [[['f', ['1', '2']], [['find', ['blah']], '0', '1']]]
    )
    assert_parses(
        csp.expr,
        'A{find(A), 0, pad:-1=1+2}',
        [['A', [['find', ['A']], '0', ['-', '1'], ['1', '+', '2']]]]
    )


def test_parsing_units_definitions():
    # Possible syntax:  (mult, offset, expt are 'numbers'; prefix is SI prefix name; base is ncIdent)
    #  new_simple = [mult] [prefix] base [+|- offset]
    #  new_complex = p.delimitedList( [mult] [prefix] base [^expt], '.')
    assert_parses(csp.units_def, 'ms = milli second', [['ms', ['milli', 'second']]])
    assert_parses(csp.units_def, 'C = kelvin - 273.15', [['C', ['kelvin', ['-', '273.15']]]])
    assert_parses(csp.units_def, 'C=kelvin+(-273.15)', [['C', ['kelvin', ['+', '(-273.15)']]]])
    assert_parses(csp.units_def, 'litre = 1000 centi metre^3', [['litre', ['1000', 'centi', 'metre', '3']]])
    assert_parses(
        csp.units_def,
        'accel_units = kilo metre . second^-2 "km/s^2"',
        [['accel_units', ['kilo', 'metre'], ['second', '-2'], 'km/s^2']])
    assert_parses(
        csp.units_def,
        'fahrenheit = (5/9) celsius + 32.0',
        [['fahrenheit', ['(5/9)', 'celsius', ['+', '32.0']]]])
    assert_parses(
        csp.units_def,
        'fahrenheit = (5/9) kelvin + (32 - 273.15 * 9 / 5)',
        [['fahrenheit', ['(5/9)', 'kelvin', ['+', '(32 - 273.15 * 9 / 5)']]]])

    assert_parses(csp.units, "units {}", [[]])
    assert_parses(csp.units, """units
{
# nM = nanomolar
nM = nano mole . litre^-1
hour = 3600 second
flux = nM . hour ^ -1

rate_const = hour^-1           # First order
rate_const_2 = nM^-1 . hour^-1 # Second order
}
""", [[['nM', ['nano', 'mole'], ['litre', '-1']],
        ['hour', ['3600', 'second']],
        ['flux', ['nM'], ['hour', '-1']],
        ['rate_const', ['hour', '-1']],
        ['rate_const_2', ['nM', '-1'], ['hour', '-1']]]],
    )


def test_parsing_accessors():
    for accessor in ['NUM_DIMS', 'SHAPE', 'NUM_ELEMENTS']:
        assert_parses(csp.accessor, '.' + accessor, [accessor])
        assert_parses(csp.expr, 'var.' + accessor, [['var', accessor]])
    for ptype in ['SIMPLE_VALUE', 'ARRAY', 'STRING', 'TUPLE', 'FUNCTION', 'NULL', 'DEFAULT']:
        assert_parses(csp.accessor, '.IS_' + ptype, ['IS_' + ptype])
        assert_parses(csp.expr, 'var.IS_' + ptype, [['var', 'IS_' + ptype]])
    assert_parses(csp.expr, 'arr.SHAPE[1]', [[['arr', 'SHAPE'], ['1']]])
    assert_parses(csp.expr, 'func(var).IS_ARRAY', [[['func', ['var']], 'IS_ARRAY']])
    assert_parses(csp.expr, 'A.SHAPE.IS_ARRAY', [['A', 'SHAPE', 'IS_ARRAY']])
    assert_does_not_parse(csp.expr, 'arr .SHAPE')


def test_parsing_map():
    assert_parses(csp.expr, 'map(func, a1, a2)', [['map', ['func', 'a1', 'a2']]])
    assert_parses(csp.expr, 'map(lambda a, b: a+b, A, B)',
                  [['map', [[[['a'], ['b']], ['a', '+', 'b']], 'A', 'B']]])
    assert_parses(csp.expr, 'map(id, a)', [['map', ['id', 'a']]])
    assert_parses(csp.expr, 'map(hof(arg), a, b, c, d, e)',
                  [['map', [['hof', ['arg']], 'a', 'b', 'c', 'd', 'e']]])
    # assert_does_not_parse(csp.expr, 'map(f)') # At present implemented just as a function call with special name


def test_parsing_fold():
    assert_parses(csp.expr, 'fold(func, array, init, dim)', [['fold', ['func', 'array', 'init', 'dim']]])
    assert_parses(csp.expr, 'fold(lambda a, b: a - b, f(), 1, 2)',
                  [['fold', [[[['a'], ['b']], ['a', '-', 'b']], ['f', []], '1', '2']]])
    assert_parses(csp.expr, 'fold(f, A)', [['fold', ['f', 'A']]])
    assert_parses(csp.expr, 'fold(f, A, 0)', [['fold', ['f', 'A', '0']]])
    assert_parses(csp.expr, 'fold(f, A, default, 1)', [['fold', ['f', 'A', [], '1']]])
    # assert_does_not_parse(csp, expr, 'fold()')
    # assert_does_not_parse(csp, expr, 'fold(f, A, i, d, extra)')


def test_parsing_wrapped_mathml_operators():
    assert_parses(csp.expr, '@3:+', [['3', '+']])
    assert_parses(csp.expr, '@1:MathML:sin', [['1', 'MathML:sin']])
    assert_parses(csp.expr, 'map(@2:/, a, b)', [['map', [['2', '/'], 'a', 'b']]])
    # assert_does_not_parse(csp.expr, '@0:+') # Best done at parse action level?
    assert_does_not_parse(csp.expr, '@1:non_mathml')
    assert_does_not_parse(csp.expr, '@ 2:*')
    assert_does_not_parse(csp.expr, '@2 :-')
    assert_does_not_parse(csp.expr, '@1:--')
    assert_does_not_parse(csp.expr, '@N:+')


def test_parsing_null_and_default():
    assert_parses(csp.expr, 'null', [[]])
    assert_parses(csp.expr, 'default', [[]])


def test_parsing_library():
    assert_parses(csp.library, 'library {}', [[]])

    mls = """library
{
    def f(a) {
        return a
    }
    f2 = lambda b: b/2
    const = 13
}
"""
    out = [[[
        ['f', [['a']], [['a']]],
        [['f2'], [[[['b']], ['b', '/', '2']]]],
        [['const'], ['13']]
    ]]]
    assert_parses(csp.library, mls, out)


def test_parsing_post_processing():
    mls = """post-processing
{
    a = check(sim:result)
    assert a > 5
}
"""
    out = [[[['a'], [['check', ['sim:result']]]], [['a', '>', '5']]]]
    assert_parses(csp.post_processing, mls, out)


def test_zzz_packrat_was_used():
    # Method name ensures this runs last!
    assert CSP.p.ParserElement.packrat_cache_stats[0] > 0

