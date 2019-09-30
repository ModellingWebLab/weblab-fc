import difflib
import filecmp
import os
import sys
import time
import unittest

# Import the module to test
# The default for this module now is to assume the Python implementation,
# so we have to override that!
sys._fc_csp_no_pyimpl = True
import fc.parsing.CompactSyntaxParser as CSP  # noqa: E402

csp = CSP.CompactSyntaxParser
# An end-of-string match that doesn't allow trailing whitespace
strict_string_end = CSP.p.StringEnd().leaveWhitespace()


class TestCompactSyntaxParser(unittest.TestCase):

    def check_parse_results(self, actual, expected):
        """Compare parse results to expected strings.

        The expected results may be given as a (nested) list or a dictionary, depending
        on whether you want to compare against matched tokens by order or results name.
        """
        def check_result(actual, expected):
            if isinstance(expected, str):
                self.assertEqual(actual, expected, '%s != %s' % (actual, expected))
            else:
                self.check_parse_results(actual, expected)
        if isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), '%s != %s' % (actual, expected))
            for i, result in enumerate(expected):
                check_result(actual[i], result)
        elif isinstance(expected, dict):
            self.assertEqual(len(actual), len(expected), '%s != %s' % (actual, expected))
            for key, value in expected.items():
                check_result(actual[key], value)

    def assertParses(self, grammar, input, results):
        """Utility method to test that a given grammar parses an input as expected."""
        actual_results = grammar.parseString(input, parseAll=True)
        self.check_parse_results(actual_results, results)

    def assertDoesNotParse(self, grammar, input):
        """Utility method to test that a given grammar fails to parse an input."""
        strict_grammar = grammar + strict_string_end
        self.assertRaises(CSP.p.ParseBaseException, strict_grammar.parseString, input)

    def assertFilesMatch(self, new_file_path, ref_file_path):
        """Utility method to check that two files have matching contents."""
        if not filecmp.cmp(new_file_path, ref_file_path):
            # Matching failed, so print something informative
            context_lines = 3
            from_date = time.ctime(os.stat(ref_file_path).st_mtime)
            to_date = time.ctime(os.stat(new_file_path).st_mtime)
            for line in difflib.unified_diff(open(ref_file_path).readlines(), open(new_file_path).readlines(),
                                             ref_file_path, new_file_path,
                                             from_date, to_date, n=context_lines):
                print(line, end=' ')
            self.fail("Output file '%s' does not match reference file '%s'" % (new_file_path, ref_file_path))

    def test_parsing_identifiers(self):
        self.assertDoesNotParse(csp.nc_ident, 'abc:def')
        self.assertDoesNotParse(csp.nc_ident, '123')

        self.assertParses(csp.ident_as_var, 'abc', ['abc'])
        self.assertParses(csp.ident, 'abc_def', ['abc_def'])
        self.assertParses(csp.ident, 'abc123', ['abc123'])
        self.assertParses(csp.ident, '_abc', ['_abc'])
        self.assertParses(csp.ident_as_var, 'abc:def', ['abc:def'])
        self.assertDoesNotParse(csp.ident, '123')

    def test_parsing_numbers(self):
        self.assertParses(csp.number, '123', ['123'])
        self.assertParses(csp.number, '1.23', ['1.23'])
        self.assertParses(csp.number, '-12.3', ['-12.3'])
        self.assertParses(csp.number, '12e3', ['12e3'])
        self.assertParses(csp.number, '-1.2e3', ['-1.2e3'])
        self.assertParses(csp.number, '1.2e3', ['1.2e3'])
        self.assertParses(csp.number, '1.2e+3', ['1.2e+3'])
        self.assertParses(csp.number, '12e-3', ['12e-3'])
        self.assertParses(csp.number, '-1e-2', ['-1e-2'])
        self.assertParses(csp.number, '0.1', ['0.1'])
        self.assertDoesNotParse(csp.number, '.123')
        self.assertDoesNotParse(csp.number, '123.')
        self.assertDoesNotParse(csp.number, '+123')
        self.assertDoesNotParse(csp.number, '1E3')

    def test_parsing_numbers_with_units(self):
        self.assertParses(csp.number, '123 :: dimensionless', ['123', 'dimensionless'])
        self.assertParses(csp.number, '-4e5::mA', ['-4e5', 'mA'])
        self.assertParses(csp.expr, '123 :: dimensionless', ['123'])
        self.assertParses(csp.expr, '4.6e5::mA', ['4.6e5'])
        self.assertDoesNotParse(csp.number, '123 :: ')
        self.assertDoesNotParse(csp.number, '123 :: prefix:units')

    def test_parsing_comments(self):
        self.assertParses(csp.comment, '# blah blah', [])
        self.assertDoesNotParse(csp.comment, '# blah blah\n')

    def test_parsing_simple_expressions(self):
        self.assertParses(csp.expr, '1', ['1'])
        self.assertParses(csp.expr, '(A)', ['A'])
        self.assertParses(
            csp.expr,
            '1 - 2 + 3 * 4 / 5 ^ 6',
            [['1', '-', '2', '+', ['3', '*', '4', '/', ['5', '^', '6']]]])
        self.assertParses(csp.expr, '-(a + -3)', [['-', ['a', '+', ['-', '3']]]])
        self.assertParses(csp.expr, '1 ^ 2 ^ 3', [['1', '^', '2', '^', '3']])
        self.assertParses(csp.expr, 'not 1 < 2', [[['not', '1'], '<', '2']])
        self.assertParses(csp.expr, 'not (1 < 2)', [['not', ['1', '<', '2']]])
        self.assertParses(csp.expr, 'not not my:var', [['not', ['not', 'my:var']]])
        self.assertParses(
            csp.expr,
            '1 < 2 <= (3 >= 4) > 5 == 6 != 7',
            [['1', '<', '2', '<=', ['3', '>=', '4'], '>', '5', '==', '6', '!=', '7']])
        self.assertParses(csp.expr, '1 < 2 + 4 > 3', [['1', '<', ['2', '+', '4'], '>', '3']])
        self.assertParses(csp.expr, '1 < 2 && 4 > 3', [[['1', '<', '2'], '&&', ['4', '>', '3']]])
        self.assertParses(csp.expr, '0 || 1 && 1', [['0', '||', '1', '&&', '1']])
        self.assertParses(csp.expr, 'A + B || C * D', [[['A', '+', 'B'], '||', ['C', '*', 'D']]])
        self.assertParses(csp.expr, 'if 1 then 2 else 3', [['1', '2', '3']])
        self.assertParses(csp.expr, 'if 1 < 2 then 3 + 4 else 5 * 6',
                          [[['1', '<', '2'], ['3', '+', '4'], ['5', '*', '6']]])

    def test_parsing_trace(self):
        self.assertParses(csp.expr, '1?', [['1']])
        self.assertParses(csp.expr, 'var?', [['var']])
        self.assertParses(csp.expr, '(1 + a)?', [[['1', '+', 'a']]])
        self.assertParses(csp.expr, '1 + a?', [['1', '+', ['a']]])

        action = csp.expr.parseString('var?', parseAll=True)
        assert action[0].expr().trace

        action = csp.expr.parseString('var', parseAll=True)
        assert not action[0].expr().trace

    def test_parsing_multi_line_expressions(self):
        self.assertParses(csp.expr, '(1 + 2) * 3', [[['1', '+', '2'], '*', '3']])
        self.assertParses(csp.expr, '((1 + 2)\\\n * 3)', [[['1', '+', '2'], '*', '3']])
        self.assertParses(csp.expr, '(1 + 2)\\\n * 3', [[['1', '+', '2'], '*', '3']])
        self.assertParses(csp.expr, '((1 + 2)\n  * 3)', [[['1', '+', '2'], '*', '3']])
        self.assertParses(csp.expr, '((1 + 2)#embedded comment\n  * 3)', [[['1', '+', '2'], '*', '3']])
        # This one doesn't match Python behaviour, but it's hard to stop it parsing while allowing
        # the above to work.
        self.assertParses(csp.expr, '(1 + 2)\n * 3', [[['1', '+', '2'], '*', '3']])

    def test_parsing_simple_assignments(self):
        self.assertParses(csp.simple_assign, 'var = value', [['var', 'value']])
        self.assertParses(csp.simple_assign, 'var = pre:value', [['var', 'pre:value']])
        self.assertDoesNotParse(csp.simple_assign, 'pre:var = value')
        self.assertParses(csp.simple_assign, 'var = 1 + 2', [['var', ['1', '+', '2']]])
        self.assertParses(csp.simple_assign_list, 'v1 = 1\nv2=2', [[['v1', '1'], ['v2', '2']]])
        self.assertParses(csp.simple_assign_list, '', [[]])

    def test_parsing_namespaces(self):
        self.assertParses(csp.ns_decl, 'namespace prefix = "urn:test"', [{'prefix': 'prefix', 'uri': 'urn:test'}])
        self.assertParses(csp.ns_decl, "namespace prefix='urn:test'", [{'prefix': 'prefix', 'uri': 'urn:test'}])
        self.assertParses(csp.ns_decls, 'namespace n1="urn:t1"#test ns\nnamespace n2 = "urn:t2"',
                          [{'prefix': 'n1', 'uri': 'urn:t1'}, {'prefix': 'n2', 'uri': 'urn:t2'}])
        self.assertParses(csp.ns_decls, '', [])
        self.assertDoesNotParse(csp.ns_decls, 'namespace n="uri"\n')

    def test_parsing_inputs(self):
        mls = """inputs {
    label_duration = 300  # How long to label for
    unlabel_time = 600000 # When to stop labelling
    use_tapasin = 1       # Whether to include Tapasin
}
"""
        out = [[[['label_duration', '300'], ['unlabel_time', '600000'], ['use_tapasin', '1']]]]
        self.assertParses(csp.inputs, mls, out)
        self.assertParses(csp.inputs, 'inputs {}', [[[]]])
        self.assertParses(csp.inputs, 'inputs\n{\n}\n', [[[]]])
        self.assertParses(csp.inputs, 'inputs{X=1}', [[[['X', '1']]]])

    def test_parsing_imports(self):
        self.assertParses(
            csp.import_stmt,
            'import std = "../library/BasicLibrary.xml"',
            [['std', '../library/BasicLibrary.xml']]
        )
        self.assertParses(
            csp.import_stmt,
            "import 'TestS1S2.xml'",
            [['', 'TestS1S2.xml']])
        self.assertParses(
            csp.imports,
            'import l1="file1"#blah\nimport "file2"',
            [['l1', 'file1'], ['', 'file2']])
        self.assertParses(csp.imports, '', [])
        self.assertDoesNotParse(csp.imports, 'import "file"\n')

    def test_parsing_imports_with_set_input(self):
        mls = """import "S1S2.txt" {
    steady_state_beats = 10
    timecourse_duration = 2000
}"""
        self.assertParses(
            csp.import_stmt,
            mls,
            [['', 'S1S2.txt', [['steady_state_beats', '10'], ['timecourse_duration', '2000']]]]
        )
        self.assertParses(csp.import_stmt, 'import "file.txt" { }', [['', 'file.txt', []]])
        self.assertDoesNotParse(csp.import_stmt, 'import "file.txt" { } \n')

    def test_parsing_model_interface(self):
        self.assertParses(csp.set_time_units, 'independent var units u', [['u']])
        self.assertParses(csp.input_variable, 'input test:var units u = 1.2', [['test:var', 'u', '1.2']])
        self.assertParses(csp.input_variable, 'input test:var units u', [['test:var', 'u']])
        self.assertParses(csp.input_variable, 'input test:var = -1e2', [['test:var', '-1e2']])
        self.assertParses(csp.input_variable, 'input test:var', [['test:var']])
        self.assertDoesNotParse(csp.input_variable, 'input no_prefix')

        self.assertParses(csp.output_variable, 'output test:var', [['test:var']])
        self.assertParses(csp.output_variable, 'output test:var units uname', [['test:var', 'uname']])
        self.assertDoesNotParse(csp.output_variable, 'output no_prefix')

        self.assertParses(
            csp.optional_variable, 'optional prefix:name', [['prefix:name', 20]])
        self.assertParses(
            csp.optional_variable,
            'optional p:n default p:v + local * 2 :: u',
            [['p:n', 12, ['p:v', '+', ['local', '*', '2']], 41]])
        self.assertDoesNotParse(csp.optional_variable, 'optional no_prefix')

        self.assertParses(csp.new_variable, 'var varname units uname = 0', [['varname', 'uname', '0']])
        self.assertParses(csp.new_variable, 'var varname units uname', [['varname', 'uname']])
        self.assertDoesNotParse(csp.new_variable, 'var prefix:varname units uname = 0')
        self.assertDoesNotParse(csp.new_variable, 'var varname = 0')
        self.assertDoesNotParse(csp.new_variable, 'var varname')

        self.assertParses(csp.clamp_variable, 'clamp p:v', [['p:v']])
        self.assertParses(csp.clamp_variable, 'clamp p:v to 0 :: U', [['p:v', '0']])

        self.assertParses(
            csp.model_equation,
            'define local_var = 1::U + model:var',
            [['local_var', ['1', '+', 'model:var']]])
        self.assertParses(
            csp.model_equation,
            'define model:var = 2.5 :: units / local_var',
            [['model:var', ['2.5', '/', 'local_var']]])
        self.assertParses(
            csp.model_equation,
            'define diff(oxmeta:membrane_voltage; oxmeta:time) = 1 :: mV_per_ms',
            [[['oxmeta:membrane_voltage', 'oxmeta:time'], '1']])

        self.assertParses(
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
        self.assertParses(csp.model_interface, mls, out)
        self.assertParses(csp.model_interface, 'model interface {}', [[]])
        self.assertParses(csp.model_interface, 'model interface#comment\n{output test:time\n}', [[['test:time']]])
        self.assertParses(csp.model_interface, 'model interface {output test:time }', [[['test:time']]])

    def test_parsing_uniform_range(self):
        self.assertParses(csp.range, 'range time units ms uniform 0:1:1000', [['time', 'ms', ['0', '1', '1000']]])
        self.assertParses(csp.range, 'range time units ms uniform 0:1000', [['time', 'ms', ['0', '1000']]])
        self.assertParses(csp.range, 'range t units s uniform 0:end', [['t', 's', ['0', 'end']]])

        # Spaces or brackets are required in this case to avoid 'start:step:end' parsing as an ident
        self.assertParses(
            csp.range,
            'range t units s uniform start : step : end',
            [['t', 's', ['start', 'step', 'end']]])
        self.assertParses(csp.range, 'range t units s uniform start:(step):end', [['t', 's', ['start', 'step', 'end']]])
        self.assertDoesNotParse(csp.range, 'range t units s uniform start:step:end')

    def test_parsing_vector_range(self):
        self.assertParses(
            csp.range,
            'range run units dimensionless vector [1, 2, 3, 4]',
            [['run', 'dimensionless', ['1', '2', '3', '4']]])

    def test_parsing_while_range(self):
        self.assertParses(
            csp.range, 'range rpt units dimensionless while rpt < 5',
            [['rpt', 'dimensionless', ['rpt', '<', '5']]])

    def test_parsing_modifiers(self):
        self.assertParses(csp.modifier_when, 'at start', ['start'])
        self.assertParses(csp.modifier_when, 'at each loop', ['each'])
        self.assertParses(csp.modifier_when, 'at end', ['end'])

        self.assertParses(csp.set_variable, 'set model:V = 5.0', ['model:V', '5.0'])
        self.assertParses(csp.set_variable, 'set model:t = time + 10.0', ['model:t', ['time', '+', '10.0']])

        self.assertParses(csp.save_state, 'save as state_name', ['state_name'])
        self.assertDoesNotParse(csp.save_state, 'save as state:name')

        self.assertParses(csp.reset_state, 'reset', [])
        self.assertParses(csp.reset_state, 'reset to state_name', ['state_name'])
        self.assertDoesNotParse(csp.reset_state, 'reset to state:name')

        mls = """modifiers {
        # Multiple
        # comment lines
        # are OK
        at start reset
        at each loop set model:input = loopVariable

        # Blank lines OK too
        at end save as savedState
} # Trailing comments are fine"""
        self.assertParses(
            csp.modifiers, mls,
            [[['start', []], ['each', ['model:input', 'loopVariable']], ['end', ['savedState']]]])
        self.assertParses(csp.modifiers, 'modifiers {at start reset}', [[['start', []]]])

    def test_parsing_timecourse_simulations(self):
        self.assertParses(
            csp.simulation,
            'simulation sim = timecourse { range time units ms uniform 1:10 }',
            [['sim', [['time', 'ms', ['1', '10']]]]])
        self.assertParses(
            csp.simulation,
            'simulation timecourse #c\n#c\n{\n range time units ms uniform 1:10\n\n }#c',
            [['', [['time', 'ms', ['1', '10']]]]])

        mls = """simulation sim = timecourse {
range time units U while time < 100
modifiers { at end save as prelim }
}"""
        self.assertParses(csp.simulation, mls,
                          [['sim', [['time', 'U', ['time', '<', '100']], [['end', ['prelim']]]]]])
        self.assertDoesNotParse(csp.simulation, 'simulation sim = timecourse {}')

    def test_parsing_one_step_simulations(self):
        self.assertParses(csp.simulation, 'simulation oneStep', [['', []]])
        self.assertParses(csp.simulation, 'simulation sim = oneStep', [['sim', []]])
        self.assertParses(csp.simulation, 'simulation oneStep 1.0', [['', ['1.0']]])
        self.assertParses(csp.simulation, 'simulation sim = oneStep step', [['sim', ['step']]])
        self.assertParses(
            csp.simulation,
            'simulation oneStep { modifiers { at start set a = 1 } }',
            [['', [[['start', ['a', '1']]]]]])

    def test_parsing_nested_simulations(self):
        self.assertParses(
            csp.simulation,
            'simulation rpt = nested { range run units U while not rpt:result\n nests sim }',
            [['rpt', [['run', 'U', ['not', 'rpt:result']], ['sim']]]])

        self.assertParses(
            csp.simulation,
            """simulation nested {
range R units U uniform 3:5
modifiers { at each loop reset to prelim }
nests sim
}""",
            [['', [['R', 'U', ['3', '5']], [['each', ['prelim']]], ['sim']]]])

        self.assertParses(
            csp.simulation,
            """simulation nested { range R units U uniform 1:2
nests simulation timecourse { range t units u uniform 1:100 } }""",
            [['', [['R', 'U', ['1', '2']], [['', [['t', 'u', ['1', '100']]]]]]]])

        self.assertDoesNotParse(
            csp.simulation,
            'simulation rpt = nested { range run units U while 1 }')

    def test_parsing_nested_protocol(self):
        self.assertParses(
            csp.simulation,
            'simulation nested { range iter units D vector [0, 1]\n nests protocol "P" { } }',
            [['', [['iter', 'D', ['0', '1']], [['P', []]]]]])

        self.assertParses(
            csp.simulation,
            """simulation nested {
    range iter units D vector [0, 1]
    nests protocol "../proto.xml" {
        input1 = 1
        input2 = 2
    }
}""",
            [['', [['iter', 'D', ['0', '1']], [['../proto.xml', [['input1', '1'], ['input2', '2']]]]]]])

        self.assertParses(
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
        self.assertParses(
            csp.simulation,
            'simulation nested { range iter units D vector [0, 1]\n nests protocol "P" { }? }',
            [['', [['iter', 'D', ['0', '1']], [['P', []]]]]])

    def test_parsing_tasks(self):
        self.assertParses(
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

    def test_parsing_output_specifications(self):
        self.assertParses(csp.output_spec, 'name = model:var "Description"', [['name', 'model:var', 'Description']])
        self.assertParses(csp.output_spec, r'name = ref:var units U "Description \"quotes\""',
                          [['name', 'ref:var', 'U', 'Description "quotes"']])
        self.assertParses(csp.output_spec, "name = ref:var units U 'Description \\'quotes\\' \"too\"'",
                          [['name', 'ref:var', 'U', 'Description \'quotes\' "too"']])
        self.assertParses(csp.output_spec, 'varname units UU', [['varname', 'UU']])
        self.assertParses(csp.output_spec, 'varname units UU "desc"', [['varname', 'UU', 'desc']])
        self.assertParses(csp.output_spec, 'optional varname units UU', [['optional', 'varname', 'UU']])
        self.assertParses(csp.output_spec, 'optional varname = ref:var', [['optional', 'varname', 'ref:var']])
        self.assertDoesNotParse(csp.output_spec, 'varname_no_units')

        self.assertParses(csp.outputs, """outputs #cccc
{ #cdc
        n1 = n2 units u1
        n3 = p:m 'd1'
        n4 units u2 "d2"
        optional n5 units u3
} #cpc
""", [[['n1', 'n2', 'u1'], ['n3', 'p:m', 'd1'], ['n4', 'u2', 'd2'], ['optional', 'n5', 'u3']]])
        self.assertParses(csp.outputs, "outputs {}", [[]])

    def test_parsing_plot_specifications(self):
        self.assertParses(csp.plot_curve, 'y against x', [['y', 'x']])
        self.assertParses(csp.plot_curve, 'y, y2 against x', [['y', 'y2', 'x']])
        self.assertDoesNotParse(csp.plot_curve, 'm:y against x')
        self.assertDoesNotParse(csp.plot_curve, 'y against m:x')
        self.assertParses(
            csp.plot_spec,
            'plot "A title\'s good" { y1, y2 against x1\n y3 against x2 }',
            [["A title's good", ['y1', 'y2', 'x1'], ['y3', 'x2']]])
        self.assertParses(csp.plot_spec, 'plot "Keys" { y against x key k }', [["Keys", ['y', 'x', 'k']]])
        self.assertDoesNotParse(csp.plot_spec, 'plot "only title" {}')
        self.assertDoesNotParse(csp.plot_spec, 'plot "only title"')

        self.assertParses(csp.plots, """plots { plot "t1" { v1 against v2 key vk }
        plot "t1" { v3, v4 against v5 }
}""", [[['t1', ['v1', 'v2', 'vk']], ['t1', ['v3', 'v4', 'v5']]]])
        self.assertParses(csp.plots, 'plots {}', [[]])

    def test_parsing_function_calls(self):
        self.assertParses(csp.function_call, 'noargs()', [['noargs', []]])
        self.assertParses(csp.function_call, 'swap(a, b)', [['swap', ['a', 'b']]])
        self.assertParses(csp.function_call, 'double(33)', [['double', ['33']]])
        self.assertParses(csp.function_call, 'double(a + b)', [['double', [['a', '+', 'b']]]])
        self.assertParses(csp.function_call, 'std:max(A)', [['std:max', ['A']]])
        self.assertDoesNotParse(csp.function_call, 'spaced (param)')
        self.assertParses(csp.expr, 'func(a,b, 3)', [['func', ['a', 'b', '3']]])

    def test_parsing_mathml_operators(self):
        # MathML that doesn't have a special operator is represented as a normal function call,
        # with the 'magic' MathML: prefix.
        self.assertEqual(len(csp.mathml_operators), 12 + 3 * 8)
        for trigbase in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']:
            self.assertTrue(trigbase in csp.mathml_operators)
            self.assertTrue(trigbase + 'h' in csp.mathml_operators)
            self.assertTrue('arc' + trigbase in csp.mathml_operators)
            self.assertTrue('arc' + trigbase + 'h' in csp.mathml_operators)
        for op in 'quotient rem max min root xor abs floor ceiling exp ln log'.split():
            self.assertTrue(op in csp.mathml_operators)
        self.assertParses(csp.expr, 'MathML:exp(MathML:floor(MathML:exponentiale))',
                          [['MathML:exp', [['MathML:floor', ['MathML:exponentiale']]]]])

    def test_parsing_assign_statements(self):
        self.assertParses(csp.assign_stmt, 'var = value', [[['var'], ['value']]])
        self.assertParses(csp.assign_stmt, 'var = pre:value', [[['var'], ['pre:value']]])
        self.assertDoesNotParse(csp.assign_stmt, 'pre:var = value')
        self.assertParses(csp.assign_stmt, 'var = 1 + 2', [[['var'], [['1', '+', '2']]]])

        self.assertParses(csp.assign_stmt, 'a, b = tuple', [[['a', 'b'], ['tuple']]])
        self.assertParses(csp.assign_stmt, 'a, b = b, a', [[['a', 'b'], ['b', 'a']]])
        self.assertParses(csp.assign_stmt, 'a, b = (b, a)', [[['a', 'b'], [['b', 'a']]]])
        self.assertDoesNotParse(csp.assign_stmt, 'p:a, p:b = e')
        self.assertDoesNotParse(csp.assign_stmt, '')

    def test_parsing_optional_assignments(self):
        self.assertParses(
            csp.assign_stmt, 'optional var = value', [[['var'], ['value']]])

        # The following should parse as a non-optional assignment!
        self.assertParses(
            csp.assign_stmt, 'optional = expr', [[['optional'], ['expr']]])

    def test_parsing_return_statements(self):
        self.assertParses(csp.return_stmt, 'return 2 * a', [[['2', '*', 'a']]])
        self.assertParses(csp.return_stmt, 'return (3 - 4)', [[['3', '-', '4']]])
        self.assertParses(csp.return_stmt, 'return a, b', [['a', 'b']])
        self.assertParses(csp.return_stmt, 'return a + 1, b - 1', [[['a', '+', '1'], ['b', '-', '1']]])
        self.assertParses(csp.return_stmt, 'return (a, b)', [[['a', 'b']]])

    def test_parsing_assert_statements(self):
        self.assertParses(csp.assert_stmt, 'assert a + b', [[['a', '+', 'b']]])
        self.assertParses(csp.assert_stmt, 'assert (a + b)', [[['a', '+', 'b']]])
        self.assertParses(csp.assert_stmt, 'assert 1', [['1']])

    def test_parsing_statement_lists(self):
        self.assertParses(csp.stmt_list, "b=-a\nassert 1", [[[['b'], [['-', 'a']]], ['1']]])

        mls = """assert a < 0 # comments are ok

# as are blank lines
b = -a
assert b > 0
c, d = a * 2, b + 1
return c, d"""
        self.assertParses(
            csp.stmt_list,
            mls,
            [[
                [['a', '<', '0']],
                [['b'], [['-', 'a']]],
                [['b', '>', '0']],
                [['c', 'd'], [['a', '*', '2'], ['b', '+', '1']]],
                ['c', 'd']
            ]])
        self.assertDoesNotParse(csp.stmt_list, '')

    def test_parsing_lambda_expressions(self):
        self.assertParses(csp.lambda_expr, 'lambda a: a + 1', [[[['a']], ['a', '+', '1']]])
        self.assertParses(csp.lambda_expr, 'lambda a, b: a + b', [[[['a'], ['b']], ['a', '+', 'b']]])
        self.assertParses(csp.lambda_expr, 'lambda a, b=2: a - b', [[[['a'], ['b', '2']], ['a', '-', 'b']]])
        self.assertParses(csp.expr, 'lambda a=c, b: a * b', [[[['a', 'c'], ['b']], ['a', '*', 'b']]])
        self.assertParses(csp.lambda_expr, 'lambda a=p:c, b: a * b', [[[['a', 'p:c'], ['b']], ['a', '*', 'b']]])
        self.assertDoesNotParse(csp.lambda_expr, 'lambda p:a: 5')

        mls = """lambda a, b {
assert a > b
c = a - b
return c
}
"""
        self.assertParses(
            csp.lambda_expr,
            mls,
            [[
                [['a'], ['b']],
                [[['a', '>', 'b']], [['c'], [['a', '-', 'b']]], ['c']]
            ]])
        self.assertParses(csp.expr, "lambda a, b { return b, a }", [[[['a'], ['b']], [['b', 'a']]]])
        self.assertParses(csp.expr, "lambda a { return a }", [[[['a']], [['a']]]])
        self.assertParses(csp.expr, 'lambda { return 1 }', [[[], [['1']]]])
        self.assertParses(csp.expr, 'lambda: 1', [[[], '1']])

    def test_parsing_function_definitions(self):
        self.assertParses(
            csp.function_defn,
            'def double(a)\n {\n return a * 2\n }',
            [['double', [['a']], [[['a', '*', '2']]]]])

        self.assertParses(
            csp.function_defn,
            'def double(a): a * 2',
            [['double', [['a']], ['a', '*', '2']]])

        # A function definition is just sugar for an assignment of a lambda expression
        self.assertParses(
            csp.stmt_list,
            'def double(a) {\n    return a * 2}',
            [[['double', [['a']], [[['a', '*', '2']]]]]])
        self.assertParses(
            csp.function_defn, 'def noargs(): 1', [['noargs', [], '1']])

    def test_parsing_nested_functions(self):
        self.assertParses(csp.function_defn, """def outer()
{
    def inner1(a): a/2
    inner2 = lambda { return 5 }
    def inner3(b) {
        return b*2
    }
    return inner1(1) + inner2() + inner3(2)
}""", [['outer', [], [['inner1', [['a']], ['a', '/', '2']],
                      [['inner2'], [[[], [['5']]]]],
                      ['inner3', [['b']], [[['b', '*', '2']]]],
                      [[['inner1', ['1']], '+', ['inner2', []], '+', ['inner3', ['2']]]]]]])

    def test_parsing_tuples(self):
        self.assertParses(csp.tuple, '(1,2)', [['1', '2']])
        self.assertParses(csp.tuple, '(1+a,2*b)', [[['1', '+', 'a'], ['2', '*', 'b']]])
        self.assertParses(csp.tuple, '(singleton,)', [['singleton']])
        self.assertDoesNotParse(csp.tuple, '(1)')  # You need a Python-style comma as above
        self.assertParses(csp.expr, '(1,2)', [['1', '2']])
        self.assertParses(csp.expr, '(1,a,3,c)', [['1', 'a', '3', 'c']])
        self.assertParses(csp.assign_stmt, 't = (1,2)', [[['t'], [['1', '2']]]])
        self.assertParses(csp.assign_stmt, 'a, b = (1,2)', [[['a', 'b'], [['1', '2']]]])

    def test_parsing_arrays(self):
        self.assertParses(csp.expr, '[1, 2, 3]', [['1', '2', '3']])
        self.assertParses(csp.array, '[[a, b], [c, d]]', [[['a', 'b'], ['c', 'd']]])
        self.assertParses(
            csp.array,
            '[ [ [1+2,a,b]],[[3/4,c,d] ]]',
            [[[[['1', '+', '2'], 'a', 'b']], [[['3', '/', '4'], 'c', 'd']]]])

    def test_parsing_array_comprehensions(self):
        self.assertParses(csp.array, '[i for i in 0:N]', [['i', ['i', ['0', 'N']]]])

        self.assertParses(csp.expr, '[i*2 for i in 0:2:4]', [[['i', '*', '2'], ['i', ['0', '2', '4']]]])

        self.assertParses(csp.array, '[i+j*5 for i in 1:3 for j in 2:4]',
                          [[['i', '+', ['j', '*', '5']], ['i', ['1', '3']], ['j', ['2', '4']]]])

        self.assertParses(csp.array, '[block for 1$i in 2:10]', [['block', ['1', 'i', ['2', '10']]]])
        self.assertParses(
            csp.array,
            '[i^j for i in 1:3 for 2$j in 4:-1:2]',
            [[['i', '^', 'j'], ['i', ['1', '3']], ['2', 'j', ['4', ['-', '1'], '2']]]])

        # Dimension specifiers can be expressions too...
        self.assertParses(
            csp.expr,
            '[i for (1+2)$i in 2:(3+5)]',
            [['i', [['1', '+', '2'], 'i', ['2', ['3', '+', '5']]]]])
        self.assertParses(
            csp.expr,
            '[i for 1+2$i in 2:4]',
            [['i', [['1', '+', '2'], 'i', ['2', '4']]]])
        self.assertDoesNotParse(csp.expr, '[i for 1 $i in 2:4]')

    def test_parsing_views(self):
        self.assertParses(csp.expr, 'A[1:3:7]', [['A', ['1', '3', '7']]])
        self.assertParses(csp.expr, 'A[2$6:-2:4]', [['A', ['2', '6', ['-', '2'], '4']]])
        self.assertParses(csp.expr, 'sim:res[1$2]', [['sim:res', ['1', '2']]])
        self.assertParses(csp.expr, 'func(A)[5]', [[['func', ['A']], ['5']]])
        self.assertParses(csp.expr, 'arr[:]', [['arr', ['', '']]])
        self.assertParses(csp.expr, 'arr[2:]', [['arr', ['2', '']]])
        self.assertParses(csp.expr, 'arr[:2:]', [['arr', ['', '2', '']]])
        self.assertParses(csp.expr, 'arr[:-alpha]', [['arr', ['', ['-', 'alpha']]]])
        self.assertParses(csp.expr, 'arr[-3:-1:]', [['arr', [['-', '3'], ['-', '1'], '']]])
        self.assertParses(csp.expr, 'genericity[*$:]', [['genericity', ['*', '', '']]])
        self.assertParses(csp.expr, 'genericity[*$0]', [['genericity', ['*', '0']]])
        self.assertParses(csp.expr, 'genericity[*$0:5]', [['genericity', ['*', '0', '5']]])
        self.assertParses(csp.expr, 'genericity[*$0:5:50]', [['genericity', ['*', '0', '5', '50']]],)
        self.assertParses(csp.expr, 'genericity[*$:5:]', [['genericity', ['*', '', '5', '']]])
        self.assertParses(csp.expr, 'genericity[*$0:]', [['genericity', ['*', '0', '']]])
        self.assertParses(csp.expr, 'multiples[3][4]', [['multiples', ['3'], ['4']]])
        self.assertParses(
            csp.expr,
            'multiples[1$3][0$:-step:0][*$0]',
            [[
                'multiples',
                ['1', '3'],
                ['0', '', ['-', 'step'], '0'],
                ['*', '0']
            ]]
        )
        self.assertParses(csp.expr, 'dimspec[dim$0:2]', [['dimspec', ['dim', '0', '2']]])
        self.assertParses(csp.expr, 'okspace[ 0$ (1+2) : a+b : 50 ]',
                          [['okspace', ['0', ['1', '+', '2'], ['a', '+', 'b'], '50']]])

        # Some spaces aren't allowed
        self.assertDoesNotParse(csp.expr, 'arr [1]')
        self.assertDoesNotParse(csp.expr, 'arr[1] [3]')
        self.assertDoesNotParse(csp.expr, 'arr[1 $ 2]')

    def test_parsing_find_and_index(self):
        # Curly braces represent index, with optional pad or shrink argument.  Find is a function call.
        self.assertParses(csp.expr, 'find(arr)', [['find', ['arr']]])
        self.assertDoesNotParse(csp.expr, 'find (arr)')
        # self.assertDoesNotParse(csp.expr, 'find(arr, extra)') # Needs special support e.g. from parse actions

        self.assertParses(csp.expr, 'arr{idxs}', [['arr', ['idxs']]])
        self.assertDoesNotParse(csp.expr, 'arr {spaced}')
        self.assertParses(csp.expr, 'arr{idxs, shrink:1}', [['arr', ['idxs', '1']]])
        self.assertParses(csp.expr, 'arr{idxs, dim, shrink:-1}', [['arr', ['idxs', 'dim', ['-', '1']]]])
        self.assertParses(csp.expr, 'arr{idxs, dim, pad:1=value}', [['arr', ['idxs', 'dim', '1', 'value']]])
        self.assertParses(csp.expr, 'arr{idxs, shrink:0, pad:1=value}', [['arr', ['idxs', '0', '1', 'value']]])
        self.assertParses(
            csp.expr,
            'f(1,2){find(blah), 0, shrink:1}',
            [[['f', ['1', '2']], [['find', ['blah']], '0', '1']]]
        )
        self.assertParses(
            csp.expr,
            'A{find(A), 0, pad:-1=1+2}',
            [['A', [['find', ['A']], '0', ['-', '1'], ['1', '+', '2']]]]
        )

    def test_parsing_units_definitions(self):
        # Possible syntax:  (mult, offset, expt are 'numbers'; prefix is SI prefix name; base is ncIdent)
        #  new_simple = [mult] [prefix] base [+|- offset]
        #  new_complex = p.delimitedList( [mult] [prefix] base [^expt], '.')
        self.assertParses(csp.units_def, 'ms = milli second', [['ms', ['milli', 'second']]])
        self.assertParses(csp.units_def, 'C = kelvin - 273.15', [['C', ['kelvin', ['-', '273.15']]]])
        self.assertParses(csp.units_def, 'C=kelvin+(-273.15)', [['C', ['kelvin', ['+', '(-273.15)']]]])
        self.assertParses(csp.units_def, 'litre = 1000 centi metre^3', [['litre', ['1000', 'centi', 'metre', '3']]])
        self.assertParses(
            csp.units_def,
            'accel_units = kilo metre . second^-2 "km/s^2"',
            [['accel_units', ['kilo', 'metre'], ['second', '-2'], 'km/s^2']])
        self.assertParses(
            csp.units_def,
            'fahrenheit = (5/9) celsius + 32.0',
            [['fahrenheit', ['(5/9)', 'celsius', ['+', '32.0']]]])
        self.assertParses(
            csp.units_def,
            'fahrenheit = (5/9) kelvin + (32 - 273.15 * 9 / 5)',
            [['fahrenheit', ['(5/9)', 'kelvin', ['+', '(32 - 273.15 * 9 / 5)']]]])

        self.assertParses(csp.units, "units {}", [[]])
        self.assertParses(csp.units, """units
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

    def test_parsing_accessors(self):
        for accessor in ['NUM_DIMS', 'SHAPE', 'NUM_ELEMENTS']:
            self.assertParses(csp.accessor, '.' + accessor, [accessor])
            self.assertParses(csp.expr, 'var.' + accessor, [['var', accessor]])
        for ptype in ['SIMPLE_VALUE', 'ARRAY', 'STRING', 'TUPLE', 'FUNCTION', 'NULL', 'DEFAULT']:
            self.assertParses(csp.accessor, '.IS_' + ptype, ['IS_' + ptype])
            self.assertParses(csp.expr, 'var.IS_' + ptype, [['var', 'IS_' + ptype]])
        self.assertParses(csp.expr, 'arr.SHAPE[1]', [[['arr', 'SHAPE'], ['1']]])
        self.assertParses(csp.expr, 'func(var).IS_ARRAY', [[['func', ['var']], 'IS_ARRAY']])
        self.assertParses(csp.expr, 'A.SHAPE.IS_ARRAY', [['A', 'SHAPE', 'IS_ARRAY']])
        self.assertDoesNotParse(csp.expr, 'arr .SHAPE')

    def test_parsing_map(self):
        self.assertParses(csp.expr, 'map(func, a1, a2)', [['map', ['func', 'a1', 'a2']]])
        self.assertParses(csp.expr, 'map(lambda a, b: a+b, A, B)',
                          [['map', [[[['a'], ['b']], ['a', '+', 'b']], 'A', 'B']]])
        self.assertParses(csp.expr, 'map(id, a)', [['map', ['id', 'a']]])
        self.assertParses(csp.expr, 'map(hof(arg), a, b, c, d, e)',
                          [['map', [['hof', ['arg']], 'a', 'b', 'c', 'd', 'e']]])
        # self.assertDoesNotParse(csp.expr, 'map(f)') # At present implemented just as a function call with special name

    def test_parsing_fold(self):
        self.assertParses(csp.expr, 'fold(func, array, init, dim)', [['fold', ['func', 'array', 'init', 'dim']]])
        self.assertParses(csp.expr, 'fold(lambda a, b: a - b, f(), 1, 2)',
                          [['fold', [[[['a'], ['b']], ['a', '-', 'b']], ['f', []], '1', '2']]])
        self.assertParses(csp.expr, 'fold(f, A)', [['fold', ['f', 'A']]])
        self.assertParses(csp.expr, 'fold(f, A, 0)', [['fold', ['f', 'A', '0']]])
        self.assertParses(csp.expr, 'fold(f, A, default, 1)', [['fold', ['f', 'A', [], '1']]])
        # self.assertDoesNotParse(csp, expr, 'fold()')
        # self.assertDoesNotParse(csp, expr, 'fold(f, A, i, d, extra)')

    def test_parsing_wrapped_mathml_operators(self):
        self.assertParses(csp.expr, '@3:+', [['3', '+']])
        self.assertParses(csp.expr, '@1:MathML:sin', [['1', 'MathML:sin']])
        self.assertParses(csp.expr, 'map(@2:/, a, b)', [['map', [['2', '/'], 'a', 'b']]])
        # self.assertDoesNotParse(csp.expr, '@0:+') # Best done at parse action level?
        self.assertDoesNotParse(csp.expr, '@1:non_mathml')
        self.assertDoesNotParse(csp.expr, '@ 2:*')
        self.assertDoesNotParse(csp.expr, '@2 :-')
        self.assertDoesNotParse(csp.expr, '@1:--')
        self.assertDoesNotParse(csp.expr, '@N:+')

    def test_parsing_null_and_default(self):
        self.assertParses(csp.expr, 'null', [[]])
        self.assertParses(csp.expr, 'default', [[]])

    def test_parsing_library(self):
        self.assertParses(csp.library, 'library {}', [[]])

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
        self.assertParses(csp.library, mls, out)

    def test_parsing_post_processing(self):
        mls = """post-processing
{
    a = check(sim:result)
    assert a > 5
}
"""
        out = [[[['a'], [['check', ['sim:result']]]], [['a', '>', '5']]]]
        self.assertParses(csp.post_processing, mls, out)

    def test_zzz_packrat_was_used(self):
        # Method name ensures this runs last!
        assert CSP.p.ParserElement.packrat_cache_stats[0] > 0
