
"""Copyright (c) 2005-2012, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import unittest
import sys

# Import the module to test
sys.path[0:0] = ['python/pycml', 'projects/FunctionalCuration/src/proto/parsing']
import CompactSyntaxParser as CSP

csp = CSP.CompactSyntaxParser
# An end-of-string match that doesn't allow trailing whitespace
strict_string_end = CSP.p.StringEnd().leaveWhitespace()

class TestCompactSyntaxParser(unittest.TestCase):
    def checkResultsList(self, actual, expected):
        """Compare parse results to expected strings."""
        self.assertEqual(len(actual), len(expected), '%s != %s' % (actual, expected))
        for i, result in enumerate(expected):
            if type(result) == type([]):
                self.checkResultsList(actual[i], result)
            else:
                self.assertEqual(actual[i], result, '%s != %s' % (actual[i], result))
    
    def assertParses(self, grammar, input, results):
        """Utility method to test that a given grammar parses an input as expected."""
        actual_results = grammar.parseString(input, parseAll=True)
        self.checkResultsList(actual_results, results)
    
    def failIfParses(self, grammar, input):
        """Utility method to test that a given grammar fails to parse an input."""
        strict_grammar = grammar + strict_string_end
        self.assertRaises(CSP.p.ParseException, strict_grammar.parseString, input)
        
    def TestParsingIdentifiers(self):
        self.assertParses(csp.ncIdent, 'abc', ['abc'])
        self.failIfParses(csp.ncIdent, 'abc:def')
        self.failIfParses(csp.ncIdent, '123')

        self.assertParses(csp.ident, 'abc', ['abc'])
        self.assertParses(csp.ident, 'abc_def', ['abc_def'])
        self.assertParses(csp.ident, 'abc123', ['abc123'])
        self.assertParses(csp.ident, '_abc', ['_abc'])
        self.assertParses(csp.ident, 'abc:def', ['abc:def'])
        self.failIfParses(csp.ident, '123')
    
    def TestParsingNumbers(self):
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
        self.failIfParses(csp.number, '.123')
        self.failIfParses(csp.number, '123.')
        self.failIfParses(csp.number, '+123')
        self.failIfParses(csp.number, '1E3')
    
    def TestParsingComments(self):
        self.assertParses(csp.comment, '# blah blah', [])
        self.failIfParses(csp.comment, '# blah blah\n')
        
    def TestParsingSimpleExpressions(self):
        self.assertParses(csp.expr, '1', [['1']])
        self.assertParses(csp.expr, '(A)', [['A']])
        self.assertParses(csp.expr, '1 - 2 + 3 * 4 / 5 ^ 6', [['1', '-', '2', '+',
                                                               ['3', '*', '4', '/', ['5', '^', '6']]]])
        self.assertParses(csp.expr, '-(a + -3)', [['-', ['a', '+', ['-', '3']]]])
        self.assertParses(csp.expr, '1 ^ 2 ^ 3', [['1', '^', '2', '^', '3']])
        self.assertParses(csp.expr, 'not 1 < 2', [[['not', '1'], '<', '2']])
        self.assertParses(csp.expr, 'not (1 < 2)', [['not', ['1', '<', '2']]])
        self.assertParses(csp.expr, 'not not my:var', [['not', ['not', 'my:var']]])
        self.assertParses(csp.expr, '1 < 2 <= (3 >= 4) > 5 == 6 != 7', [['1', '<', '2', '<=', ['3', '>=', '4'],
                                                                         '>', '5', '==', '6', '!=', '7']])
        self.assertParses(csp.expr, '1 < 2 + 4 > 3', [['1', '<', ['2', '+', '4'], '>', '3']])
        self.assertParses(csp.expr, '1 < 2 && 4 > 3', [[['1', '<', '2'], '&&', ['4', '>', '3']]])
        self.assertParses(csp.expr, '0 || 1 && 1', [['0', '||', '1', '&&', '1']])
        self.assertParses(csp.expr, 'A + B || C * D', [[['A', '+', 'B'], '||', ['C', '*', 'D']]])
        self.assertParses(csp.expr, 'if 1 then 2 else 3', [[['1'], ['2'], ['3']]])
        self.assertParses(csp.expr, 'if 1 < 2 then 3 + 4 else 5 * 6',
                          [[['1', '<', '2'], ['3', '+', '4'], ['5', '*', '6']]])
    
    def TestParsingMultiLineExpressions(self):
        self.assertParses(csp.expr, '(1 + 2) * 3', [[['1', '+', '2'], '*', '3']])
        self.assertParses(csp.expr, '((1 + 2)\\\n * 3)', [[['1', '+', '2'], '*', '3']])
        self.assertParses(csp.expr, '(1 + 2)\\\n * 3', [[['1', '+', '2'], '*', '3']])
        self.assertParses(csp.expr, '((1 + 2)\n  * 3)', [[['1', '+', '2'], '*', '3']])
        self.assertParses(csp.expr, '((1 + 2)#embedded comment\n  * 3)', [[['1', '+', '2'], '*', '3']])
        # This one doesn't match Python behaviour, but it's hard to stop it parsing while allowing
        # the above to work.
        self.assertParses(csp.expr, '(1 + 2)\n * 3', [[['1', '+', '2'], '*', '3']])
    
    def TestParsingSimpleAssignments(self):
        self.assertParses(csp.simpleAssign, 'var = value', [['var', 'value']])
        self.assertParses(csp.simpleAssign, 'var = pre:value', [['var', 'pre:value']])
        self.failIfParses(csp.simpleAssign, 'pre:var = value')
        self.assertParses(csp.simpleAssign, 'var = 1 + 2', [['var', ['1', '+', '2']]])

        self.assertParses(csp.simpleAssignList, 'v1 = 1\nv2=2', [['v1', '1'], ['v2', '2']])
        self.assertParses(csp.simpleAssignList, '', [])
    
    def TestParsingNamespaces(self):
        self.assertParses(csp.nsDecl, 'namespace prefix = "urn:test"', [['prefix', 'urn:test']])
        self.assertParses(csp.nsDecl, "namespace prefix='urn:test'", [['prefix', 'urn:test']])
        self.assertParses(csp.nsDecls, 'namespace n1="urn:t1"#test ns\nnamespace n2 = "urn:t2"',
                          [['n1', 'urn:t1'], ['n2', 'urn:t2']])
        self.assertParses(csp.nsDecls, '', [])
    
    def TestParsingInputs(self):
        '''Test parsing protocol input declarations'''
        self.assertParses(csp.inputs, """inputs {
    label_duration = 300  # How long to label for
    unlabel_time = 600000 # When to stop labelling
    use_tapasin = 1       # Whether to include Tapasin
}
""", [['label_duration', '300'], ['unlabel_time', '600000'], ['use_tapasin', '1']])
        self.assertParses(csp.inputs, 'inputs {}', [])
        self.assertParses(csp.inputs, 'inputs\n{\n}\n', [])
        self.assertParses(csp.inputs, 'inputs{X=1}', [['X', '1']])
    
    def TestParsingImports(self):
        self.assertParses(csp.importStmt, 'import std = "../../../src/proto/library/BasicLibrary.xml"',
                          [['std', '../../../src/proto/library/BasicLibrary.xml']])
        self.assertParses(csp.importStmt, "import 'TestS1S2.xml'", [['', 'TestS1S2.xml']])
        self.assertParses(csp.imports, 'import l1="file1"#blah\nimport "file2"',
                          [['l1', 'file1'], ['', 'file2']])
        self.assertParses(csp.imports, '', [])
    
    def TestParsingModelInterface(self):
        self.assertParses(csp.setTimeUnits, 'independent var units u', ['u'])
        
        self.assertParses(csp.inputVariable, 'input test:var units u = 1.2', [['test:var', 'u', '1.2']])
        self.assertParses(csp.inputVariable, 'input test:var units u', [['test:var', 'u', '']])
        self.assertParses(csp.inputVariable, 'input test:var = -1e2', [['test:var', '', '-1e2']])
        self.assertParses(csp.inputVariable, 'input test:var', [['test:var', '', '']])
        
        self.assertParses(csp.outputVariable, 'output test:var', [['test:var', '']])
        self.assertParses(csp.outputVariable, 'output test:var units uname', [['test:var', 'uname']])
        
        self.assertParses(csp.newVariable, 'var varname units uname = 0', [['varname', 'uname', '0']])
        self.assertParses(csp.newVariable, 'var varname units uname', [['varname', 'uname', '']])
        self.failIfParses(csp.newVariable, 'var prefix:varname units uname = 0')
        self.failIfParses(csp.newVariable, 'var varname = 0')
        self.failIfParses(csp.newVariable, 'var varname')
        
        self.assertParses(csp.modelEquation, 'define local_var = 1 + model:var',
                          [['local_var', ['1', '+', 'model:var']]])
        self.assertParses(csp.modelEquation, 'define model:var = 2.5 / local_var',
                          [['model:var', ['2.5', '/', 'local_var']]])
        
        self.assertParses(csp.unitsConversion, 'convert uname1 to uname2 by lambda u: u / model:var',
                          [['uname1', 'uname2', [[['u']], ['u', '/', 'model:var']]]])

        self.assertParses(csp.modelInterface, """model interface {  # Comments can go here
    independent var units t
    
    input test:v1 = 0  # a comment
    input test:v2 units u
    output test:time
    # comments are always ignored
    output test:v3 units u
    var local units dimensionless = 5
    define test:v3 = test:v2 * local
    convert u1 to u2 by lambda u: u * test:v3
}""", [['t', ['test:v1', '', '0'], ['test:v2', 'u', ''], ['test:time', ''], ['test:v3', 'u'],
        ['local', 'dimensionless', '5'], ['test:v3', ['test:v2', '*', 'local']],
        ['u1', 'u2', [[['u']], ['u', '*', 'test:v3']]]]])
        self.assertParses(csp.modelInterface, 'model interface {}', [[]])
        self.assertParses(csp.modelInterface, 'model interface#comment\n{output test:time\n}', [[['test:time', '']]])
        self.assertParses(csp.modelInterface, 'model interface {output test:time }', [[['test:time', '']]])

    def TestParsingUniformRange(self):
        self.assertParses(csp.range, 'range time units ms uniform 0:1:1000', [['time', 'ms', ['0', '1', '1000']]])
        self.assertParses(csp.range, 'range time units ms uniform 0:1000', [['time', 'ms', ['0', '1000']]])

#    def TestParsingVectorRange(self):
#        self.assertParses(csp.range, 'range run units dimensionless vector [1, 2, 3, 4]',
#                          [['run', 'dimensionless', ['1', '2', '3', '4']]])

    def TestParsingWhileRange(self):
        self.assertParses(csp.range, 'range rpt units dimensionless while rpt < 5',
                          [['rpt', 'dimensionless', ['rpt', '<', '5']]])

    def TestParsingModifiers(self):
        self.assertParses(csp.modifierWhen, 'at start', ['start'])
        self.assertParses(csp.modifierWhen, 'at each loop', ['each'])
        self.assertParses(csp.modifierWhen, 'at end', ['end'])
        
        self.assertParses(csp.setVariable, 'set model:V = 5.0', ['model:V', '5.0'])
        self.assertParses(csp.setVariable, 'set model:t = time + 10.0', ['model:t', ['time', '+', '10.0']])
        
        self.assertParses(csp.saveState, 'save as state_name', ['state_name'])
        self.failIfParses(csp.saveState, 'save as state:name')
        
        self.assertParses(csp.resetState, 'reset', [])
        self.assertParses(csp.resetState, 'reset to state_name', ['state_name'])
        self.failIfParses(csp.resetState, 'reset to state:name')
        
        self.assertParses(csp.modifiers, """modifiers {
        # Multiple
        # comment lines
        # are OK
        at start reset
        at each loop set model:input = loopVariable
        
        # Blank lines OK too
        at end save as savedState
} # Trailing comments are fine""",
                          [[['start', []], ['each', ['model:input', 'loopVariable']], ['end', ['savedState']]]])
        self.assertParses(csp.modifiers, 'modifiers {at start reset}', [[['start', []]]])
    
    def TestParsingTimecourseSimulations(self):
        self.assertParses(csp.simulation, 'simulation sim = timecourse { range time units ms uniform 1:10 }',
                          ['sim', [['time', 'ms', ['1', '10']]]])
        self.assertParses(csp.simulation, 'simulation timecourse #c\n#c\n{\n range time units ms uniform 1:10\n\n }#c',
                          ['', [['time', 'ms', ['1', '10']]]])
        self.assertParses(csp.simulation, """simulation sim = timecourse {
range time units U while time < 100
modifiers { at end save as prelim }
}""",
                          ['sim', [['time', 'U', ['time', '<', '100']], [['end', ['prelim']]]]])
        self.failIfParses(csp.simulation, 'simulation sim = timecourse {}')
    
    def TestParsingNestedSimulations(self):
        self.assertParses(csp.simulation,
                          'simulation rpt = nested { range run units U while not rpt:result\n nests sim }',
                          ['rpt', [['run', 'U', ['not', 'rpt:result']], ['sim']]])
        self.assertParses(csp.simulation, """simulation nested {
range R units U uniform 3:5
modifiers { at each loop reset to prelim }
nests sim
}""",
                          ['', [['R', 'U', ['3', '5']], [['each', ['prelim']]], ['sim']]])
        self.failIfParses(csp.simulation, 'simulation rpt = nested { range run units U while 1 }')
    
    def TestParsingOutputSpecifications(self):
        self.assertParses(csp.outputSpec, 'name = model:var "Description"', [['name', 'model:var', '', 'Description']])
        self.assertParses(csp.outputSpec, r'name = ref:var units U "Description \"quotes\""',
                          [['name', 'ref:var', 'U', 'Description "quotes"']])
        self.assertParses(csp.outputSpec, "name = ref:var units U 'Description \\'quotes\\' \"too\"'",
                          [['name', 'ref:var', 'U', 'Description \'quotes\' "too"']])
        self.assertParses(csp.outputSpec, 'varname units UU', [['varname', 'UU', '']])
        self.assertParses(csp.outputSpec, 'varname units UU "desc"', [['varname', 'UU', 'desc']])
        self.failIfParses(csp.outputSpec, 'varname_no_units')
        
        self.assertParses(csp.outputs, """outputs #cccc
{ #cdc
        n1 = n2 units u1
        n3 = p:m 'd1'
        n4 units u2 "d2"
} #cpc
""", [[['n1', 'n2', 'u1', ''], ['n3', 'p:m', '', 'd1'], ['n4', 'u2', 'd2']]])
        self.assertParses(csp.outputs, "outputs {}", [[]])
    
    def TestParsingPlotSpecifications(self):
        self.assertParses(csp.plotCurve, 'y against x', [['y', 'x']])
        self.assertParses(csp.plotCurve, 'y, y2 against x', [['y', 'y2', 'x']])
        self.failIfParses(csp.plotCurve, 'm:y against x')
        self.failIfParses(csp.plotCurve, 'y against m:x')
        self.assertParses(csp.plotSpec, 'plot "A title\'s good" { y1, y2 against x1\n y3 against x2 }',
                          [["A title's good", ['y1', 'y2', 'x1'], ['y3', 'x2']]])
        self.failIfParses(csp.plotSpec, 'plot "only title" {}')
        self.failIfParses(csp.plotSpec, 'plot "only title"')
        
        self.assertParses(csp.plots, """plots { plot "t1" { v1 against v2 }
        plot "t1" { v3, v4 against v5 }
}""", [[['t1', ['v1', 'v2']], ['t1', ['v3', 'v4', 'v5']]]])
        self.assertParses(csp.plots, 'plots {}', [[]])
    
    def TestParsingFunctionCalls(self):
        self.assertParses(csp.functionCall, 'noargs()', [['noargs', []]])
        self.assertParses(csp.functionCall, 'swap(a, b)', [['swap', ['a', 'b']]])
        self.assertParses(csp.functionCall, 'double(33)', [['double', ['33']]])
        self.assertParses(csp.functionCall, 'double(a + b)', [['double', [['a', '+', 'b']]]])
        self.assertParses(csp.functionCall, 'std:max(A)', [['std:max', ['A']]])
        self.failIfParses(csp.functionCall, 'spaced (param)')
        self.assertParses(csp.expr, 'func(a,b, 3)', [['func', ['a', 'b', '3']]])
    
    def TestParsingMathmlOperators(self):
        # MathML that doesn't have a special operator is represented as a normal function call,
        # with the 'magic' MathML: prefix.
        self.assertEqual(len(csp.mathmlOperators), 12 + 3*8)
        for trigbase in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']:
            self.assert_(trigbase in csp.mathmlOperators)
            self.assert_(trigbase + 'h' in csp.mathmlOperators)
            self.assert_('arc' + trigbase in csp.mathmlOperators)
            self.assert_('arc' + trigbase + 'h' in csp.mathmlOperators)
        for op in 'quotient rem max min root xor abs floor ceiling exp ln log'.split():
            self.assert_(op in csp.mathmlOperators)
        self.assertParses(csp.expr, 'MathML:exp(MathML:floor(MathML:exponentiale))',
                          [['MathML:exp', [['MathML:floor', ['MathML:exponentiale']]]]])
    
    def TestParsingAssignStatements(self):
        self.assertParses(csp.assignStmt, 'var = value', [[['var'], ['value']]])
        self.assertParses(csp.assignStmt, 'var = pre:value', [[['var'], ['pre:value']]])
        self.failIfParses(csp.assignStmt, 'pre:var = value')
        self.assertParses(csp.assignStmt, 'var = 1 + 2', [[['var'], [['1', '+', '2']]]])

        self.assertParses(csp.assignStmt, 'a, b = tuple', [[['a', 'b'], ['tuple']]])
        self.assertParses(csp.assignStmt, 'a, b = b, a', [[['a', 'b'], ['b', 'a']]])
        self.assertParses(csp.assignStmt, 'a, b = (b, a)', [[['a', 'b'], [['b', 'a']]]]) # TODO: same nesting as previous case
        self.failIfParses(csp.assignStmt, 'p:a, p:b = e')
        self.failIfParses(csp.assignStmt, '')
    
    def TestParsingReturnStatements(self):
        self.assertParses(csp.returnStmt, 'return 2 * a', [[['2', '*', 'a']]])
        self.assertParses(csp.returnStmt, 'return (3 - 4)', [[['3', '-', '4']]])
        self.assertParses(csp.returnStmt, 'return a, b', [['a', 'b']])
        self.assertParses(csp.returnStmt, 'return a + 1, b - 1', [[['a', '+', '1'], ['b', '-', '1']]])
        self.assertParses(csp.returnStmt, 'return (a, b)', [[['a', 'b']]])
    
    def TestParsingAssertStatements(self):
        self.assertParses(csp.assertStmt, 'assert a + b', [[['a', '+', 'b']]])
        self.assertParses(csp.assertStmt, 'assert (a + b)', [[['a', '+', 'b']]])
        self.assertParses(csp.assertStmt, 'assert 1', [[['1']]])
    
    def TestParsingStatementLists(self):
        self.assertParses(csp.stmtList, "b=-a\nassert 1", [[['b'], [['-', 'a']]], ['1']])
        self.assertParses(csp.stmtList, """assert a < 0 # comments are ok

# as are blank lines
b = -a
assert b > 0
c, d = a * 2, b + 1
return c, d""", [[['a', '<', '0']],
                 [['b'], [['-', 'a']]],
                 [['b', '>', '0']],
                 [['c', 'd'], [['a', '*', '2'], ['b', '+', '1']]],
                 ['c', 'd']])
        self.failIfParses(csp.stmtList, '')
    
    def TestParsingLambdaExpressions(self):
        self.assertParses(csp.lambdaExpr, 'lambda a: a + 1', [[[['a']], ['a', '+', '1']]])
        self.assertParses(csp.lambdaExpr, 'lambda a, b: a + b', [[[['a'], ['b']], ['a', '+', 'b']]])
        self.assertParses(csp.lambdaExpr, 'lambda a, b=2: a - b', [[[['a'], ['b', '2']], ['a', '-', 'b']]])
        self.assertParses(csp.expr, 'lambda a=c, b: a * b', [[[['a', 'c'], ['b']], ['a', '*', 'b']]])
        self.assertParses(csp.lambdaExpr, 'lambda a=p:c, b: a * b', [[[['a', 'p:c'], ['b']], ['a', '*', 'b']]])
        self.failIfParses(csp.lambdaExpr, 'lambda p:a: 5')

        self.assertParses(csp.lambdaExpr, """lambda a, b {
assert a > b
c = a - b
return c
}
""", [[[['a'], ['b']], [['a', '>', 'b']],
                       [['c'], [['a', '-', 'b']]],
                       ['c']]])
        self.assertParses(csp.expr, "lambda a { return a }", [[[['a']], ['a']]])

    def TestParsingFunctionDefinitions(self):
        self.assertParses(csp.functionDefn, 'def double(a)\n {\n return a * 2\n }',
                          [['double', [['a']], [['a', '*', '2']]]])
        self.assertParses(csp.functionDefn, 'def double(a): a * 2',
                          [['double', [['a']], ['a', '*', '2']]])
        # A function definition is just sugar for an assignment of a lambda expression
        self.assertParses(csp.stmtList, 'def double(a) {\n    return a * 2}',
                          [['double', [['a']], [['a', '*', '2']]]])
    
    def TestParsingNestedFunctions(self):
        # Currently we can't do this, as we don't delimit the contained statement list in any way!
        # Need to choose between indentation level (Python) or brace-delimited
        pass
    
    def TestParsingTuples(self):
        self.assertParses(csp.tuple, '(1,2)', [['1', '2']])
        self.assertParses(csp.tuple, '(1+a,2*b)', [[['1', '+', 'a'], ['2', '*', 'b']]])
        self.assertParses(csp.tuple, '(singleton,)', [['singleton']])
        self.failIfParses(csp.tuple, '(1)') # You need a Python-style comma as above
        self.assertParses(csp.expr, '(1,2)', [['1', '2']])
        self.assertParses(csp.expr, '(1,a,3,c)', [['1', 'a', '3', 'c']])
        self.assertParses(csp.assignStmt, 't = (1,2)', [[['t'], [['1', '2']]]])
        self.assertParses(csp.assignStmt, 'a, b = (1,2)', [[['a', 'b'], [['1', '2']]]])
    
    def TestParsingArrays(self):
        self.assertParses(csp.expr, '[1, 2, 3]', [['1', '2', '3']])
        self.assertParses(csp.array, '[[a, b], [c, d]]', [[['a', 'b'], ['c', 'd']]])
        self.assertParses(csp.array, '[ [ [1+2,a,b]],[[3/4,c,d] ]]', [[[[['1', '+', '2'],'a','b']],
                                                                        [[['3', '/', '4'],'c','d']]]])

    def TestParsingArrayComprehensions(self):
        self.assertParses(csp.array, '[i for i in 0:N]', [['i', ['i', '0', 'N']]])
        self.assertParses(csp.expr, '[i*2 for i in 0:2:4]', [[['i', '*', '2'], ['i', '0', '2', '4']]])
        self.assertParses(csp.array, '[i+j*5 for i in 1:3 for j in 2:4]',
                          [[['i', '+', ['j', '*', '5']], ['i', '1', '3'], ['j', '2', '4']]])
        self.assertParses(csp.array, '[block for 1$i in 2:10]', [['block', ['1', 'i', '2', '10']]])
        self.assertParses(csp.array, '[i^j for i in 1:3 for 2$j in 4:-1:2]',
                          [[['i', '^', 'j'], ['i', '1', '3'], ['2', 'j', '4', ['-', '1'], '2']]])
        # Dimension specifiers can be expressions too...
        self.assertParses(csp.expr, '[i for (1+2)$i in 2:(3+5)]', [['i', [['1', '+', '2'], 'i', '2', ['3', '+', '5']]]])
        self.assertParses(csp.expr, '[i for 1+2$i in 2:4]', [['i', [['1', '+', '2'], 'i', '2', '4']]])
        self.failIfParses(csp.expr, '[i for 1 $i in 2:4]')
    
    def TestParsingViews(self):
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
        self.assertParses(csp.expr, 'genericity[*$0:5:50]', [['genericity', ['*', '0', '5', '50']]])
        self.assertParses(csp.expr, 'genericity[*$:5:]', [['genericity', ['*', '', '5', '']]])
        self.assertParses(csp.expr, 'genericity[*$0:]', [['genericity', ['*', '0', '']]])
        self.assertParses(csp.expr, 'multiples[3][4]', [['multiples', ['3'], ['4']]])
        self.assertParses(csp.expr, 'multiples[1$3][0$:-step:0][*$0]',
                          [['multiples', ['1', '3'], ['0', '', ['-', 'step'], '0'], ['*', '0']]])
        self.assertParses(csp.expr, 'dimspec[dim$0:2]', [['dimspec', ['dim', '0', '2']]])
        self.assertParses(csp.expr, 'okspace[ 0$ (1+2) : a+b : 50 ]',
                          [['okspace', ['0', ['1', '+', '2'], ['a', '+', 'b'], '50']]])
        # Some spaces aren't allowed
        self.failIfParses(csp.expr, 'arr [1]')
        self.failIfParses(csp.expr, 'arr[1] [3]')
        self.failIfParses(csp.expr, 'arr[1 $ 2]')
    
    def TestParsingFindAndIndex(self):
        # Curly braces represent index, with optional pad or shrink argument.  Find is a function call.
        self.assertParses(csp.expr, 'find(arr)', [['find', ['arr']]])
        self.failIfParses(csp.expr, 'find (arr)')
        #self.failIfParses(csp.expr, 'find(arr, extra)') # Needs special support e.g. from parse actions
        
        self.assertParses(csp.expr, 'arr{idxs}', [['arr', ['idxs']]])
        self.failIfParses(csp.expr, 'arr {spaced}')
        self.assertParses(csp.expr, 'arr{idxs, shrink:dim}', [['arr', ['idxs', 'dim']]])
        self.assertParses(csp.expr, 'arr{idxs, pad:dim=value}', [['arr', ['idxs', 'dim', 'value']]])
        self.failIfParses(csp.expr, 'arr{idxs, shrink:dim, pad:dim=value}')
        
        self.assertParses(csp.expr, 'f(1,2){find(blah), shrink:0}', [[['f', ['1', '2']], [['find', ['blah']], '0']]])
        self.assertParses(csp.expr, 'A{find(A), pad:0=1+2}', [['A', [['find', ['A']], '0', ['1', '+', '2']]]])

    def TestParsingUnitsDefinitions(self):
        # Possible syntax:  (mult, offset, expt are 'numbers'; prefix is SI prefix name; base is ncIdent)
        #  new_simple = [mult] [prefix] base [+|- offset]
        #  new_complex = p.delimitedList( [mult] [prefix] base [^expt], '.')
        self.assertParses(csp.unitsDef, 'ms = milli second', [['ms', ['1', 'milli', 'second', '1']]])
        self.assertParses(csp.unitsDef, 'C = kelvin - 273.15', [['C', ['1', '', 'kelvin', '1', ['-', '273.15']]]])
        self.assertParses(csp.unitsDef, 'C=kelvin-273.15', [['C', ['1', '', 'kelvin', '1', ['-', '273.15']]]])
        self.assertParses(csp.unitsDef, 'litre = 1000 centi metre^3', [['litre', ['1000', 'centi', 'metre', '3']]])
        self.assertParses(csp.unitsDef, 'accel_units = kilo metre . second^-2',
                          [['accel_units', ['1', 'kilo', 'metre', '1'], ['1', '', 'second', '-2']]])
        self.assertParses(csp.unitsDef, 'fahrenheit = (5/9) celsius + 32.0',
                          [['fahrenheit', [['5', '/', '9'], '', 'celsius', '1', ['+', '32.0']]]])
        self.assertParses(csp.unitsDef, 'fahrenheit = (5/9) kelvin + (32 - 273.15 * 9 / 5)',
                          [['fahrenheit', [['5', '/', '9'], '', 'kelvin', '1', ['+', ['32', '-', ['273.15', '*', '9', '/', '5']]]]]])
        
        self.assertParses(csp.units, "units {}", [])
        self.assertParses(csp.units, """units
{
# nM = nanomolar
nM = nano mole . litre^-1
hour = 3600 second
flux = nM . hour ^ -1

rate_const = hour^-1           # First order
rate_const_2 = nM^-1 . hour^-1 # Second order
}
""", [['nM', ['1', 'nano', 'mole', '1'], ['1', '', 'litre', '-1']],
      ['hour', ['3600', '', 'second', '1']],
      ['flux', ['1', '', 'nM', '1'], ['1', '', 'hour', '-1']],
      ['rate_const', ['1', '', 'hour', '-1']],
      ['rate_const_2', ['1', '', 'nM', '-1'], ['1', '', 'hour', '-1']]])
    
    def TestParsingAccessors(self):
        for accessor in ['NUM_DIMS', 'SHAPE', 'NUM_ELEMENTS']:
            self.assertParses(csp.accessor, '.' + accessor, [accessor])
            self.assertParses(csp.expr, 'var.' + accessor, [['var', accessor]])
        for ptype in ['SIMPLE_VALUE', 'ARRAY', 'STRING', 'TUPLE', 'FUNCTION', 'NULL', 'DEFAULT']:
            self.assertParses(csp.accessor, '.IS_' + ptype, ['IS_' + ptype])
            self.assertParses(csp.expr, 'var.IS_' + ptype, [['var', 'IS_' + ptype]])
        self.assertParses(csp.expr, 'arr.SHAPE[1]', [[['arr', 'SHAPE'], ['1']]])
        self.assertParses(csp.expr, 'func(var).IS_ARRAY', [[['func', ['var']], 'IS_ARRAY']])
        self.assertParses(csp.expr, 'A.SHAPE.IS_ARRAY', [['A', 'SHAPE', 'IS_ARRAY']])
        self.failIfParses(csp.expr, 'arr .SHAPE')
    
    def TestParsingMap(self):
        self.assertParses(csp.expr, 'map(func, a1, a2)', [['map', ['func', 'a1', 'a2']]])
        self.assertParses(csp.expr, 'map(lambda a, b: a+b, A, B)',
                          [['map', [[[['a'], ['b']], ['a', '+', 'b']], 'A', 'B']]])
        self.assertParses(csp.expr, 'map(id, a)', [['map', ['id', 'a']]])
        self.assertParses(csp.expr, 'map(hof(arg), a, b, c, d, e)',
                          [['map', [['hof', ['arg']], 'a', 'b', 'c', 'd', 'e']]])
        # self.failIfParses(csp.expr, 'map(f)') # At present implemented just as a function call with special name
    
    def TestParsingFold(self):
        self.assertParses(csp.expr, 'fold(func, array, init, dim)', [['fold', ['func', 'array', 'init', 'dim']]])
        self.assertParses(csp.expr, 'fold(lambda a, b: a - b, f(), 1, 2)',
                          [['fold', [[[['a'], ['b']], ['a', '-', 'b']], ['f', []], '1', '2']]])
        self.assertParses(csp.expr, 'fold(f, A)', [['fold', ['f', 'A']]])
        self.assertParses(csp.expr, 'fold(f, A, 0)', [['fold', ['f', 'A', '0']]])
        self.assertParses(csp.expr, 'fold(f, A, default, 1)', [['fold', ['f', 'A', [], '1']]])
        #self.failIfParses(csp, expr, 'fold()')
        #self.failIfParses(csp, expr, 'fold(f, A, i, d, extra)')

    def TestParsingWrappedMathmlOperators(self):
        self.assertParses(csp.expr, '@3:+', [['3', '+']])
        self.assertParses(csp.expr, '@1:MathML.sin', [['1', 'MathML.sin']])
        self.assertParses(csp.expr, 'map(@2:/, a, b)', [['map', [['2', '/'], 'a', 'b']]])
        #self.failIfParses(csp.expr, '@0:+') # Best done at parse action level?
        self.failIfParses(csp.expr, '@1:non_mathml')
        self.failIfParses(csp.expr, '@ 2:*')
        self.failIfParses(csp.expr, '@2 :-')
        self.failIfParses(csp.expr, '@1:--')
        self.failIfParses(csp.expr, '@N:+')

    def TestParsingNullAndDefault(self):
        self.assertParses(csp.expr, 'null', [[]])
        self.assertParses(csp.expr, 'default', [[]])

    def TestParsingLibrary(self):
        self.assertParses(csp.library, 'library {}', [])
        self.assertParses(csp.library, """library
{
    def f(a) {
        return a
    }
    f2 = lambda b: b/2
    const = 13
}
""", [['f', [['a']], ['a']],
      [['f2'], [[[['b']], ['b', '/', '2']]]],
      [['const'], ['13']]])
    
    def TestParsingUseImports(self):
        pass
    
    def TestParsingFullProtocols(self):
        # I won't compare against expected values for these at this stage!  Eventually we could compare against the XML versions.
        pass
