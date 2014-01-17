
"""Copyright (c) 2005-2014, University of Oxford.
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

import difflib
import filecmp
import glob
import itertools
import os
import unittest
import sys
import time

# Import the module to test
import CompactSyntaxParser as CSP
CSP.DoXmlImports() # The default for this module now is to assume the Python implementation

csp = CSP.CompactSyntaxParser
# An end-of-string match that doesn't allow trailing whitespace
strict_string_end = CSP.p.StringEnd().leaveWhitespace()

# For debugging and error messages
def X2S(xml):
    """Serialize XML to a compact string."""
    return CSP.ET.tostring(xml, pretty_print=False)

def WithUnits(base, units):
    """Shorthand for generating the test case for a number with units."""
    return (base, {'{%s}units' % CSP.CELLML_NS: units})    

class TestCompactSyntaxParser(unittest.TestCase):
    def checkParseResults(self, actual, expected):
        """Compare parse results to expected strings.
        
        The expected results may be given as a (nested) list or a dictionary, depending
        on whether you want to compare against matched tokens by order or results name.
        """
        def check_result(actual, expected):
            if isinstance(expected, str):
                self.assertEqual(actual, expected, '%s != %s' % (actual, expected))
            else:
                self.checkParseResults(actual, expected)
        if isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), '%s != %s' % (actual, expected))
            for i, result in enumerate(expected):
                check_result(actual[i], result)
        elif isinstance(expected, dict):
            self.assertEqual(len(actual.keys()), len(expected), '%s != %s' % (actual, expected))
            for key, value in expected.iteritems():
                check_result(actual[key], value)
    
    def assertHasLocalName(self, xmlElement, localName):
        ending = '}' + localName
        self.assertEqual(xmlElement.tag[-len(ending):], ending, '%s does not have local name %s' % (xmlElement.tag, localName))
    
    def checkXml(self, actualXml, expectedXml, checkEmpty=True):
        if isinstance(expectedXml, str):
            if ':' in expectedXml:
                expectedXml, content = expectedXml.split(':', 1)
                self.assertEqual(actualXml.text, content)
            if expectedXml.startswith('csymbol-'):
                self.assertHasLocalName(actualXml, 'csymbol')
                expectedXml = ('csymbol', {'definitionURL': 'https://chaste.cs.ox.ac.uk/nss/protocol/'+expectedXml[8:]})
            else:
                if checkEmpty:
                    self.assertEqual(len(actualXml), 0, '%s has unexpected children' % X2S(actualXml))
                self.assertHasLocalName(actualXml, expectedXml)
        if isinstance(expectedXml, tuple):
            self.checkXml(actualXml, expectedXml[0], checkEmpty=False)
            for children_or_attrs in expectedXml[1:]:
                if isinstance(children_or_attrs, list): # Child elements
                    self.assertEqual(len(actualXml), len(children_or_attrs), '%s != %s' % (X2S(actualXml), children_or_attrs))
                    for i, expected_child in enumerate(children_or_attrs):
                        self.checkXml(actualXml[i], expected_child)
                elif isinstance(children_or_attrs, dict): # Attributes
                    actual_attrs = actualXml.attrib
                    actual_attrs.pop('{%s}loc' % CSP.PROTO_NS, None) # Remove loc attribute if present
                    self.assertEqual(len(actual_attrs), len(children_or_attrs), '%s != %s' % (actual_attrs, children_or_attrs))
                    self.assertEqual(actual_attrs, children_or_attrs)
                else:
                    self.fail('Bad test case')
    
    def assertParses(self, grammar, input, results, expectedXml=None):
        """Utility method to test that a given grammar parses an input as expected."""
        actual_results = grammar.parseString(input, parseAll=True)
        if expectedXml:
            self.assert_(len(actual_results) > 0, 'No results available')
            self.assert_(hasattr(actual_results[0], 'xml'), 'No XML available')
            self.assert_(callable(actual_results[0].xml), 'No XML available')
            actual_xml = actual_results[0].xml()
#            print X2S(actual_xml)
            self.checkXml(actual_xml, expectedXml)
        self.checkParseResults(actual_results, results)
    
    def failIfParses(self, grammar, input):
        """Utility method to test that a given grammar fails to parse an input."""
        strict_grammar = grammar + strict_string_end
        self.assertRaises(CSP.p.ParseBaseException, strict_grammar.parseString, input)
    
    def assertXmlEqual(self, newElement, refElement):
        """Utility method for comparing two XML elements."""
        self.assertEqual(newElement.tag, refElement.tag)
        def Strip(text):
            if isinstance(text, str):
                text = text.strip()
            return text or None
        self.assertEqual(len(newElement), len(refElement))
        self.assertEqual(Strip(newElement.text), Strip(refElement.text))
        self.assertEqual(Strip(newElement.tail), Strip(refElement.tail))
        # Remove some attributes that we shouldn't compare, if present
        for attr_name in ['{%s}loc' % CSP.PROTO_NS, '{http://www.w3.org/XML/1998/namespace}base']:
            for elt in [newElement, refElement]:
                elt.attrib.pop(attr_name, None)
        self.assertEqual(sorted(newElement.attrib.items()), sorted(refElement.attrib.items()))
        for newChild, refChild in itertools.izip(newElement, refElement):
            self.assertXmlEqual(newChild, refChild)
    
    def assertFilesMatch(self, newFilePath, refFilePath):
        """Utility method to check that two files have matching contents."""
        if not filecmp.cmp(newFilePath, refFilePath):
            # Matching failed, so print something informative
            context_lines = 3
            from_date = time.ctime(os.stat(refFilePath).st_mtime)
            to_date = time.ctime(os.stat(newFilePath).st_mtime)
            for line in difflib.unified_diff(open(refFilePath).readlines(), open(newFilePath).readlines(),
                                             refFilePath, newFilePath,
                                             from_date, to_date, n=context_lines):
                print line,
            self.fail("Output file '%s' does not match reference file '%s'" % (newFilePath, refFilePath))
        
    def TestParsingIdentifiers(self):
        self.assertParses(csp.ncIdentAsVar, 'abc', ['abc'], 'ci:abc')
        self.failIfParses(csp.ncIdent, 'abc:def')
        self.failIfParses(csp.ncIdent, '123')

        self.assertParses(csp.identAsVar, 'abc', ['abc'], 'ci:abc')
        self.assertParses(csp.ident, 'abc_def', ['abc_def'])
        self.assertParses(csp.ident, 'abc123', ['abc123'])
        self.assertParses(csp.ident, '_abc', ['_abc'])
        self.assertParses(csp.identAsVar, 'abc:def', ['abc:def'], 'ci:abc:def')
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
    
    def TestParsingNumbersWithUnits(self):
        self.assertParses(csp.number, '123 :: dimensionless', ['123', 'dimensionless'])
        self.assertParses(csp.number, '-4e5::mA', ['-4e5', 'mA'])
        self.assertParses(csp.expr, '123 :: dimensionless', ['123'], WithUnits('cn:123', 'dimensionless'))
        self.assertParses(csp.expr, '4.6e5::mA', ['4.6e5'], WithUnits('cn:4.6e5', 'mA'))
        self.failIfParses(csp.number, '123 :: ')
        self.failIfParses(csp.number, '123 :: prefix:units')
    
    def TestParsingComments(self):
        self.assertParses(csp.comment, '# blah blah', [])
        self.failIfParses(csp.comment, '# blah blah\n')
        
    def TestParsingSimpleExpressions(self):
        self.assertParses(csp.expr, '1', ['1'], 'cn')
        self.assertParses(csp.expr, '(A)', ['A'], 'ci')
        self.assertParses(csp.expr, '1 - 2 + 3 * 4 / 5 ^ 6',
                          [['1', '-', '2', '+', ['3', '*', '4', '/', ['5', '^', '6']]]],
                          ('apply', ['plus', ('apply', ['minus', 'cn', 'cn']),
                                     ('apply', ['divide', ('apply', ['times', 'cn', 'cn']), ('apply', ['power', 'cn', 'cn'])])]))
        self.assertParses(csp.expr, '-(a + -3)', [['-', ['a', '+', ['-', '3']]]])
        self.assertParses(csp.expr, '1 ^ 2 ^ 3', [['1', '^', '2', '^', '3']],
                          ('apply', ['power', ('apply', ['power', 'cn', 'cn']), 'cn']))
        self.assertParses(csp.expr, 'not 1 < 2', [[['not', '1'], '<', '2']])
        self.assertParses(csp.expr, 'not (1 < 2)', [['not', ['1', '<', '2']]])
        self.assertParses(csp.expr, 'not not my:var', [['not', ['not', 'my:var']]],
                          ('apply', ['not', ('apply', ['not', 'ci'])]))
        self.assertParses(csp.expr, '1 < 2 <= (3 >= 4) > 5 == 6 != 7', [['1', '<', '2', '<=', ['3', '>=', '4'],
                                                                         '>', '5', '==', '6', '!=', '7']])
        self.assertParses(csp.expr, '1 < 2 + 4 > 3', [['1', '<', ['2', '+', '4'], '>', '3']])
        self.assertParses(csp.expr, '1 < 2 && 4 > 3', [[['1', '<', '2'], '&&', ['4', '>', '3']]])
        self.assertParses(csp.expr, '0 || 1 && 1', [['0', '||', '1', '&&', '1']],
                          ('apply', ['and', ('apply', ['or', 'cn', 'cn']), 'cn']))
        self.assertParses(csp.expr, 'A + B || C * D', [[['A', '+', 'B'], '||', ['C', '*', 'D']]])
        self.assertParses(csp.expr, 'if 1 then 2 else 3', [['1', '2', '3']],
                          ('piecewise', [('piece', ['cn', 'cn']), ('otherwise', ['cn'])]))
        self.assertParses(csp.expr, 'if 1 < 2 then 3 + 4 else 5 * 6',
                          [[['1', '<', '2'], ['3', '+', '4'], ['5', '*', '6']]])
    
    def TestParsingTrace(self):
        trace = {'{%s}trace' % CSP.PROTO_NS: '1'}
        self.assertParses(csp.expr, '1?', [['1']], ('cn:1', trace))
        self.assertParses(csp.expr, 'var?', [['var']], ('ci:var', trace))
        self.assertParses(csp.expr, '(1 + a)?', [[['1', '+', 'a']]], ('apply', trace, ['plus', 'cn:1', 'ci:a']))
        self.assertParses(csp.expr, '1 + a?', [['1', '+', ['a']]], ('apply', ['plus', 'cn:1', ('ci:a', trace)]))
    
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
        self.assertParses(csp.simpleAssign, 'var = value', [['var', 'value']],
                          ('apply', ['eq', 'ci:var', 'ci:value']))
        self.assertParses(csp.simpleAssign, 'var = pre:value', [['var', 'pre:value']])
        self.failIfParses(csp.simpleAssign, 'pre:var = value')
        self.assertParses(csp.simpleAssign, 'var = 1 + 2', [['var', ['1', '+', '2']]],
                          ('apply', ['eq', 'ci:var', ('apply', ['plus', 'cn:1', 'cn:2'])]))

        self.assertParses(csp.simpleAssignList, 'v1 = 1\nv2=2', [[['v1', '1'], ['v2', '2']]],
                          ('apply', ['csymbol-statementList', ('apply', ['eq', 'ci', 'cn']), ('apply', ['eq', 'ci', 'cn'])]))
        self.assertParses(csp.simpleAssignList, '', [[]])
    
    def TestParsingNamespaces(self):
        self.assertParses(csp.nsDecl, 'namespace prefix = "urn:test"', [{'prefix': 'prefix', 'uri': 'urn:test'}])
        self.assertParses(csp.nsDecl, "namespace prefix='urn:test'", [{'prefix': 'prefix', 'uri': 'urn:test'}])
        self.assertParses(csp.nsDecls, 'namespace n1="urn:t1"#test ns\nnamespace n2 = "urn:t2"',
                          {'namespace': [{'prefix': 'n1', 'uri': 'urn:t1'}, {'prefix': 'n2', 'uri': 'urn:t2'}]})
        self.assertParses(csp.nsDecls, '', [])
        self.failIfParses(csp.nsDecls, 'namespace n="uri"\n')
    
    def TestParsingInputs(self):
        '''Test parsing protocol input declarations'''
        self.assertParses(csp.inputs, """inputs {
    label_duration = 300  # How long to label for
    unlabel_time = 600000 # When to stop labelling
    use_tapasin = 1       # Whether to include Tapasin
}
""", [[[['label_duration', '300'], ['unlabel_time', '600000'], ['use_tapasin', '1']]]])
        self.assertParses(csp.inputs, 'inputs {}', [[[]]], [])
        self.assertParses(csp.inputs, 'inputs\n{\n}\n', [[[]]])
        self.assertParses(csp.inputs, 'inputs{X=1}', [[[['X', '1']]]],
                          ('inputs', [('apply', ['csymbol-statementList', ('apply', ['eq', 'ci:X', 'cn:1'])])]))
    
    def TestParsingImports(self):
        self.assertParses(csp.importStmt, 'import std = "../../../src/proto/library/BasicLibrary.xml"',
                          [['std', '../../../src/proto/library/BasicLibrary.xml']],
                          ('import', {'source': '../../../src/proto/library/BasicLibrary.xml', 'prefix': 'std'}))
        self.assertParses(csp.importStmt, "import 'TestS1S2.xml'", [['', 'TestS1S2.xml']],
                          ('import', {'source': 'TestS1S2.xml', 'mergeDefinitions': 'true'}))
        self.assertParses(csp.imports, 'import l1="file1"#blah\nimport "file2"',
                          [['l1', 'file1'], ['', 'file2']])
        self.assertParses(csp.imports, '', [])
        self.failIfParses(csp.imports, 'import "file"\n')

    def TestParsingImportsWithSetInput(self):
        self.assertParses(csp.importStmt, """import "S1S2.txt" {
    steady_state_beats = 10
    timecourse_duration = 2000
}""", [['', 'S1S2.txt', [['steady_state_beats', '10'], ['timecourse_duration', '2000']]]],
                          ('import', {'source': 'S1S2.txt', 'mergeDefinitions': 'true'},
                           [('setInput', {'name': 'steady_state_beats'}, ['cn:10']),
                            ('setInput', {'name': 'timecourse_duration'}, ['cn:2000'])]))
        self.assertParses(csp.importStmt, 'import "file.txt" { }', [['', 'file.txt', []]],
                          ('import', {'source': 'file.txt', 'mergeDefinitions': 'true'}))
        self.failIfParses(csp.importStmt, 'import "file.txt" { } \n')

    def TestParsingModelInterface(self):
        self.assertParses(csp.setTimeUnits, 'independent var units u', [['u']],
                          ('setIndependentVariableUnits', {'units': 'u'}))
        
        self.assertParses(csp.inputVariable, 'input test:var units u = 1.2', [['test:var', 'u', '1.2']],
                          ('specifyInputVariable', {'name': 'test:var', 'units': 'u', 'initial_value': '1.2'}))
        self.assertParses(csp.inputVariable, 'input test:var units u', [['test:var', 'u']],
                          ('specifyInputVariable', {'name': 'test:var', 'units': 'u'}))
        self.assertParses(csp.inputVariable, 'input test:var = -1e2', [['test:var', '-1e2']],
                          ('specifyInputVariable', {'name': 'test:var', 'initial_value': '-1e2'}))
        self.assertParses(csp.inputVariable, 'input test:var', [['test:var']],
                          ('specifyInputVariable', {'name': 'test:var'}))
        self.failIfParses(csp.inputVariable, 'input no_prefix')
        
        self.assertParses(csp.outputVariable, 'output test:var', [['test:var']],
                          ('specifyOutputVariable', {'name': 'test:var'}))
        self.assertParses(csp.outputVariable, 'output test:var units uname', [['test:var', 'uname']],
                          ('specifyOutputVariable', {'name': 'test:var', 'units': 'uname'}))
        self.failIfParses(csp.outputVariable, 'output no_prefix')
        
        self.assertParses(csp.optionalVariable, 'optional prefix:name', [['prefix:name']],
                          ('specifyOptionalVariable', {'name': 'prefix:name'}))
        self.failIfParses(csp.optionalVariable, 'optional no_prefix')
        
        self.assertParses(csp.newVariable, 'var varname units uname = 0', [['varname', 'uname', '0']],
                          ('declareNewVariable', {'name': 'varname', 'units': 'uname', 'initial_value': '0'}))
        self.assertParses(csp.newVariable, 'var varname units uname', [['varname', 'uname']],
                          ('declareNewVariable', {'name': 'varname', 'units': 'uname'}))
        self.failIfParses(csp.newVariable, 'var prefix:varname units uname = 0')
        self.failIfParses(csp.newVariable, 'var varname = 0')
        self.failIfParses(csp.newVariable, 'var varname')
        
        self.assertParses(csp.clampVariable, 'clamp p:v', [['p:v']],
                          ('addOrReplaceEquation', [('apply', ['eq', 'ci:p:v', 'ci:p:v'])]))
        self.assertParses(csp.clampVariable, 'clamp p:v to 0 :: U', [['p:v', '0']],
                          ('addOrReplaceEquation', [('apply', ['eq', 'ci:p:v', WithUnits('cn:0', 'U')])]))
        
        self.assertParses(csp.modelEquation, 'define local_var = 1::U + model:var',
                          [['local_var', ['1', '+', 'model:var']]],
                          ('addOrReplaceEquation', [('apply', ['eq', 'ci:local_var',
                                                               ('apply', ['plus', WithUnits('cn:1', 'U'), 'ci:model:var'])])]))
        self.assertParses(csp.modelEquation, 'define model:var = 2.5 :: units / local_var',
                          [['model:var', ['2.5', '/', 'local_var']]],
                          ('addOrReplaceEquation', [('apply', ['eq', 'ci:model:var',
                                                               ('apply', ['divide', WithUnits('cn:2.5', 'units'), 'ci:local_var'])])]))
        self.assertParses(csp.modelEquation, 'define diff(oxmeta:membrane_voltage; oxmeta:time) = 1 :: mV_per_ms',
                          [[['oxmeta:membrane_voltage', 'oxmeta:time'], '1']],
                          ('addOrReplaceEquation', [('apply', ['eq', ('apply', ['diff', ('bvar', ['ci:oxmeta:time']),
                                                                                'ci:oxmeta:membrane_voltage']),
                                                               WithUnits('cn:1', 'mV_per_ms')])]))
        
        self.assertParses(csp.unitsConversion, 'convert uname1 to uname2 by lambda u: u / model:var',
                          [['uname1', 'uname2', [[['u']], ['u', '/', 'model:var']]]],
                          ('unitsConversionRule', {'desiredDimensions': 'uname2', 'actualDimensions': 'uname1'},
                           [('lambda', [('bvar', ['ci:u']), ('apply', ['divide', 'ci:u', 'ci:model:var'])])]))

        self.assertParses(csp.modelInterface, """model interface {  # Comments can go here
    independent var units t
    
    input test:v1 = 0  # a comment
    input test:v2 units u
    output test:time
    # comments are always ignored
    output test:v3 units u
    
    optional test:opt
    
    var local units dimensionless = 5
    define test:v3 = test:v2 * local
    convert u1 to u2 by lambda u: u * test:v3
}""", [[['t'], ['test:v1', '0'], ['test:v2', 'u'], ['test:time'], ['test:v3', 'u'], ['test:opt'],
        ['local', 'dimensionless', '5'], ['test:v3', ['test:v2', '*', 'local']],
        ['u1', 'u2', [[['u']], ['u', '*', 'test:v3']]]]],
                          ('modelInterface', [('setIndependentVariableUnits', {'units': 't'}),
                                              ('specifyInputVariable', {'name': 'test:v1', 'initial_value': '0'}),
                                              ('specifyInputVariable', {'name': 'test:v2', 'units': 'u'}),
                                              ('specifyOutputVariable', {'name': 'test:time'}),
                                              ('specifyOutputVariable', {'name': 'test:v3', 'units': 'u'}),
                                              ('specifyOptionalVariable', {'name': 'test:opt'}),
                                              ('declareNewVariable', {'name': 'local', 'units': 'dimensionless', 'initial_value': '5'}),
                                              ('addOrReplaceEquation', [('apply', ['eq', 'ci:test:v3', ('apply', ['times', 'ci:test:v2', 'ci:local'])])]),
                                              ('unitsConversionRule', {'desiredDimensions': 'u2', 'actualDimensions': 'u1'},
                                               [('lambda', [('bvar', ['ci:u']), ('apply', ['times', 'ci:u', 'ci:test:v3'])])])]))
        self.assertParses(csp.modelInterface, 'model interface {}', [[]])
        self.assertParses(csp.modelInterface, 'model interface#comment\n{output test:time\n}', [[['test:time']]])
        self.assertParses(csp.modelInterface, 'model interface {output test:time }', [[['test:time']]])

    def TestParsingUniformRange(self):
        self.assertParses(csp.range, 'range time units ms uniform 0:1:1000', [['time', 'ms', ['0', '1', '1000']]],
                          ('uniformStepper', {'name': 'time', 'units': 'ms'},
                           [('start', ['cn:0']), ('stop', ['cn:1000']), ('step', ['cn:1'])]))
        self.assertParses(csp.range, 'range time units ms uniform 0:1000', [['time', 'ms', ['0', '1000']]],
                          ('uniformStepper', {'name': 'time', 'units': 'ms'},
                           [('start', ['cn:0']), ('stop', ['cn:1000']), ('step', ['cn:1'])]))
        self.assertParses(csp.range, 'range t units s uniform 0:end', [['t', 's', ['0', 'end']]],
                          ('uniformStepper', {'name': 't', 'units': 's'},
                           [('start', ['cn:0']), ('stop', ['ci:end']), ('step', ['cn:1'])]))
        # Spaces or brackets are required in this case to avoid 'start:step:end' parsing as an ident
        self.assertParses(csp.range, 'range t units s uniform start : step : end', [['t', 's', ['start', 'step', 'end']]],
                          ('uniformStepper', {'name': 't', 'units': 's'},
                           [('start', ['ci:start']), ('stop', ['ci:end']), ('step', ['ci:step'])]))
        self.assertParses(csp.range, 'range t units s uniform start:(step):end', [['t', 's', ['start', 'step', 'end']]])
        self.failIfParses(csp.range, 'range t units s uniform start:step:end')

    def TestParsingVectorRange(self):
        self.assertParses(csp.range, 'range run units dimensionless vector [1, 2, 3, 4]',
                          [['run', 'dimensionless', ['1', '2', '3', '4']]],
                          ('vectorStepper', {'name': 'run', 'units': 'dimensionless'},
                           [('apply', ['csymbol-newArray', 'cn:1', 'cn:2', 'cn:3', 'cn:4'])]))

    def TestParsingWhileRange(self):
        self.assertParses(csp.range, 'range rpt units dimensionless while rpt < 5',
                          [['rpt', 'dimensionless', ['rpt', '<', '5']]],
                          ('whileStepper', {'name': 'rpt', 'units': 'dimensionless'},
                           [('condition', [('apply', ['lt', 'ci:rpt', 'cn:5'])])]))

    def TestParsingModifiers(self):
        self.assertParses(csp.modifierWhen, 'at start', ['start'], 'when:AT_START_ONLY')
        self.assertParses(csp.modifierWhen, 'at each loop', ['each'], 'when:EVERY_LOOP')
        self.assertParses(csp.modifierWhen, 'at end', ['end'], 'when:AT_END')
        
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
                          [[['start', []], ['each', ['model:input', 'loopVariable']], ['end', ['savedState']]]],
                          ('modifiers', [('resetState', ['when:AT_START_ONLY']),
                                         ('setVariable', ['when:EVERY_LOOP', 'name:model:input', ('value', ['ci:loopVariable'])]),
                                         ('saveState', ['when:AT_END', 'name:savedState'])]))
        self.assertParses(csp.modifiers, 'modifiers {at start reset}', [[['start', []]]])
    
    def TestParsingTimecourseSimulations(self):
        self.assertParses(csp.simulation, 'simulation sim = timecourse { range time units ms uniform 1:10 }',
                          [['sim', [['time', 'ms', ['1', '10']]]]],
                          ('timecourseSimulation', {'prefix': 'sim'}, [('uniformStepper', {'name': 'time', 'units': 'ms'},
                                                                        [('start', ['cn:1']), ('stop', ['cn:10']), ('step', ['cn:1'])]),
                                                                       'modifiers']))
        self.assertParses(csp.simulation, 'simulation timecourse #c\n#c\n{\n range time units ms uniform 1:10\n\n }#c',
                          [['', [['time', 'ms', ['1', '10']]]]],
                          ('timecourseSimulation', {}, [('uniformStepper', {'name': 'time', 'units': 'ms'},
                                                         [('start', ['cn:1']), ('stop', ['cn:10']), ('step', ['cn:1'])]),
                                                        'modifiers']))
        self.assertParses(csp.simulation, """simulation sim = timecourse {
range time units U while time < 100
modifiers { at end save as prelim }
}""",
                          [['sim', [['time', 'U', ['time', '<', '100']], [['end', ['prelim']]]]]],
                          ('timecourseSimulation', {'prefix': 'sim'}, [('whileStepper', {'name': 'time', 'units': 'U'},
                                                                        [('condition', [('apply', ['lt', 'ci:time', 'cn:100'])])]),
                                                                       ('modifiers', [('saveState', ['when:AT_END', 'name:prelim'])])]))
        self.failIfParses(csp.simulation, 'simulation sim = timecourse {}')
    
    def TestParsingOneStepSimulations(self):
        self.assertParses(csp.simulation, 'simulation oneStep', [['', []]], ('oneStep', {}))
        self.assertParses(csp.simulation, 'simulation sim = oneStep', [['sim', []]], ('oneStep', {'prefix': 'sim'}))
        self.assertParses(csp.simulation, 'simulation oneStep 1.0', [['', ['1.0']]], ('oneStep', {'step': '1.0'}))
        self.assertParses(csp.simulation, 'simulation sim = oneStep step', [['sim', ['step']]],
                          ('oneStep', {'prefix': 'sim', 'step': 'step'}))
        self.assertParses(csp.simulation, 'simulation oneStep { modifiers { at start set a = 1 } }',
                          [['', [[['start', ['a', '1']]]]]],
                          ('oneStep', {}, [('modifiers', [('setVariable', ['when:AT_START_ONLY', 'name:a', ('value', ['cn:1'])])])]))
    
    def TestParsingNestedSimulations(self):
        self.assertParses(csp.simulation,
                          'simulation rpt = nested { range run units U while not rpt:result\n nests sim }',
                          [['rpt', [['run', 'U', ['not', 'rpt:result']], ['sim']]]],
                          ('nestedSimulation', {'prefix': 'rpt'},
                           [('whileStepper', [('condition', [('apply', ['not', 'ci:rpt:result'])])]),
                            'modifiers', ('subTask', {'task': 'sim'})]))
        self.assertParses(csp.simulation, """simulation nested {
range R units U uniform 3:5
modifiers { at each loop reset to prelim }
nests sim
}""",
                          [['', [['R', 'U', ['3', '5']], [['each', ['prelim']]], ['sim']]]],
                          ('nestedSimulation', {},
                           [('uniformStepper', [('start', ['cn:3']), ('stop', ['cn:5']), ('step', ['cn:1'])]),
                            ('modifiers', [('resetState', ['when:EVERY_LOOP', 'state:prelim'])]),
                            ('subTask', {'task': 'sim'})]))
        self.assertParses(csp.simulation, """simulation nested { range R units U uniform 1:2
nests simulation timecourse { range t units u uniform 1:100 } }""",
                          [['', [['R', 'U', ['1', '2']], [['', [['t', 'u', ['1', '100']]]]]]]],
                          ('nestedSimulation', {},
                           [('uniformStepper', [('start', ['cn:1']), ('stop', ['cn:2']), ('step', ['cn:1'])]),
                            'modifiers',
                            ('timecourseSimulation', {},
                             [('uniformStepper', [('start', ['cn:1']), ('stop', ['cn:100']), ('step', ['cn:1'])]), 'modifiers'])]))
        self.failIfParses(csp.simulation, 'simulation rpt = nested { range run units U while 1 }')
    
    def TestParsingNestedProtocol(self):
        self.assertParses(csp.simulation, 'simulation nested { range iter units D vector [0, 1]\n nests protocol "P" { } }',
                          [['', [['iter', 'D', ['0', '1']], [['P', []]]]]],
                          ('nestedSimulation', {}, [('vectorStepper',),
                                                    'modifiers',
                                                    ('nestedProtocol', {'source': 'P'}, [])]))
        self.assertParses(csp.simulation, """simulation nested {
    range iter units D vector [0, 1]
    nests protocol "../proto.xml" {
        input1 = 1
        input2 = 2
    }
}""", [['', [['iter', 'D', ['0', '1']], [['../proto.xml', [['input1', '1'], ['input2', '2']]]]]]],
                          ('nestedSimulation', {}, [('vectorStepper',),
                                                    'modifiers',
                                                    ('nestedProtocol', {'source': '../proto.xml'},
                                                     [('setInput', {'name': 'input1'}, ['cn:1']),
                                                      ('setInput', {'name': 'input2'}, ['cn:2'])])]))
        self.assertParses(csp.simulation, """simulation nested {
    range iter units D vector [0, 1]
    nests protocol "proto.txt" {
        input = iter
        select output oname
    }
}""", [['', [['iter', 'D', ['0', '1']], [['proto.txt', [['input', 'iter']], 'oname']]]]],
                          ('nestedSimulation', {}, [('vectorStepper',),
                                                    'modifiers',
                                                    ('nestedProtocol', {'source': 'proto.txt'},
                                                     [('setInput', {'name': 'input'}, ['ci:iter']),
                                                      ('selectOutput', {'name': 'oname'}, [])])]))
        # Tracing a nested protocol
        self.assertParses(csp.simulation, 'simulation nested { range iter units D vector [0, 1]\n nests protocol "P" { }? }',
                          [['', [['iter', 'D', ['0', '1']], [['P', []]]]]],
                          ('nestedSimulation', {}, [('vectorStepper',),
                                                    'modifiers',
                                                    ('nestedProtocol', {'source': 'P', '{%s}trace' % CSP.PROTO_NS: '1'}, [])]))
    
    def TestParsingTasks(self):
        self.assertParses(csp.tasks, """tasks {
    simulation timecourse { range time units second uniform 1:1000 }
    simulation main = nested { range n units dimensionless vector [i*2 for i in 1:4] 
                               nests inner }
}
""", [[['', [['time', 'second', ['1', '1000']]]],
       ['main', [['n', 'dimensionless', [['i', '*', '2'], ['i', ['1', '4']]]], ['inner']]]]],
                          ('simulations', [('timecourseSimulation', {}, [('uniformStepper',), 'modifiers']),
                                           ('nestedSimulation', {'prefix': 'main'}, [('vectorStepper',), 'modifiers', ('subTask',)])]))
    
    def TestParsingOutputSpecifications(self):
        self.assertParses(csp.outputSpec, 'name = model:var "Description"', [['name', 'model:var', 'Description']],
                          ('raw', {'name': 'name', 'ref': 'model:var', 'description': 'Description'}))
        self.assertParses(csp.outputSpec, r'name = ref:var units U "Description \"quotes\""',
                          [['name', 'ref:var', 'U', 'Description "quotes"']],
                          ('postprocessed', {'name': 'name', 'ref': 'ref:var', 'units': 'U', 'description': 'Description "quotes"'}))
        self.assertParses(csp.outputSpec, "name = ref:var units U 'Description \\'quotes\\' \"too\"'",
                          [['name', 'ref:var', 'U', 'Description \'quotes\' "too"']],
                          ('postprocessed', {'name': 'name', 'ref': 'ref:var', 'units': 'U', 'description': 'Description \'quotes\' "too"'}))
        self.assertParses(csp.outputSpec, 'varname units UU', [['varname', 'UU']],
                          ('postprocessed', {'name': 'varname', 'units': 'UU'}))
        self.assertParses(csp.outputSpec, 'varname units UU "desc"', [['varname', 'UU', 'desc']],
                          ('postprocessed', {'name': 'varname', 'units': 'UU', 'description': 'desc'}))
        self.failIfParses(csp.outputSpec, 'varname_no_units')
        
        self.assertParses(csp.outputs, """outputs #cccc
{ #cdc
        n1 = n2 units u1
        n3 = p:m 'd1'
        n4 units u2 "d2"
} #cpc
""", [[['n1', 'n2', 'u1'], ['n3', 'p:m', 'd1'], ['n4', 'u2', 'd2']]],
     ('outputVariables', [('postprocessed', {'name': 'n1', 'ref': 'n2', 'units': 'u1'}),
                          ('raw', {'name': 'n3', 'ref': 'p:m', 'description': 'd1'}),
                          ('postprocessed', {'name': 'n4', 'units': 'u2', 'description': 'd2'})]))
        self.assertParses(csp.outputs, "outputs {}", [[]])
    
    def TestParsingPlotSpecifications(self):
        # TODO: Test these more once the XML syntax catches up!
        self.assertParses(csp.plotCurve, 'y against x', [['y', 'x']])
        self.assertParses(csp.plotCurve, 'y, y2 against x', [['y', 'y2', 'x']])
        self.failIfParses(csp.plotCurve, 'm:y against x')
        self.failIfParses(csp.plotCurve, 'y against m:x')
        self.assertParses(csp.plotSpec, 'plot "A title\'s good" { y1, y2 against x1\n y3 against x2 }',
                          [["A title's good", ['y1', 'y2', 'x1'], ['y3', 'x2']]])
        self.assertParses(csp.plotSpec, 'plot "Keys" { y against x key k }', [["Keys", ['y', 'x', 'k']]],
                          ('plot', ['title:Keys', 'x:x', 'y:y', 'key:k']))
        self.failIfParses(csp.plotSpec, 'plot "only title" {}')
        self.failIfParses(csp.plotSpec, 'plot "only title"')
        
        self.assertParses(csp.plots, """plots { plot "t1" { v1 against v2 key vk }
        plot "t1" { v3, v4 against v5 }
}""", [[['t1', ['v1', 'v2', 'vk']], ['t1', ['v3', 'v4', 'v5']]]])
        self.assertParses(csp.plots, 'plots {}', [[]])
    
    def TestParsingFunctionCalls(self):
        self.assertParses(csp.functionCall, 'noargs()', [['noargs', []]], ('apply', ['ci:noargs']))
        self.assertParses(csp.functionCall, 'swap(a, b)', [['swap', ['a', 'b']]],
                          ('apply', ['ci:swap', 'ci:a', 'ci:b']))
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
                          [['MathML:exp', [['MathML:floor', ['MathML:exponentiale']]]]],
                          ('apply', ['exp', ('apply', ['floor', 'exponentiale'])]))
    
    def TestParsingAssignStatements(self):
        self.assertParses(csp.assignStmt, 'var = value', [[['var'], ['value']]], ('apply', ['eq', 'ci', 'ci']))
        self.assertParses(csp.assignStmt, 'var = pre:value', [[['var'], ['pre:value']]])
        self.failIfParses(csp.assignStmt, 'pre:var = value')
        self.assertParses(csp.assignStmt, 'var = 1 + 2', [[['var'], [['1', '+', '2']]]],
                          ('apply', ['eq', 'ci', ('apply', ['plus', 'cn', 'cn'])]))

        self.assertParses(csp.assignStmt, 'a, b = tuple', [[['a', 'b'], ['tuple']]],
                          ('apply', ['eq', ('apply', ['csymbol-tuple', 'ci:a', 'ci:b']), 'ci:tuple']))
        self.assertParses(csp.assignStmt, 'a, b = b, a', [[['a', 'b'], ['b', 'a']]],
                          ('apply', ['eq', ('apply', ['csymbol-tuple', 'ci:a', 'ci:b']), ('apply', ['csymbol-tuple', 'ci:b', 'ci:a'])]))
        self.assertParses(csp.assignStmt, 'a, b = (b, a)', [[['a', 'b'], [['b', 'a']]]],
                          ('apply', ['eq', ('apply', ['csymbol-tuple', 'ci', 'ci']), ('apply', ['csymbol-tuple', 'ci', 'ci'])]))
        self.failIfParses(csp.assignStmt, 'p:a, p:b = e')
        self.failIfParses(csp.assignStmt, '')
    
    def TestParsingReturnStatements(self):
        self.assertParses(csp.returnStmt, 'return 2 * a', [[['2', '*', 'a']]],
                          ('apply', ['csymbol-return', ('apply', ['times', 'cn', 'ci'])]))
        self.assertParses(csp.returnStmt, 'return (3 - 4)', [[['3', '-', '4']]],
                          ('apply', ['csymbol-return', ('apply', ['minus', 'cn', 'cn'])]))
        self.assertParses(csp.returnStmt, 'return a, b', [['a', 'b']],
                          ('apply', ['csymbol-return', 'ci', 'ci']))
        self.assertParses(csp.returnStmt, 'return a + 1, b - 1', [[['a', '+', '1'], ['b', '-', '1']]])
        self.assertParses(csp.returnStmt, 'return (a, b)', [[['a', 'b']]],
                          ('apply', ['csymbol-return', ('apply', ['csymbol', 'ci:a', 'ci:b'])]))
    
    def TestParsingAssertStatements(self):
        self.assertParses(csp.assertStmt, 'assert a + b', [[['a', '+', 'b']]],
                          ('apply', ['csymbol-assert', ('apply', ['plus', 'ci', 'ci'])]))
        self.assertParses(csp.assertStmt, 'assert (a + b)', [[['a', '+', 'b']]],
                          ('apply', ['csymbol-assert', ('apply', ['plus', 'ci', 'ci'])]))
        self.assertParses(csp.assertStmt, 'assert 1', [['1']],
                          ('apply', [('csymbol', {'definitionURL': 'https://chaste.cs.ox.ac.uk/nss/protocol/assert'}), 'cn']))
    
    def TestParsingStatementLists(self):
        self.assertParses(csp.stmtList, "b=-a\nassert 1", [[[['b'], [['-', 'a']]], ['1']]],
                          ('apply', ['csymbol-statementList',
                                     ('apply', ['eq', 'ci', ('apply', ['minus', 'ci'])]),
                                     ('apply', ['csymbol-assert', 'cn'])]))
        self.assertParses(csp.stmtList, """assert a < 0 # comments are ok

# as are blank lines
b = -a
assert b > 0
c, d = a * 2, b + 1
return c, d""", [[[['a', '<', '0']],
                  [['b'], [['-', 'a']]],
                  [['b', '>', '0']],
                  [['c', 'd'], [['a', '*', '2'], ['b', '+', '1']]],
                  ['c', 'd']]])
        self.failIfParses(csp.stmtList, '')
    
    def TestParsingLambdaExpressions(self):
        self.assertParses(csp.lambdaExpr, 'lambda a: a + 1', [[[['a']], ['a', '+', '1']]],
                          ('lambda', [('bvar', ['ci']), ('apply', ['plus', 'ci', 'cn'])]))
        self.assertParses(csp.lambdaExpr, 'lambda a, b: a + b', [[[['a'], ['b']], ['a', '+', 'b']]],
                          ('lambda', [('bvar', ['ci']), ('bvar', ['ci']), ('apply', ['plus', 'ci', 'ci'])]))
        self.assertParses(csp.lambdaExpr, 'lambda a, b=2: a - b', [[[['a'], ['b', '2']], ['a', '-', 'b']]],
                          ('lambda', [('bvar', ['ci']),
                                      ('semantics', [('bvar', ['ci']), ('annotation-xml', ['cn'])]),
                                      ('apply', ['minus', 'ci', 'ci'])]))
        self.assertParses(csp.expr, 'lambda a=c, b: a * b', [[[['a', 'c'], ['b']], ['a', '*', 'b']]])
        self.assertParses(csp.lambdaExpr, 'lambda a=p:c, b: a * b', [[[['a', 'p:c'], ['b']], ['a', '*', 'b']]])
        self.failIfParses(csp.lambdaExpr, 'lambda p:a: 5')

        self.assertParses(csp.lambdaExpr, """lambda a, b {
assert a > b
c = a - b
return c
}
""", [[[['a'], ['b']], [[['a', '>', 'b']],
                        [['c'], [['a', '-', 'b']]],
                        ['c']]]])
        self.assertParses(csp.expr, "lambda a, b { return b, a }", [[[['a'], ['b']], [['b', 'a']]]])
        self.assertParses(csp.expr, "lambda a { return a }", [[[['a']], [['a']]]],
                          ('lambda', [('bvar', ['ci']), ('apply', ['csymbol-statementList', ('apply', ['csymbol', 'ci'])])]))
        self.assertParses(csp.expr, 'lambda { return 1 }', [[[], [['1']]]],
                          ('lambda', [('apply', ['csymbol-statementList', ('apply', ['csymbol-return', 'cn'])])]))
        self.assertParses(csp.expr, 'lambda: 1', [[[], '1']],
                          ('lambda', [('cn')]))

    def TestParsingFunctionDefinitions(self):
        self.assertParses(csp.functionDefn, 'def double(a)\n {\n return a * 2\n }',
                          [['double', [['a']], [[['a', '*', '2']]]]],
                          ('apply', ['eq', 'ci',
                                     ('lambda', [('bvar', ['ci']),
                                                 ('apply', ['csymbol-statementList',
                                                            ('apply', ['csymbol-return', ('apply', ['times', 'ci', 'cn'])])])])]))
        self.assertParses(csp.functionDefn, 'def double(a): a * 2',
                          [['double', [['a']], ['a', '*', '2']]])
        # A function definition is just sugar for an assignment of a lambda expression
        self.assertParses(csp.stmtList, 'def double(a) {\n    return a * 2}',
                          [[['double', [['a']], [[['a', '*', '2']]]]]],
                          ('apply', ['csymbol',
                                     ('apply', ['eq', 'ci',
                                                ('lambda', [('bvar', ['ci']),
                                                            ('apply', ['csymbol-statementList',
                                                                       ('apply', ['csymbol-return', ('apply', ['times', 'ci', 'cn'])])])])])]))
        self.assertParses(csp.functionDefn, 'def noargs(): 1', [['noargs', [], '1']])
    
    def TestParsingNestedFunctions(self):
        self.assertParses(csp.functionDefn, """def outer()
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
    
    def TestParsingTuples(self):
        self.assertParses(csp.tuple, '(1,2)', [['1', '2']], ('apply', ['csymbol-tuple', 'cn', 'cn']))
        self.assertParses(csp.tuple, '(1+a,2*b)', [[['1', '+', 'a'], ['2', '*', 'b']]])
        self.assertParses(csp.tuple, '(singleton,)', [['singleton']], ('apply', ['csymbol-tuple', 'ci']))
        self.failIfParses(csp.tuple, '(1)') # You need a Python-style comma as above
        self.assertParses(csp.expr, '(1,2)', [['1', '2']], ('apply', ['csymbol', 'cn', 'cn']))
        self.assertParses(csp.expr, '(1,a,3,c)', [['1', 'a', '3', 'c']], ('apply', ['csymbol-tuple', 'cn', 'ci', 'cn', 'ci']))
        self.assertParses(csp.assignStmt, 't = (1,2)', [[['t'], [['1', '2']]]])
        self.assertParses(csp.assignStmt, 'a, b = (1,2)', [[['a', 'b'], [['1', '2']]]])
    
    def TestParsingArrays(self):
        self.assertParses(csp.expr, '[1, 2, 3]', [['1', '2', '3']],
                          ('apply', ['csymbol-newArray', 'cn', 'cn', 'cn']))
        self.assertParses(csp.array, '[[a, b], [c, d]]', [[['a', 'b'], ['c', 'd']]],
                          ('apply', ['csymbol-newArray', ('apply', ['csymbol-newArray', 'ci', 'ci']),
                                     ('apply', ['csymbol-newArray', 'ci', 'ci'])]))
        self.assertParses(csp.array, '[ [ [1+2,a,b]],[[3/4,c,d] ]]', [[[[['1', '+', '2'],'a','b']],
                                                                        [[['3', '/', '4'],'c','d']]]])

    def TestParsingArrayComprehensions(self):
        self.assertParses(csp.array, '[i for i in 0:N]', [['i', ['i', ['0', 'N']]]],
                          ('apply', ['csymbol-newArray',
                                     ('domainofapplication', [('apply', ['csymbol-tuple', 'cn:0', 'cn:1', 'ci:N', 'csymbol-string:i'])]),
                                     'ci:i']))
        self.assertParses(csp.expr, '[i*2 for i in 0:2:4]', [[['i', '*', '2'], ['i', ['0', '2', '4']]]])
        self.assertParses(csp.array, '[i+j*5 for i in 1:3 for j in 2:4]',
                          [[['i', '+', ['j', '*', '5']], ['i', ['1', '3']], ['j', ['2', '4']]]])
        self.assertParses(csp.array, '[block for 1$i in 2:10]', [['block', ['1', 'i', ['2', '10']]]])
        self.assertParses(csp.array, '[i^j for i in 1:3 for 2$j in 4:-1:2]',
                          [[['i', '^', 'j'], ['i', ['1', '3']], ['2', 'j', ['4', ['-', '1'], '2']]]],
                          ('apply', ['csymbol-newArray',
                                     ('domainofapplication', [('apply', ['csymbol-tuple', 'cn:1', 'cn:1', 'cn:3', 'csymbol-string:i']),
                                                              ('apply', ['csymbol-tuple', 'cn:2', 'cn:4', ('apply', ['minus', 'cn:1']), 'cn:2', 'csymbol-string:j'])]),
                                     ('apply', ['power', 'ci:i', 'ci:j'])]))
        # Dimension specifiers can be expressions too...
        self.assertParses(csp.expr, '[i for (1+2)$i in 2:(3+5)]', [['i', [['1', '+', '2'], 'i', ['2', ['3', '+', '5']]]]])
        self.assertParses(csp.expr, '[i for 1+2$i in 2:4]', [['i', [['1', '+', '2'], 'i', ['2', '4']]]],
                          ('apply', ['csymbol-newArray',
                                     ('domainofapplication', [('apply', ['csymbol-tuple', ('apply', ['plus', 'cn:1', 'cn:2']),
                                                                         'cn:2', 'cn:1', 'cn:4', 'csymbol-string:i'])]),
                                     'ci:i']))
        self.failIfParses(csp.expr, '[i for 1 $i in 2:4]')
    
    def TestParsingViews(self):
        self.assertParses(csp.expr, 'A[1:3:7]', [['A', ['1', '3', '7']]],
                          ('apply', ['csymbol-view', 'ci:A',
                                     ('apply', ['csymbol-tuple', 'cn:0', 'cn:1', 'cn:3', 'cn:7']),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'A[2$6:-2:4]', [['A', ['2', '6', ['-', '2'], '4']]],
                          ('apply', ['csymbol-view', 'ci:A',
                                     ('apply', ['csymbol-tuple', 'cn:2', 'cn:6', ('apply', ['minus', 'cn:2']), 'cn:4']),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'sim:res[1$2]', [['sim:res', ['1', '2']]],
                          ('apply', ['csymbol-view', 'ci:sim:res',
                                     ('apply', ['csymbol-tuple', 'cn:1', 'cn:2', 'cn:0', 'cn:2']),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'func(A)[5]', [[['func', ['A']], ['5']]])
        self.assertParses(csp.expr, 'arr[:]', [['arr', ['', '']]],
                          ('apply', ['csymbol-view', 'ci:arr',
                                     ('apply', ['csymbol-tuple', 'cn:0', 'csymbol-null', 'cn:1', 'csymbol-null']),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'arr[2:]', [['arr', ['2', '']]],
                          ('apply', ['csymbol-view', 'ci:arr',
                                     ('apply', ['csymbol-tuple', 'cn:0', 'cn:2', 'cn:1', 'csymbol-null']),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'arr[:2:]', [['arr', ['', '2', '']]],
                          ('apply', ['csymbol-view', 'ci:arr',
                                     ('apply', ['csymbol-tuple', 'cn:0', 'csymbol-null', 'cn:2', 'csymbol-null']),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'arr[:-alpha]', [['arr', ['', ['-', 'alpha']]]],
                          ('apply', ['csymbol-view', 'ci:arr',
                                     ('apply', ['csymbol-tuple', 'cn:0', 'csymbol-null', 'cn:1', ('apply', ['minus', 'ci:alpha'])]),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'arr[-3:-1:]', [['arr', [['-', '3'], ['-', '1'], '']]])
        self.assertParses(csp.expr, 'genericity[*$:]', [['genericity', ['*', '', '']]],
                          ('apply', ['csymbol-view', 'ci:genericity',
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'genericity[*$0]', [['genericity', ['*', '0']]],
                          ('apply', ['csymbol-view', 'ci:genericity',
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'cn:0', 'cn:0', 'cn:0'])]))
        self.assertParses(csp.expr, 'genericity[*$0:5]', [['genericity', ['*', '0', '5']]],
                          ('apply', ['csymbol-view', 'ci:genericity',
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'cn:0', 'cn:1', 'cn:5'])]))
        self.assertParses(csp.expr, 'genericity[*$0:5:50]', [['genericity', ['*', '0', '5', '50']]],
                          ('apply', ['csymbol-view', 'ci:genericity',
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'cn:0', 'cn:5', 'cn:50'])]))
        self.assertParses(csp.expr, 'genericity[*$:5:]', [['genericity', ['*', '', '5', '']]],
                          ('apply', ['csymbol-view', 'ci:genericity',
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:5', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'genericity[*$0:]', [['genericity', ['*', '0', '']]],
                          ('apply', ['csymbol-view', 'ci:genericity',
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'cn:0', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'multiples[3][4]', [['multiples', ['3'], ['4']]],
                          ('apply', ['csymbol-view', 'ci:multiples',
                                     ('apply', ['csymbol-tuple', 'cn:0', 'cn:3', 'cn:0', 'cn:3']),
                                     ('apply', ['csymbol-tuple', 'cn:1', 'cn:4', 'cn:0', 'cn:4']),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'multiples[1$3][0$:-step:0][*$0]',
                          [['multiples', ['1', '3'], ['0', '', ['-', 'step'], '0'], ['*', '0']]],
                          ('apply', ['csymbol-view', 'ci:multiples',
                                     ('apply', ['csymbol-tuple', 'cn:1', 'cn:3', 'cn:0', 'cn:3']),
                                     ('apply', ['csymbol-tuple', 'cn:0', 'csymbol-null', ('apply', ['minus', 'ci:step']), 'cn:0']),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'cn:0', 'cn:0', 'cn:0'])]))
        self.assertParses(csp.expr, 'dimspec[dim$0:2]', [['dimspec', ['dim', '0', '2']]],
                          ('apply', ['csymbol-view', 'ci:dimspec',
                                     ('apply', ['csymbol-tuple', 'ci:dim', 'cn:0', 'cn:1', 'cn:2']),
                                     ('apply', ['csymbol-tuple', 'csymbol-null', 'csymbol-null', 'cn:1', 'csymbol-null'])]))
        self.assertParses(csp.expr, 'okspace[ 0$ (1+2) : a+b : 50 ]',
                          [['okspace', ['0', ['1', '+', '2'], ['a', '+', 'b'], '50']]])
        # Some spaces aren't allowed
        self.failIfParses(csp.expr, 'arr [1]')
        self.failIfParses(csp.expr, 'arr[1] [3]')
        self.failIfParses(csp.expr, 'arr[1 $ 2]')
    
    def TestParsingFindAndIndex(self):
        # Curly braces represent index, with optional pad or shrink argument.  Find is a function call.
        self.assertParses(csp.expr, 'find(arr)', [['find', ['arr']]],
                          ('apply', ['csymbol-find', 'ci:arr']))
        self.failIfParses(csp.expr, 'find (arr)')
        #self.failIfParses(csp.expr, 'find(arr, extra)') # Needs special support e.g. from parse actions
        
        self.assertParses(csp.expr, 'arr{idxs}', [['arr', ['idxs']]],
                          ('apply', ['csymbol-index', 'ci:arr', 'ci:idxs', 'csymbol-defaultParameter', 'csymbol-defaultParameter']))
        self.failIfParses(csp.expr, 'arr {spaced}')
        self.assertParses(csp.expr, 'arr{idxs, shrink:1}', [['arr', ['idxs', '1']]],
                          ('apply', ['csymbol-index', 'ci:arr', 'ci:idxs', 'csymbol-defaultParameter', 'cn:1']))
        self.assertParses(csp.expr, 'arr{idxs, dim, shrink:-1}', [['arr', ['idxs', 'dim', ['-', '1']]]],
                          ('apply', ['csymbol-index', 'ci:arr', 'ci:idxs', 'ci:dim', ('apply', ['minus', 'cn:1'])]))
        self.assertParses(csp.expr, 'arr{idxs, dim, pad:1=value}', [['arr', ['idxs', 'dim', '1', 'value']]],
                          ('apply', ['csymbol-index', 'ci:arr', 'ci:idxs', 'ci:dim', 'csymbol-defaultParameter', 'cn:1', 'ci:value']))
        self.assertParses(csp.expr, 'arr{idxs, shrink:0, pad:1=value}', [['arr', ['idxs', '0', '1', 'value']]],
                          ('apply', ['csymbol-index', 'ci:arr', 'ci:idxs', 'csymbol-defaultParameter', 'cn:0', 'cn:1', 'ci:value']))
        self.assertParses(csp.expr, 'f(1,2){find(blah), 0, shrink:1}', [[['f', ['1', '2']], [['find', ['blah']], '0', '1']]],
                          ('apply', ['csymbol-index', ('apply', ['ci:f', 'cn:1', 'cn:2']),
                                     ('apply', ['csymbol-find', 'ci:blah']), 'cn:0', 'cn:1']))
        self.assertParses(csp.expr, 'A{find(A), 0, pad:-1=1+2}', [['A', [['find', ['A']], '0', ['-', '1'], ['1', '+', '2']]]],
                          ('apply', ['csymbol-index', 'ci:A', ('apply', ['csymbol-find', 'ci:A']),
                                     'cn:0', 'csymbol-defaultParameter', ('apply', ['minus', 'cn:1']), ('apply', ['plus', 'cn:1', 'cn:2'])]))

    def TestParsingUnitsDefinitions(self):
        # Possible syntax:  (mult, offset, expt are 'numbers'; prefix is SI prefix name; base is ncIdent)
        #  new_simple = [mult] [prefix] base [+|- offset]
        #  new_complex = p.delimitedList( [mult] [prefix] base [^expt], '.')
        self.assertParses(csp.unitsDef, 'ms = milli second', [['ms', ['milli', 'second']]],
                          ('units', {'name': 'ms'}, [('unit', {'units': 'second', 'prefix': 'milli'})]))
        self.assertParses(csp.unitsDef, 'C = kelvin - 273.15', [['C', ['kelvin', ['-', '273.15']]]],
                          ('units', {'name': 'C'}, [('unit', {'units': 'kelvin', 'offset': '-273.15'})]))
        self.assertParses(csp.unitsDef, 'C=kelvin+(-273.15)', [['C', ['kelvin', ['+', '(-273.15)']]]],
                          ('units', {'name': 'C'}, [('unit', {'units': 'kelvin', 'offset': '-273.14999999999998'})]))
        self.assertParses(csp.unitsDef, 'litre = 1000 centi metre^3', [['litre', ['1000', 'centi', 'metre', '3']]],
                          ('units', {'name': 'litre'}, [('unit', {'units': 'metre', 'multiplier': '1000', 'prefix': 'centi', 'exponent': '3'})]))
        self.assertParses(csp.unitsDef, 'accel_units = kilo metre . second^-2 "km/s^2"',
                          [['accel_units', ['kilo', 'metre'], ['second', '-2'], 'km/s^2']],
                          ('units', {'name': 'accel_units'}, [('unit', {'units': 'metre', 'prefix': 'kilo'}), ('unit', {'units': 'second', 'exponent': '-2'})]))
        self.assertParses(csp.unitsDef, 'fahrenheit = (5/9) celsius + 32.0',
                          [['fahrenheit', ['(5/9)', 'celsius', ['+', '32.0']]]],
                          ('units', {'name': 'fahrenheit'}, [('unit', {'units': 'celsius', 'offset': '32.0', 'multiplier': '0.55555555555555558'})]))
        self.assertParses(csp.unitsDef, 'fahrenheit = (5/9) kelvin + (32 - 273.15 * 9 / 5)',
                          [['fahrenheit', ['(5/9)', 'kelvin', ['+', '(32 - 273.15 * 9 / 5)']]]],
                          ('units', {'name': 'fahrenheit'}, [('unit', {'units': 'kelvin', 'offset': '-459.66999999999996', 'multiplier': '0.55555555555555558'})]))
        
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
                          ('units', [('units', {'name': 'nM'}, [('unit', {'units': 'mole', 'prefix': 'nano'}), 'unit']),
                                     ('units', {'name': 'hour'}, [('unit', {'units': 'second', 'multiplier': '3600'})]),
                                     ('units', {'name': 'flux'}, [('unit', {'units': 'nM'}), ('unit', {'units': 'hour', 'exponent': '-1'})]),
                                     ('units', {'name': 'rate_const'}, [('unit', {'units': 'hour', 'exponent': '-1'})]),
                                     ('units', {'name': 'rate_const_2'}, ['unit', 'unit'])]))
    
    def TestParsingAccessors(self):
        for accessor in ['NUM_DIMS', 'SHAPE', 'NUM_ELEMENTS']:
            self.assertParses(csp.accessor, '.' + accessor, [accessor])
            self.assertParses(csp.expr, 'var.' + accessor, [['var', accessor]],
                              ('apply', ['csymbol-accessor:' + accessor, 'ci']))
        for ptype in ['SIMPLE_VALUE', 'ARRAY', 'STRING', 'TUPLE', 'FUNCTION', 'NULL', 'DEFAULT']:
            self.assertParses(csp.accessor, '.IS_' + ptype, ['IS_' + ptype])
            self.assertParses(csp.expr, 'var.IS_' + ptype, [['var', 'IS_' + ptype]])
        self.assertParses(csp.expr, 'arr.SHAPE[1]', [[['arr', 'SHAPE'], ['1']]])
        self.assertParses(csp.expr, 'func(var).IS_ARRAY', [[['func', ['var']], 'IS_ARRAY']])
        self.assertParses(csp.expr, 'A.SHAPE.IS_ARRAY', [['A', 'SHAPE', 'IS_ARRAY']],
                          ('apply', ['csymbol-accessor:IS_ARRAY', ('apply', ['csymbol-accessor:SHAPE', 'ci'])]))
        self.failIfParses(csp.expr, 'arr .SHAPE')
    
    def TestParsingMap(self):
        self.assertParses(csp.expr, 'map(func, a1, a2)', [['map', ['func', 'a1', 'a2']]],
                          ('apply', ['csymbol-map', 'ci:func', 'ci:a1', 'ci:a2']))
        self.assertParses(csp.expr, 'map(lambda a, b: a+b, A, B)',
                          [['map', [[[['a'], ['b']], ['a', '+', 'b']], 'A', 'B']]],
                          ('apply', ['csymbol-map', ('lambda', [('bvar', ['ci:a']), ('bvar', ['ci:b']),
                                                                ('apply', ['plus', 'ci:a', 'ci:b'])]), 'ci:A', 'ci:B']))
        self.assertParses(csp.expr, 'map(id, a)', [['map', ['id', 'a']]])
        self.assertParses(csp.expr, 'map(hof(arg), a, b, c, d, e)',
                          [['map', [['hof', ['arg']], 'a', 'b', 'c', 'd', 'e']]])
        # self.failIfParses(csp.expr, 'map(f)') # At present implemented just as a function call with special name
    
    def TestParsingFold(self):
        self.assertParses(csp.expr, 'fold(func, array, init, dim)', [['fold', ['func', 'array', 'init', 'dim']]],
                          ('apply', ['csymbol-fold', 'ci:func', 'ci:array', 'ci:init', 'ci:dim']))
        self.assertParses(csp.expr, 'fold(lambda a, b: a - b, f(), 1, 2)',
                          [['fold', [[[['a'], ['b']], ['a', '-', 'b']], ['f', []], '1', '2']]])
        self.assertParses(csp.expr, 'fold(f, A)', [['fold', ['f', 'A']]])
        self.assertParses(csp.expr, 'fold(f, A, 0)', [['fold', ['f', 'A', '0']]])
        self.assertParses(csp.expr, 'fold(f, A, default, 1)', [['fold', ['f', 'A', [], '1']]])
        #self.failIfParses(csp, expr, 'fold()')
        #self.failIfParses(csp, expr, 'fold(f, A, i, d, extra)')

    def TestParsingWrappedMathmlOperators(self):
        self.assertParses(csp.expr, '@3:+', [['3', '+']], 'csymbol-wrap/3:plus')
        self.assertParses(csp.expr, '@1:MathML:sin', [['1', 'MathML:sin']], 'csymbol-wrap/1:sin')
        self.assertParses(csp.expr, 'map(@2:/, a, b)', [['map', [['2', '/'], 'a', 'b']]])
        #self.failIfParses(csp.expr, '@0:+') # Best done at parse action level?
        self.failIfParses(csp.expr, '@1:non_mathml')
        self.failIfParses(csp.expr, '@ 2:*')
        self.failIfParses(csp.expr, '@2 :-')
        self.failIfParses(csp.expr, '@1:--')
        self.failIfParses(csp.expr, '@N:+')

    def TestParsingNullAndDefault(self):
        self.assertParses(csp.expr, 'null', [[]], 'csymbol-null')
        self.assertParses(csp.expr, 'default', [[]], 'csymbol-defaultParameter')

    def TestParsingLibrary(self):
        self.assertParses(csp.library, 'library {}', [[]])
        self.assertParses(csp.library, """library
{
    def f(a) {
        return a
    }
    f2 = lambda b: b/2
    const = 13
}
""", [[[['f', [['a']], [['a']]],
        [['f2'], [[[['b']], ['b', '/', '2']]]],
        [['const'], ['13']]]]],
                          ('library', [('apply', ['csymbol-statementList',
                                                  ('apply', ['eq', 'ci:f',
                                                             ('lambda', [('bvar', ['ci:a']),
                                                                         ('apply', ['csymbol-statementList',
                                                                                    ('apply', ['csymbol-return', 'ci:a'])])])]),
                                                  ('apply', ['eq', 'ci:f2',
                                                             ('lambda', [('bvar', ['ci:b']), ('apply', ['divide', 'ci:b', 'cn:2'])])]),
                                                  ('apply', ['eq', 'ci:const', 'cn:13'])])]))
    
    def TestParsingPostProcessing(self):
        self.assertParses(csp.postProcessing, """post-processing
{
    a = check(sim:result)
    assert a > 5
}
""", [[[['a'], [['check', ['sim:result']]]],
       [['a', '>', '5']]]],
                          ('post-processing', [('apply', ['csymbol-statementList',
                                                          ('apply', ['eq', 'ci:a', ('apply', ['ci:check', 'ci:sim:result'])]),
                                                          ('apply', ['csymbol-assert', ('apply', ['gt', 'ci:a', 'cn:5'])])])]))
    
    def TestParsingFullProtocols(self):
        test_folder = 'projects/FunctionalCuration/test/protocols/compact'
        ref_folder = 'projects/FunctionalCuration/test/data/CompactSyntaxParser'
        output_folder = os.path.join(CHASTE_TEST_OUTPUT, 'TestCompactSyntaxParser')
        try:
            os.makedirs(output_folder)
        except OSError:
            pass
        for proto_filename in glob.glob(os.path.join(test_folder, '*.txt')):
            proto_base = os.path.splitext(os.path.basename(proto_filename))[0]
            print proto_base, '...'
            parsed_tree = csp().ParseFile(proto_filename)
            # Check the xml:base attribute
            self.assert_('{http://www.w3.org/XML/1998/namespace}base' in parsed_tree.getroot().attrib)
            self.assertEqual(parsed_tree.getroot().base, proto_filename)
            # We write to file for easy creation of new reference versions
            output_file_path = os.path.join(output_folder, proto_base + '.xml')
            output_file = open(output_file_path, 'w')
            parsed_tree.write(output_file, pretty_print=True, xml_declaration=True)
            output_file.close()
            ref_file_path = os.path.join(ref_folder, proto_base + '.xml')
            if os.path.exists(ref_file_path):
                self.assertXmlEqual(parsed_tree.getroot(), CSP.ET.parse(ref_file_path).getroot())
        CSP.Actions.source_file = '' # Avoid the last name leaking to subsequent tests

    def TestZzzPackratWasUsed(self):
        # Method name ensures this runs last!
        self.assert_(len(CSP.p.ParserElement._exprArgCache) > 0)
