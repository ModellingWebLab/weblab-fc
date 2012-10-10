
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

import pyparsing as p

__all__ = ['CompactSyntaxParser']

# Necessary for reasonable speed when using operatorPrecedences
p.ParserElement.enablePackrat()

def MakeKw(keyword, suppress=True):
    """Helper function to create a parser for the given keyword."""
    kw = p.Keyword(keyword)
    if suppress:
        kw = kw.suppress()
    return kw

class Optional(p.Optional):
    """An Optional pattern that doesn't consume whitespace if the contents don't match."""
    def __init__(self, *args, **kwargs):
        super(Optional, self).__init__(*args, **kwargs)
        self.callPreparse = False
        self._optionalNotMatched = p.Optional(p.Empty()).defaultValue
    
    def parseImpl( self, instring, loc, doActions=True ):
        try:
            loc, tokens = self.expr._parse( instring, loc, doActions, callPreParse=True )
        except (p.ParseException,IndexError):
            if self.defaultValue is not self._optionalNotMatched:
                if self.expr.resultsName:
                    tokens = p.ParseResults([ self.defaultValue ])
                    tokens[self.expr.resultsName] = self.defaultValue
                else:
                    tokens = [ self.defaultValue ]
            else:
                tokens = []
        return loc, tokens

def OptionalDelimitedList(expr, delim):
    """Like delimitedList, but the list may be empty."""
    return p.delimitedList(expr, delim) | p.Empty()

def DelimitedMultiList(elements, delimiter):
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
    rest = DelimitedMultiList(elements[1:], delimiter)
    if unbounded:
        result = p.Forward()
        result << ((expr + delimiter + result) | expr | rest)
    else:
        if isinstance(rest, p.Empty):
            result = expr | rest
        else:
            result = (expr + delimiter + rest) | expr | rest
    return result

class CompactSyntaxParser(object):
    """A parser that converts a compact textual syntax for protocols into XML."""
    # Single-line Python-style comments
    comment = p.Regex(r'#.*').setWhitespaceChars(' \t').suppress().setName('Comment')

    # Punctuation etc.
    eq = p.Suppress('=')
    colon = p.Suppress(':')
    nl = p.OneOrMore(Optional(comment) + p.LineEnd().suppress()) # Any line can end with a comment
    obrace = Optional(nl) + p.Suppress('{') + Optional(nl)
    cbrace = Optional(nl) + p.Suppress('}') + Optional(nl)
    
    # Numbers can be given in scientific notation, with an optional leading minus sign.
    number = p.Regex(r'-?[0-9]+((\.[0-9]+)?(e[-+]?[0-9]+)?)?').setName('Number')
    
    # Identifiers
    ncIdent = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*').setName('ncIdent')
    ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*(:[_a-zA-Z][_0-9a-zA-Z]*)*').setName('Ident')
    # This may become more specific in future
    quotedUri = (p.QuotedString('"') | p.QuotedString("'")).setName('QuotedUri')
    quotedString = p.quotedString.setName('QuotedString').setParseAction(p.removeQuotes)
    
    # Basic expressions from the "post-processing" language
    #######################################################
    
    # Expressions must be constructed recursively
    expr = p.Forward().setName('Expression')
    
    # A vector written like 1:2:5 or 1:5
    numericRange = p.Group(number + colon + number + Optional(colon + number))
    
    # If-then-else
    ifExpr = p.Group(MakeKw('if') + expr + MakeKw('then') + expr + MakeKw('else') + expr)
    
    # The main expression grammar
    atom = number | ifExpr | ident
    expr << p.operatorPrecedence(atom, [('^', 2, p.opAssoc.LEFT),
                                        ('-', 1, p.opAssoc.RIGHT),
                                        (p.oneOf('* /'), 2, p.opAssoc.LEFT),
                                        (p.oneOf('+ -'), 2, p.opAssoc.LEFT),
                                        ('not', 1, p.opAssoc.RIGHT),
                                        (p.oneOf('== != <= >= < >'), 2, p.opAssoc.LEFT),
                                        (p.oneOf('&& ||'), 2, p.opAssoc.LEFT)
                                       ])

    # Newlines in expressions may be escaped with a backslash
    expr.ignore('\\' + p.LineEnd())
    # Embedded comments are also OK
    expr.ignore(comment)
    
    # Simple assignment
    simpleAssign = p.Group(ncIdent + eq + expr)
    simpleAssignList = OptionalDelimitedList(simpleAssign, nl)

    # Miscellaneous constructs making up protocols
    ##############################################
    
    # Namespace declarations
    nsDecl = p.Group(MakeKw('namespace') + ncIdent + eq + quotedUri)
    nsDecls = p.ZeroOrMore(nsDecl + nl)
    
    # Protocol input declarations, with default values
    inputs = MakeKw('inputs') + obrace + simpleAssignList + cbrace

    # Import statements
    importStmt = p.Group(MakeKw('import') + Optional(ncIdent + eq, default='') + quotedUri)
    imports = OptionalDelimitedList(importStmt, nl)
    
    # Model interface section
    #########################
    unitsRef = MakeKw('units') + ncIdent
    
    # Setting the units for the independent variable
    setTimeUnits = MakeKw('independent') + MakeKw('var') + unitsRef
    # Input variables, with optional units and initial value
    inputVariable = p.Group(MakeKw('input') + ident
                            + Optional(unitsRef, default='')
                            + Optional(eq + number, default=''))
    # Model outputs of interest, with optional units
    outputVariable = p.Group(MakeKw('output') + ident + Optional(unitsRef, default=''))
    # New variables added to the model, with optional initial value
    newVariable = p.Group(MakeKw('var') + ncIdent + unitsRef + Optional(eq + number, default=''))
    # Adding or replacing equations in the model
    modelEquation = p.Group(MakeKw('define') + ident + eq + expr)
    
    modelInterface = p.Group(MakeKw('model') + MakeKw('interface') + obrace +
                             DelimitedMultiList([(setTimeUnits, False),
                                                 (inputVariable, True),
                                                 (outputVariable, True),
                                                 (newVariable, True),
                                                 (modelEquation, True)], nl) + cbrace)
    
    # Simulation definitions
    ########################
    
    # Ranges
    uniformRange = MakeKw('uniform') + numericRange
    vectorRange = MakeKw('vector') + expr
    whileRange = MakeKw('while') + expr
    range = p.Group(MakeKw('range') + ncIdent + unitsRef + (uniformRange | vectorRange | whileRange))
    
    # Modifiers
    modifierWhen = MakeKw('at') + (MakeKw('start', False) |
                                   (MakeKw('each', False) + MakeKw('loop')) |
                                   MakeKw('end', False))
    setVariable = MakeKw('set') + ident + eq + expr
    saveState = MakeKw('save') + MakeKw('as') + ncIdent
    resetState = MakeKw('reset') + Optional(MakeKw('to') + ncIdent)
    modifier = p.Group(modifierWhen + p.Group(setVariable | saveState | resetState))
    modifiers = p.Group(MakeKw('modifiers') + obrace + OptionalDelimitedList(modifier, nl) + cbrace)
    
    # The simulations themselves
    basicSimContent = range + Optional(nl + modifiers)
    timecourseSim = p.Group(MakeKw('timecourse') + obrace + basicSimContent + cbrace)
    simulation = MakeKw('simulation') + Optional(ncIdent + eq, default='') + timecourseSim


def GetNamedGrammars():
    """Get an iterator over all the named grammars in CompactSyntaxParser."""
    for parser in dir(CompactSyntaxParser):
        parser = getattr(CompactSyntaxParser, parser)
        if isinstance(parser, p.ParserElement):
            yield parser

def EnableDebug(grammars=None):
    """Enable debugging of our (named) grammars."""
    def DisplayLoc(instring, loc):
        return " at loc " + str(loc) + "(%d,%d)" % ( p.lineno(loc,instring), p.col(loc,instring) )
    
    def SuccessDebugAction( instring, startloc, endloc, expr, toks ):
        print ("Matched " + str(expr) + " -> " + str(toks.asList()) + DisplayLoc(instring, endloc))
    
    def ExceptionDebugAction( instring, loc, expr, exc ):
        print ("Exception raised:" + str(exc) + DisplayLoc(instring, loc))

    for parser in grammars or GetNamedGrammars():
        parser.setDebugActions(None, SuccessDebugAction, ExceptionDebugAction)

def DisableDebug(grammars=None):
    """Stop debugging our (named) grammars."""
    for parser in grammars or GetNamedGrammars():
        parser.setDebug(False)

class Debug(object):
    """A Python 2.6+ context manager that enables debugging just for the enclosed block."""
    def __init__(self, grammars=None):
        self._grammars or GetNamedGrammars()
    def __enter__(self):
        EnableDebug(self._grammars)
    def __exit__(self, type, value, traceback):
        DisableDebug(self._grammars)
