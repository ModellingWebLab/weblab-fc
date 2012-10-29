
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

# TODO: We may be able to get rid of some of the p.Group wrapping when we add parse actions

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

def Adjacent(parser):
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

def UnIgnore(parser):
    """Stop ignoring things in the given parser."""
    for child in getattr(parser, 'exprs', []):
        UnIgnore(child)
    if hasattr(parser, 'expr'):
        UnIgnore(parser.expr)
    parser.ignoreExprs = []

def MonkeyPatch():
    """Monkey-patch some pyparsing methods to behave slightly differently."""
    def ignore( self, other ):
        """Improved ignore that avoids ignoring self by accident."""
        if isinstance( other, p.Suppress ):
            if other not in self.ignoreExprs and other != self:
                self.ignoreExprs.append( other.copy() )
        else:
            self.ignoreExprs.append( p.Suppress( other.copy() ) )
        return self

    import new
    setattr(p.ParserElement, 'ignore', new.instancemethod(locals()['ignore'], None, p.ParserElement))

MonkeyPatch()


class CompactSyntaxParser(object):
    """A parser that converts a compact textual syntax for protocols into XML."""
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
    nl = p.Suppress(p.OneOrMore(Optional(comment) + p.LineEnd())) # Any line can end with a comment
    obrace = Optional(nl) + p.Suppress('{') + Optional(nl)
    cbrace = Optional(nl) + p.Suppress('}') + Optional(nl)
    
    # Numbers can be given in scientific notation, with an optional leading minus sign.
    number = p.Regex(r'-?[0-9]+((\.[0-9]+)?(e[-+]?[0-9]+)?)?').setName('Number')
    
    # Identifiers
    ncIdent = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*').setName('ncIdent')
    ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*(:[_a-zA-Z][_0-9a-zA-Z]*)*').setName('Ident')
    # Used for descriptive text
    quotedString = (p.QuotedString('"', escChar="\\") | p.QuotedString("'", escChar="\\")).setName('QuotedString')
    # This may become more specific in future
    quotedUri = quotedString.copy().setName('QuotedUri')
    
    # Expressions from the "post-processing" language
    #################################################
    
    # Expressions and statements must be constructed recursively
    expr = p.Forward().setName('Expression')
    stmtList = p.Forward().setName('StatementList')
    
    # A vector written like 1:2:5 or 1:5
    numericRange = p.Group(number + colon + number + Optional(colon + number))

    # Creating arrays
    dimSpec = Optional(expr + Adjacent(dollar)) + ncIdent
    comprehension = p.Group(MakeKw('for') + dimSpec + MakeKw('in') +
                            expr + colon + expr + Optional(colon + expr))
    array = p.Group(osquare + expr + (p.OneOrMore(comprehension) | p.ZeroOrMore(comma + expr)) + csquare)
    
    # Array views
    optExpr = Optional(expr, default='')
    viewSpec = p.Group(Adjacent(osquare) + Optional(('*' | expr) + Adjacent(dollar)) +
                       optExpr + Optional(colon + optExpr + Optional(colon + optExpr)) + csquare)
    
    # If-then-else
    ifExpr = p.Group(MakeKw('if') + expr + MakeKw('then') + expr + MakeKw('else') + expr)
    
    # Lambda definitions
    paramDecl = p.Group(ncIdent + Optional(eq + expr)) # TODO: check we can write XML for a full expr as default value
    paramList = p.Group(OptionalDelimitedList(paramDecl, comma))
    lambdaExpr = p.Group(MakeKw('lambda') + paramList + ((colon + expr) | (obrace + stmtList + cbrace)))
    
    # Function calls
    argList = p.Group(OptionalDelimitedList(expr, comma))
    functionCall = p.Group(ident + Adjacent(oparen) + argList + cparen) # TODO: allow lambdas, not just ident?
    
    # Tuples
    tuple = p.Group(oparen + expr + comma + OptionalDelimitedList(expr, comma) + cparen)
    
    # Accessors
    accessor = p.Combine(Adjacent(p.Suppress('.')) +
                         p.oneOf('IS_SIMPLE_VALUE IS_ARRAY IS_STRING IS_TUPLE IS_FUNCTION IS_NULL IS_DEFAULT NUM_DIMS NUM_ELEMENTS SHAPE'))

    # Indexing
    pad = MakeKw('pad') + Adjacent(colon) + expr + eq + expr
    shrink = MakeKw('shrink') + Adjacent(colon) + expr
    index = p.Group(Adjacent(p.Suppress('{')) + expr + Optional(comma + (pad|shrink)) + p.Suppress('}'))

    # Special values
    nullValue = p.Group(MakeKw('null'))
    defaultValue = p.Group(MakeKw('default'))
    
    # Recognised MathML operators
    mathmlOperators = set('''quotient rem max min root xor abs floor ceiling exp ln log
                             sin cos tan   sec csc cot   sinh cosh tanh   sech csch coth
                             arcsin arccos arctan   arccosh arccot arccoth
                             arccsc arccsch arcsec   arcsech arcsinh arctanh'''.split())

    # Wrapping MathML operators into lambdas
    mathmlOperator = (p.oneOf('^ * / + - not == != <= >= < > && ||') |
                      p.Combine('MathML.' + p.oneOf(' '.join(mathmlOperators))))
    wrap = p.Group(p.Suppress('@') + Adjacent(p.Word(p.nums)) + Adjacent(colon) + mathmlOperator)
    
    # The main expression grammar.  Atoms are ordered according to rough speed of detecting mis-match.
    atom = array | wrap | number | ifExpr | nullValue | defaultValue | lambdaExpr | functionCall | ident | tuple
    expr << p.operatorPrecedence(atom, [(accessor, 1, p.opAssoc.LEFT),
                                        (viewSpec, 1, p.opAssoc.LEFT),
                                        (index, 1, p.opAssoc.LEFT),
                                        ('^', 2, p.opAssoc.LEFT),
                                        ('-', 1, p.opAssoc.RIGHT),
                                        (p.oneOf('* /'), 2, p.opAssoc.LEFT),
                                        (p.oneOf('+ -'), 2, p.opAssoc.LEFT),
                                        ('not', 1, p.opAssoc.RIGHT),
                                        (p.oneOf('== != <= >= < >'), 2, p.opAssoc.LEFT),
                                        (p.oneOf('&& ||'), 2, p.opAssoc.LEFT)
                                       ])

    # Newlines in expressions may be escaped with a backslash
    expr.ignore('\\' + p.LineEnd())
    # Bare newlines are OK provided we started with a bracket.
    # However, it's quite hard to enforce that restriction.
    expr.ignore(p.Literal('\n'))
    # Embedded comments are also OK
    expr.ignore(comment)
    # Avoid mayhem
    UnIgnore(nl)
    
    # Statements from the "post-processing" language
    ################################################
    
    # Simple assignment (i.e. not to a tuple)
    simpleAssign = p.Group(ncIdent + eq + expr)
    simpleAssignList = OptionalDelimitedList(simpleAssign, nl)
    
    # Assertions and function returns
    assertStmt = p.Group(MakeKw('assert') + expr)
    returnStmt = p.Group(MakeKw('return') + p.delimitedList(expr))
    
    # Full assignment, to a tuple of names or single name
    assignStmt = p.Group(p.Group(p.delimitedList(ncIdent)) + eq + p.Group(p.delimitedList(expr)))
    
    # Function definition
    functionDefn = p.Group(MakeKw('def') + ncIdent + oparen + paramList + cparen +
                           ((colon + expr) | (obrace + stmtList + Optional(nl) + p.Suppress('}'))))
    
    stmtList << p.delimitedList(assertStmt | returnStmt | functionDefn | assignStmt, nl)

    # Miscellaneous constructs making up protocols
    ##############################################
    
    # Namespace declarations
    nsDecl = p.Group(MakeKw('namespace') + ncIdent + eq + quotedUri)
    nsDecls = p.ZeroOrMore(nsDecl + nl)
    
    # Protocol input declarations, with default values
    inputs = MakeKw('inputs') + obrace + simpleAssignList + cbrace

    # Import statements & use-imports
    importStmt = p.Group(MakeKw('import') + Optional(ncIdent + eq, default='') + quotedUri)
    imports = OptionalDelimitedList(importStmt, nl)
    useImports = p.Group(MakeKw('use') + MakeKw('imports') + ncIdent)
    
    # Library, globals defined using post-processing language.
    # Strictly speaking returns aren't allowed, but that gets picked up later.
    library = MakeKw('library') + obrace + Optional(stmtList) + cbrace
    
    # Units definitions
    siPrefix = p.oneOf('deka hecto kilo mega giga tera peta exa zetta yotta'
                       'deci centi milli micro nano pico femto atto zepto yocto')
    _num_or_expr = number | (oparen + expr + cparen)
    unitRef = p.Group(Optional(_num_or_expr, '1') + Optional(siPrefix, '') + ncIdent + Optional(p.Suppress('^') + number, '1')
                      + Optional(p.Group(p.oneOf('- +') + _num_or_expr)))
    unitsDef = p.Group(ncIdent + eq + p.delimitedList(unitRef, '.'))
    units = MakeKw('units') + obrace + OptionalDelimitedList(useImports | unitsDef, nl) + cbrace
    
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
    # Units conversion rules
    unitsConversion = p.Group(MakeKw('convert') + ncIdent + MakeKw('to') + ncIdent +
                              MakeKw('by') + lambdaExpr)
    
    modelInterface = p.Group(MakeKw('model') + MakeKw('interface') + obrace +
                             DelimitedMultiList([(useImports, True),
                                                 (setTimeUnits, False),
                                                 (inputVariable, True),
                                                 (outputVariable, True),
                                                 (newVariable, True),
                                                 (modelEquation, True),
                                                 (unitsConversion, True)], nl) + cbrace)
    
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
    timecourseSim = p.Group(MakeKw('timecourse') + obrace + range + Optional(nl + modifiers) + cbrace)
    nestedSim = p.Group(MakeKw('nested') + obrace + range + nl + Optional(modifiers) +
                        p.Group(MakeKw('nests') + ident) + cbrace)
    simulation = MakeKw('simulation') + Optional(ncIdent + eq, default='') + (timecourseSim | nestedSim)

    # Output specifications
    #######################
    
    outputDesc = Optional(quotedString, default='')
    outputSpec = p.Group(ncIdent + ((unitsRef + outputDesc) |
                                    (eq + ident + Optional(unitsRef, default='') + outputDesc)))
    outputs = p.Group(MakeKw('outputs') + obrace + OptionalDelimitedList(useImports | outputSpec, nl) + cbrace)

    # Plot specifications
    #####################
    
    plotCurve = p.Group(p.delimitedList(ncIdent, ',') + MakeKw('against') + ncIdent)
    plotSpec = p.Group(MakeKw('plot') + quotedString + obrace +
                       plotCurve + p.ZeroOrMore(nl + plotCurve) + cbrace)
    plots = p.Group(MakeKw('plots') + obrace + p.ZeroOrMore(useImports | plotSpec) + cbrace)


################################################################################
# Parser debugging support
################################################################################

def GetNamedGrammars(obj=CompactSyntaxParser):
    """Get a list of all the named grammars in the given object."""
    grammars = []
    for parser in dir(obj):
        parser = getattr(obj, parser)
        if isinstance(parser, p.ParserElement):
            grammars.append(parser)
    return grammars

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
        self._grammars = list(grammars or GetNamedGrammars())   
    def __enter__(self):
        EnableDebug(self._grammars)
    def __exit__(self, type, value, traceback):
        DisableDebug(self._grammars)
