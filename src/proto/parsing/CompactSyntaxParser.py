
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
# TODO: Allow units definitions to have a description, e.g. {/Symbol m}A/cm^2
# TODO: Nested protocols

import sys

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
    nl = p.Suppress(p.OneOrMore(Optional(comment) + p.LineEnd())).setName('Newline(s)') # Any line can end with a comment
    obrace = (Optional(nl) + p.Suppress('{') + Optional(nl)).setName('{')
    cbrace = (Optional(nl) + p.Suppress('}') + Optional(nl)).setName('}')
    embedded_cbrace = (Optional(nl) + p.Suppress('}')).setName('}')
    
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
    
    # A vector written like 1:2:5 or 1:5 or A:B:C
    numericRange = expr + colon + expr + Optional(colon + expr)

    # Creating arrays
    dimSpec = Optional(expr + Adjacent(dollar)) + ncIdent
    comprehension = p.Group(MakeKw('for') - dimSpec + MakeKw('in') + numericRange)
    array = p.Group(osquare + expr + (p.OneOrMore(comprehension) | p.ZeroOrMore(comma + expr)) + csquare).setName('Array')
    
    # Array views
    optExpr = Optional(expr, default='')
    viewSpec = p.Group(Adjacent(osquare) - Optional(('*' | expr) + Adjacent(dollar)) +
                       optExpr + Optional(colon + optExpr + Optional(colon + optExpr)) + csquare).setName('ViewSpec')
    
    # If-then-else
    ifExpr = p.Group(MakeKw('if') - expr + MakeKw('then') - expr + MakeKw('else') - expr).setName('IfThenElse')
    
    # Lambda definitions
    paramDecl = p.Group(ncIdent + Optional(eq + expr)) # TODO: check we can write XML for a full expr as default value
    paramList = p.Group(OptionalDelimitedList(paramDecl, comma))
    lambdaExpr = p.Group(MakeKw('lambda') - paramList + ((colon + expr) | (obrace + stmtList + embedded_cbrace))).setName('Lambda')
    
    # Function calls
    argList = p.Group(OptionalDelimitedList(expr, comma))
    functionCall = p.Group(ident + Adjacent(oparen) - argList + cparen).setName('FnCall') # TODO: allow lambdas, not just ident?
    
    # Tuples
    tuple = p.Group(oparen + expr + comma + OptionalDelimitedList(expr, comma) + cparen).setName('Tuple')
    
    # Accessors
    accessor = p.Combine(Adjacent(p.Suppress('.')) -
                         p.oneOf('IS_SIMPLE_VALUE IS_ARRAY IS_STRING IS_TUPLE IS_FUNCTION IS_NULL IS_DEFAULT '
                                 'NUM_DIMS NUM_ELEMENTS SHAPE')).setName('Accessor')

    # Indexing
    pad = MakeKw('pad') + Adjacent(colon) - expr + eq + expr
    shrink = MakeKw('shrink') + Adjacent(colon) - expr
    index = p.Group(Adjacent(p.Suppress('{')) - expr + Optional(comma + (pad|shrink)) + p.Suppress('}')).setName('Index')

    # Special values
    nullValue = p.Group(MakeKw('null')).setName('Null')
    defaultValue = p.Group(MakeKw('default')).setName('Default')
    
    # Recognised MathML operators
    mathmlOperators = set('''quotient rem max min root xor abs floor ceiling exp ln log
                             sin cos tan   sec csc cot   sinh cosh tanh   sech csch coth
                             arcsin arccos arctan   arccosh arccot arccoth
                             arccsc arccsch arcsec   arcsech arcsinh arctanh'''.split())

    # Wrapping MathML operators into lambdas
    mathmlOperator = (p.oneOf('^ * / + - not == != <= >= < > && ||') |
                      p.Combine('MathML:' + p.oneOf(' '.join(mathmlOperators))))
    wrap = p.Group(p.Suppress('@') - Adjacent(p.Word(p.nums)) + Adjacent(colon) + mathmlOperator).setName('WrapMathML')
    
    # The main expression grammar.  Atoms are ordered according to rough speed of detecting mis-match.
    atom = (array | wrap | number | ifExpr | nullValue | defaultValue | lambdaExpr | functionCall | ident | tuple).setName('Atom')
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
    assertStmt = p.Group(MakeKw('assert') - expr).setName('AssertStmt')
    returnStmt = p.Group(MakeKw('return') - p.delimitedList(expr)).setName('ReturnStmt')
    
    # Full assignment, to a tuple of names or single name
    assignStmt = p.Group(p.Group(p.delimitedList(ncIdent)) + eq + p.Group(p.delimitedList(expr))).setName('AssignStmt')
    
    # Function definition
    functionDefn = p.Group(MakeKw('def') - ncIdent + oparen + paramList + cparen +
                           ((colon + expr) | (obrace + stmtList + Optional(nl) + p.Suppress('}')))).setName('FunctionDef')
    
    stmtList << p.delimitedList(assertStmt | returnStmt | functionDefn | assignStmt, nl)

    # Miscellaneous constructs making up protocols
    ##############################################
    
    # Namespace declarations
    nsDecl = p.Group(MakeKw('namespace') - ncIdent + eq + quotedUri).setName('NamespaceDecl')
    nsDecls = OptionalDelimitedList(nsDecl, nl)
    
    # Protocol input declarations, with default values
    inputs = (MakeKw('inputs') + obrace - simpleAssignList + cbrace).setName('Inputs')

    # Import statements & use-imports
    importStmt = p.Group(MakeKw('import') - Optional(ncIdent + eq, default='') + quotedUri +
                         Optional(obrace - simpleAssignList + embedded_cbrace)).setName('Import')
    imports = OptionalDelimitedList(importStmt, nl).setName('Imports')
    useImports = p.Group(MakeKw('use') + MakeKw('imports') - ncIdent).setName('UseImports')
    
    # Library, globals defined using post-processing language.
    # Strictly speaking returns aren't allowed, but that gets picked up later.
    library = (MakeKw('library') + obrace - Optional(stmtList) + cbrace).setName('Library')
    
    # Post-processing
    postProcessing = (MakeKw('post-processing') + obrace - 
                      OptionalDelimitedList(useImports | assertStmt | returnStmt | functionDefn | assignStmt, nl) +
                      cbrace).setName('PostProc')
    
    # Units definitions
    siPrefix = p.oneOf('deka hecto kilo mega giga tera peta exa zetta yotta'
                       'deci centi milli micro nano pico femto atto zepto yocto')
    _num_or_expr = number | (oparen + expr + cparen)
    unitRef = p.Group(Optional(_num_or_expr, '1') + Optional(siPrefix, '') + ncIdent + Optional(p.Suppress('^') + number, '1')
                      + Optional(p.Group(p.oneOf('- +') + _num_or_expr)))
    unitsDef = p.Group(ncIdent + eq + p.delimitedList(unitRef, '.')).setName('UnitsDefinition')
    units = (MakeKw('units') + obrace - OptionalDelimitedList(useImports | unitsDef, nl) + cbrace).setName('Units')
    
    # Model interface section
    #########################
    unitsRef = MakeKw('units') + ncIdent
    
    # Setting the units for the independent variable
    setTimeUnits = MakeKw('independent') + MakeKw('var') - unitsRef
    # Input variables, with optional units and initial value
    inputVariable = p.Group(MakeKw('input') - ident
                            + Optional(unitsRef, default='')
                            + Optional(eq + number, default='')).setName('InputVariable')
    # Model outputs of interest, with optional units
    outputVariable = p.Group(MakeKw('output') - ident + Optional(unitsRef, default='')).setName('OutputVariable')
    # New variables added to the model, with optional initial value
    newVariable = p.Group(MakeKw('var') - ncIdent + unitsRef + Optional(eq + number, default='')).setName('NewVariable')
    # Adding or replacing equations in the model
    modelEquation = p.Group(MakeKw('define') - ident + eq + expr).setName('AddOrReplaceEquation')
    # Units conversion rules
    unitsConversion = p.Group(MakeKw('convert') - ncIdent + MakeKw('to') + ncIdent +
                              MakeKw('by') - lambdaExpr).setName('UnitsConversion')
    
    modelInterface = p.Group(MakeKw('model') + MakeKw('interface') + obrace -
                             DelimitedMultiList([(useImports, True),
                                                 (setTimeUnits, False),
                                                 (inputVariable, True),
                                                 (outputVariable, True),
                                                 (newVariable, True),
                                                 (modelEquation, True),
                                                 (unitsConversion, True)], nl) + cbrace).setName('ModelInterface')
    
    # Simulation definitions
    ########################
    
    # Ranges
    uniformRange = MakeKw('uniform') + p.Group(numericRange)
    vectorRange = MakeKw('vector') + expr
    whileRange = MakeKw('while') + expr
    range = p.Group(MakeKw('range') + ncIdent + unitsRef + (uniformRange | vectorRange | whileRange)).setName('Range')
    
    # Modifiers
    modifierWhen = MakeKw('at') - (MakeKw('start', False) |
                                   (MakeKw('each', False) + MakeKw('loop')) |
                                   MakeKw('end', False))
    setVariable = MakeKw('set') - ident + eq + expr
    saveState = MakeKw('save') + MakeKw('as') + ncIdent
    resetState = MakeKw('reset') + Optional(MakeKw('to') + ncIdent)
    modifier = p.Group(modifierWhen + p.Group(setVariable | saveState | resetState)).setName('Modifier')
    modifiers = p.Group(MakeKw('modifiers') + obrace - OptionalDelimitedList(modifier, nl) + cbrace).setName('Modifiers')
    
    # The simulations themselves
    simulation = p.Forward().setName('Simulation')
    timecourseSim = p.Group(MakeKw('timecourse') + obrace + range + Optional(nl + modifiers) + cbrace).setName('TimecourseSim')
    nestedSim = p.Group(MakeKw('nested') + obrace + range + nl + Optional(modifiers) +
                        p.Group(MakeKw('nests') + (simulation | ident)) + cbrace).setName('NestedSim')
    simulation << MakeKw('simulation') - Optional(ncIdent + eq, default='') + (timecourseSim | nestedSim)

    tasks = p.Group(MakeKw('tasks') + obrace - p.ZeroOrMore(p.Group(simulation)) + cbrace).setName('Tasks')

    # Output specifications
    #######################
    
    outputDesc = Optional(quotedString, default='')
    outputSpec = p.Group(ncIdent + ((unitsRef + outputDesc) |
                                    (eq + ident + Optional(unitsRef, default='') + outputDesc))).setName('Output')
    outputs = p.Group(MakeKw('outputs') + obrace - OptionalDelimitedList(useImports | outputSpec, nl) + cbrace).setName('Outputs')

    # Plot specifications
    #####################
    
    plotCurve = p.Group(p.delimitedList(ncIdent, ',') + MakeKw('against') + ncIdent).setName('Curve')
    plotSpec = p.Group(MakeKw('plot') - quotedString + obrace +
                       plotCurve + p.ZeroOrMore(nl + plotCurve) + cbrace).setName('Plot')
    plots = p.Group(MakeKw('plots') + obrace - p.ZeroOrMore(useImports | plotSpec) + cbrace).setName('Plots')
    
    # Parsing a full protocol
    #########################
    
    protocol = p.And(map(Optional, [nl, nsDecls + nl, inputs, imports + nl, library, units, modelInterface,
                                    tasks, postProcessing, outputs, plots])).setName('Protocol')
    
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
    
    def _Try(self, callable, *args, **kwargs):
        """Try calling the given parse command, increasing the stack depth limit if needed."""
        r = None # Result
        while self._stack_depth_factor < 3:
            try:
                r = callable(*args, **kwargs)
            except RuntimeError, msg:
                print >> sys.stderr, "Got RuntimeError:", msg
                self._stack_depth_factor += 0.5
                new_limit = int(self._stack_depth_factor * self._original_stack_limit)
                print >> sys.stderr, "Increasing recursion limit to", new_limit
                sys.setrecursionlimit(new_limit)
            else:
                break # Parsed OK
        if not r:
            raise RuntimeError("Failed to parse expression even with a recursion limit of %d; giving up!"
                               % (int(self._stack_depth_factor * self._original_stack_limit),))
        return r
    
    def ParseFile(self, fileOrFilename):
        """Main entry point for parsing a complete protocol file."""
        return self._Try(self.protocol.parseFile, fileOrFilename, parseAll=True)
    

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
