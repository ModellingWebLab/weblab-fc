
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

def MakeKw(keyword):
    """Helper function to create a parser for the given keyword."""
    return p.Keyword(keyword).suppress()

class CompactSyntaxParser(object):
    """A parser that converts a compact textual syntax for protocols into XML."""
    # Single-line Python-style comments
    comment = p.Regex(r'#.*').setWhitespaceChars(' \t').suppress()

    # Punctuation etc.
    eq = p.Literal('=').suppress()
    nl = p.Optional(comment) + p.LineEnd().suppress() # Any line can end with a comment
    obrace = p.Suppress('{')
    cbrace = p.Suppress('}')
    
    # Numbers can be given in scientific notation, with an optional leading minus sign.
    number = p.Regex(r'-?[0-9]+((\.[0-9]+)?(e[-+]?[0-9]+)?)?').setName('Number')
    
    # Identifiers
    ncIdent = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*')
    ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*(:[_a-zA-Z][_0-9a-zA-Z]*)*')
    # Variable references may need to allow for extra characters in ontology terms in future
    varRef = ident
    # This may become more specific in future
    quotedUri = p.QuotedString('"') | p.QuotedString("'")
    
    # Basic expressions from the "post-processing" language
    #######################################################
    
    # Expressions must be constructed recursively
    expr = p.Forward().setName('Expression')
    
    # The main expression grammar
    atom = number | ident
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
    
    # Simple assignment
    simpleAssign = p.Group(ncIdent + eq + expr)
    simpleAssignList = p.ZeroOrMore(simpleAssign + nl)

    # Miscellaneous constructs making up protocols
    ##############################################
    
    # Namespace declarations
    nsDecl = p.Group(MakeKw('namespace') + ncIdent + eq + quotedUri)
    nsDecls = p.ZeroOrMore(nsDecl + nl)
    
    # Protocol input declarations, with default values
    inputs = MakeKw('inputs') + obrace + simpleAssignList + cbrace

    # Import statements
    importStmt = p.Group(MakeKw('import') + p.Optional(ncIdent + eq, default='') + quotedUri)
    imports = p.ZeroOrMore(importStmt + nl)
    
    # Model interface section
    #########################
    
    # Input variables, with optional units and initial value
    inputVariable = p.Group(MakeKw('input') + varRef
                            + p.Optional(MakeKw('units') + ncIdent, default='')
                            + p.Optional(eq + number, default=''))

