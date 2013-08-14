
"""Copyright (c) 2005-2013, University of Oxford.
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

import math
import numexpr as ne

from .abstract import AbstractExpression
from ...utility.error_handling import ProtocolError
from .. import values as V


class Plus(AbstractExpression):
    """Addition."""
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        if isinstance(operands[0], V.Array):
            arr_names = [env.FreshIdent() for i in range(len(operands))]
            arr_dict = {}
            for i,operand in enumerate(operands):
                arr_dict[arr_names[i]] = operand.array
            expression = ' + '.join(arr_names)
            result = V.Array(ne.evaluate(expression, local_dict=arr_dict))

        else:
            try:
                result = V.Simple(sum([v.value for v in operands]))
            except AttributeError:
                raise ProtocolError("Operator 'plus' requires all operands to evaluate to numbers;", v, "does not.")
        return result

    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' + '.join(operands)
        return expression

class Minus(AbstractExpression):
    """Subtraction."""
    def __init__(self, *children):
        super(Minus, self).__init__(*children)
        if len(self.children) != 1 and len(self.children) != 2:
            raise ProtocolError("Operator 'minus' requires one or two operands, not", len(self.children))

    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            if len(self.children) == 1:
                result = -operands[0].value
            else:
                result = operands[0].value - operands[1].value
        except AttributeError:
            raise ProtocolError("Operator 'minus' requires all operands to evaluate numbers")
        return V.Simple(result)

    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        if len(operands) == 1:
            expression = -operands[0]
        else:
            expression = ' - '.join(operands) 
        return expression

class Times(AbstractExpression):
    """Multiplication"""
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        if any(isinstance(operand, V.Array) for operand in operands):
            arr_names = [env.FreshIdent() for i in range(len(operands))]
            arr_dict = {}
            for i,operand in enumerate(operands):
                try:
                    arr_dict[arr_names[i]] = operand.array
                except AttributeError:
                    arr_dict[arr_names[i]] = operand.value
            expression = ' * '.join(arr_names)
            result = V.Array(ne.evaluate(expression, local_dict=arr_dict))
        else:
            try:
                result = V.Simple(reduce(lambda x, y: x*y, [v.value for v in operands], 1))
            except AttributeError:
                raise ProtocolError("Operator 'times' requires all operands to evaluate to an Array or numbers;", v, "does not.")
        return result
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' * '.join(operands) 
        return expression
    
class Divide(AbstractExpression):
    """Division."""
    def __init__(self, *children):
        super(Divide, self).__init__(*children)
        if len(self.children) != 1 and len(self.children) != 2:
            raise ProtocolError("Operator 'divide' requires one or two operands, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = operands[0].value/operands[1].value
        except AttributeError:
            raise ProtocolError("Operator 'divide' requires all operands to evaluate to numbers")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' / '.join(operands) 
        return expression
    
class Max(AbstractExpression):
    """Returns maximum value."""
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = max([v.value for v in operands])
        except AttributeError:
            raise ProtocolError("Operator 'max' requires all operands to evaluate to numbers")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "___np.maximum(" + ','.join(operands) + ")"  
        return expression 
            
class Min(AbstractExpression):
    """Returns minimum value."""
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = min([v.value for v in operands])
        except AttributeError:
            raise ProtocolError("Operator 'min' requires all operands to evaluate to numbers")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "___np.minimum(" + ','.join(arr_names) + ")"  
        return expression 
    
class Rem(AbstractExpression):
    """Remainder operator."""
    def __init__(self, *children):
        super(Rem, self).__init__(*children)
        if len(self.children) != 2:
            raise ProtocolError("Operator 'rem' requires two operands, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = operands[0].value%operands[1].value
        except AttributeError:
            raise ProtocolError("Operator 'rem' requires all operands to evaluate to numbers")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' % '.join(operands)  
        return expression 
    
class Power(AbstractExpression):
    """Power operator."""
    def __init__(self, *children):
        super(Power, self).__init__(*children)
        if len(self.children) != 2:
            raise ProtocolError("Power operator requires two operands, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = operands[0].value ** operands[1].value
        except AttributeError:
            raise ProtocolError("Operator 'power' requires all operands to evaluate to numbers")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = operands[0] + ' ** ' + operands[1] 
        return expression 
    
class Root(AbstractExpression):
    """Root operator."""
    def __init__(self, *children):
        super(Root, self).__init__(*children)
        if len(self.children) != 1 and len(self.children) != 2:
            raise ProtocolError("Operator 'root' requires one operand, optionally with a degree qualifier, you entered", len(self.children), "inputs")
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            if len(self.children) == 1:
                result = operands[0].value ** .5
            else:
                result = operands[1].value ** (1.0/operands[0].value)
        except AttributeError:
            raise ProtocolError("Operator 'root' requires its operand to evaluate to a number")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        if len(operands) == 1:
            expression = operands[0] + "** .5"
        elif len(operands) == 2:
            expression = operands[0] + "** (1/" + operands[1] + ")"
        return expression 
    
class Abs(AbstractExpression):
    """Absolute value operator."""
    def __init__(self, *children):
        super(Abs, self).__init__(*children)
        if len(self.children) != 1:
            raise ProtocolError("Operator 'absolute value' requires one operand, not", len(self.children))
            
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = abs(operands[0].value)
        except AttributeError:
            raise ProtocolError("Operator 'absolute value' requires its operand to evaluate to a number")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "abs(" + operands[0] + ")"
        return expression 

class Floor(AbstractExpression):
    """Floor operator."""
    def __init__(self, *children):
        super(Floor, self).__init__(*children)
        if len(self.children) != 1:
            raise ProtocolError("Operator 'floor' requires one operand, not", len(self.children))
                
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = math.floor(operands[0].value)
        except AttributeError:
            raise ProtocolError("Operator 'floor' requires its operand to evaluate to a number")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "___np.floor(" + operands[0] + ")"
        return expression 
    
class Ceiling(AbstractExpression):
    """Ceiling operator."""
    def __init__(self, *children):
        super(Ceiling, self).__init__(*children)
        if len(self.children) != 1:
            raise ProtocolError("Operator 'ceiling' requires one operand, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = math.ceil(operands[0].value)
        except AttributeError:
            raise ProtocolError("Operator 'ceiling' requires its operand to evaluate to a number")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "___np.ceil(" + operands[0] + ")"
        return expression

class Exp(AbstractExpression):
    """Exponential operator."""
    def __init__(self, *children):
        super(Exp, self).__init__(*children)
        if len(self.children) != 1:
            raise ProtocolError("Operator 'exp' requires one operand, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = math.exp(operands[0].value)
        except AttributeError:
            raise ProtocolError("Operator 'exp' requires a number as its operand")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "exp(" + operands[0] + ")"
        return expression 
    
class Ln(AbstractExpression):
    """Natural logarithm operator."""
    def __init__(self, *children):
        super(Ln, self).__init__(*children)
        if len(self.children) != 1:
            raise ProtocolError("Operator 'ln' requires one operand, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = math.log(operands[0].value)
        except AttributeError:
            raise ProtocolError("Natural logarithm operator requires a number as its operand")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "log(" + operands[0] + ")"
        return expression 
        
class Log(AbstractExpression):
    """logarithmic operator."""
    def __init__(self, *children):
        super(Log, self).__init__(*children)
        if len(self.children) != 1 and len(self.children) != 2:
            raise ProtocolError("Logarithmic operator requires one operand, and optionally a log_base qualifier, you entered", len(self.children), "inputs")
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        log_base = 10
        if len(self.children) == 2:
            log_base = operands[0].value
        try:
            if log_base == 10:
                result = math.log10(operands[0].value)
            else:
                result = math.log(operands[1].value,log_base)
        except AttributeError:
            raise ProtocolError("Logarithm operator requires its operands to evaluate to numbers")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        if len(operands) == 1:
            expression = "log10(" + operands[0] + ")"
        elif len(operands) == 2:
            expression = "log10(" + operands[1] + ") / log10(" + operands[0] + ")"
        return expression 
    
class And(AbstractExpression):
    """Boolean And Operator"""
    def Interpret(self, env):
        if len(self.children) == 0:
            raise ProtocolError("Boolean operator 'and' requires operands")
        result = True
        for child in self.children:
            v = child.Evaluate(env)
            try:
                result = result and v.value
            except AttributeError:
                raise ProtocolError("Boolean operator 'and' requires its operands to be simple values")
            if not v.value:
                break
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "___np.logical_and(" + ','.join(operands) + ")"  
        return expression 
    
class Or(AbstractExpression):
    """Boolean Or Operator"""
    def Interpret(self, env):
        if len(self.children) == 0:
            raise ProtocolError("Boolean operator 'or' requires operands")
        result = False
        for child in self.children:
            v = child.Evaluate(env)
            try:
                result = result or v.value
            except AttributeError:
                raise ProtocolError("Boolean operator 'or' requires its operands to be simple values")
            if v.value:
                break
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "___np.logical_or(" + ','.join(operands) + ")"  
        return expression 

class Xor(AbstractExpression):
    """Boolean Xor Operator"""
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) == 0:
            raise ProtocolError("Boolean operator 'xor' requires operands")
        result = False
        try:
            for v in operands: 
                result = result != v.value
        except AttributeError:
            raise ProtocolError("Boolean operator 'xor' requires its operands to be simple values")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "___np.logical_xor(" + ','.join(operands) + ")"  
        return expression 

class Not(AbstractExpression):
    """Boolean Not Operator"""
    def __init__(self, *children):
        super(Not, self).__init__(*children)
        if len(self.children) != 1:
            raise ProtocolError("Boolean operator 'not' requires one operand, not", len(self.children))
    
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = not operands[0].value
        except AttributeError:
            raise ProtocolError("Boolean operator 'not' requires its operand to be a simple value")
        return V.Simple(result) 
    
    def Compile(self):
        operands = [ "bool(" + child.Compile() + ")" for child in self.children]
        expression = "___np.logical_not(" + ','.join(operands) + ")"  
        return expression 
    
class Eq(AbstractExpression):
    """Equality Operator"""
    def __init__(self, *children):
        super(Eq, self).__init__(*children)
        if len(self.children) != 2:
            raise ProtocolError("Boolean operator 'equal' requires two operands, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = operands[0].value == operands[1].value
        except AttributeError:
            raise ProtocolError("Equality operator requires its operands to be simple values")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' == '.join(operands) 
        return expression
    
class Neq(AbstractExpression):
    """Not equal Operator"""
    def __init__(self, *children):
        super(Neq, self).__init__(*children)
        if len(self.children) != 2:
            raise ProtocolError("Boolean operator 'not equal' requires two operands, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = operands[0].value != operands[1].value
        except AttributeError:
            raise ProtocolError("Not equal operator requires its operands to be simple values")
        return V.Simple(result) 
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' != '.join(operands)  
        return expression 
    
class Lt(AbstractExpression):
    """Less than Operator"""
    def __init__(self, *children):
        super(Lt, self).__init__(*children)
        if len(self.children) != 2:
            raise ProtocolError("Boolean operator 'less than' requires two operands, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = operands[0].value < operands[1].value
        except AttributeError:
            raise ProtocolError("Less than operator requires its operands to be simple values, you entered a", type(operands[0]), 'and', type(operands[1]))
        return V.Simple(result)       
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' < '.join(operands)  
        return expression 
    
class Gt(AbstractExpression):
    """Greater than Operator"""
    def __init__(self, *children):
        super(Gt, self).__init__(*children)
        if len(self.children) != 2:
            raise ProtocolError("Boolean operator 'greater than' requires two operands, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = operands[0].value > operands[1].value
        except AttributeError:
            raise ProtocolError("Greater than operator requires its operands to be simple values")
        return V.Simple(result) 
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' > '.join(operands)  
        return expression 
     
class Leq(AbstractExpression):
    """Less than or equal to Operator"""
    def __init__(self, *children):
        super(Leq, self).__init__(*children)
        if len(self.children) != 2:
            raise ProtocolError("Boolean operator 'less than or equal to' requires two operands, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = operands[0].value <= operands[1].value
        except AttributeError:
            raise ProtocolError("Less than or equal to operator requires its operands to be simple values")
        return V.Simple(result) 
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' <= '.join(operands)  
        return expression 
    
class Geq(AbstractExpression):
    """Greater than or equal to Operator"""
    def __init__(self, *children):
        super(Geq, self).__init__(*children)
        if len(self.children) != 2:
            raise ProtocolError("Boolean operator 'greater than or equal to' requires two operands, not", len(self.children))
        
    def Interpret(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = operands[0].value >= operands[1].value
        except AttributeError:
            raise ProtocolError("Greater than or equal to operator requires its operands to be simple values")
        return V.Simple(result) 
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' >= '.join(operands)  
        return expression  