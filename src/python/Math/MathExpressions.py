
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

import Values as V
import numexpr as ne
import math
from ErrorHandling import ProtocolError
from AbstractExpression import AbstractExpression

class Const(AbstractExpression):
    """Class for constant value as expression."""
    def __init__(self, value):
        self.value = value
        
    def Evaluate(self,env):
        return self.value    
        
class Plus(AbstractExpression):
    """Addition."""
    def Evaluate(self, env):
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
                raise ProtocolError("Operator 'plus' requires all operands to evaluate to numbers or an Array;", v, "does not.")
        return result
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' + '.join(operands)  
        return expression 
    
class Minus(AbstractExpression):
    """Subtraction."""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 1 and len(self.children) != 2:
            raise ProtocolError("Operator 'minus' requires one or two operands, not", len(self.children))
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
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if isinstance(operands[0], V.Array):
            arr_names = [env.FreshIdent() for i in range(len(operands))]
            arr_dict = {}            
            for i,operand in enumerate(operands):
                arr_dict[arr_names[i]] = operand.array
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
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Operator 'divide' requires two operands, not", len(self.children))
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
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = max([v.value for v in operands])
        except AttributeError:
            raise ProtocolError("Operator 'max' requires all operands to evaluate to numbers")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "numpy.maximum(" + ','.join(operands) + ")"  
        return expression 
            
class Min(AbstractExpression):
    """Returns minimum value."""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        try:
            result = min([v.value for v in operands])
        except AttributeError:
            raise ProtocolError("Operator 'min' requires all operands to evaluate to numbers")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "numpy.minimum(" + ','.join(arr_names) + ")"  
        return expression 
    
class Rem(AbstractExpression):
    """Remainder operator."""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Operator 'rem' requires two operands, not", len(self.children))
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
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Operator 'power' requires two operands, not", len(self.children))
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
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 1 and len(self.children) != 2:
            raise ProtocolError("Operator 'root' requires one operand, optionally with a degree qualifier, you entered", len(self.children), "inputs")
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
        else:
            raise ProtocolError("Root operator requires one argument, (not",len(operands), ") optionally with a degree qualifier")
        return expression 
    
class Abs(AbstractExpression):
    """Absolute value operator."""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 1:
            raise ProtocolError("Operator 'absolute value' requires one operand, not", len(self.children))
        try:
            result = abs(operands[0].value)
        except AttributeError:
            raise ProtocolError("Operator 'absolute value' requires its operand to evaluate to a number")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        if len(operands) == 1:
            expression = "abs(" + operands[0] + ")"
        else:
            raise ProtocolError("abs operator requires one argument, not", len(operands))
        return expression 

class Floor(AbstractExpression):
    """Floor operator."""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 1:
            raise ProtocolError("Operator 'floor' requires one operand, not", len(self.children))
        try:
            result = math.floor(operands[0].value)
        except AttributeError:
            raise ProtocolError("Operator 'floor' requires its operand to evaluate to a number")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        if len(operands) == 1:
            expression = "numpy.floor(" + operands[0] + ")"
        else:
            raise ProtocolError("floor operator requires one argument, not", len(operands))
        return expression 
    
class Ceiling(AbstractExpression):
    """Ceiling operator."""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 1:
            raise ProtocolError("Operator 'ceiling' requires one operand, not", len(self.children))
        try:
            result = math.ceil(operands[0].value)
        except AttributeError:
            raise ProtocolError("Operator 'ceiling' requires its operand to evaluate to a number")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        if len(operands) == 1:
            expression = "numpy.ceil(" + operands[0] + ")"
        else:
            raise ProtocolError("ceiling operator requires one argument, not", len(operands))
        return expression

class Exp(AbstractExpression):
    """Exponential operator."""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 1:
            raise ProtocolError("Exponential operator requires one operand, not", len(self.children))
        try:
            result = math.exp(operands[0].value)
        except AttributeError:
            raise ProtocolError("Operator 'exp' requires a number as its operand")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        if len(operands) == 1:
            expression = "exp(" + operands[0] + ")"
        else:
            raise ProtocolError("exp operator requires one argument, not", len(operands))
        return expression 
    
    
class Ln(AbstractExpression):
    """Natural logarithm operator."""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 1:
            raise ProtocolError("Natural logarithm operator requires one operand, not", len(self.children))
        try:
            result = math.log(operands[0].value)
        except AttributeError:
            raise ProtocolError("Natural logarithm operator requires a number as its operand")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        if len(operands) == 1:
            expression = "log(" + operands[0] + ")"
        else:
            raise ProtocolError("ln operator requires one argument, not", len(operands))
        return expression 
        
class Log(AbstractExpression):
    """logarithmic operator."""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 1 and len(self.children) != 2:
            raise ProtocolError("Logarithmic operator requires one operand and optionally a log_base qualifier, you entered", len(self.children), "inputs")
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
        else:
            raise ProtocolError("log operator requires one argument, (not", len(operands)), ") optionally with a log-base qualifier"
        return expression 
    
class And(AbstractExpression):
    """Boolean And Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) == 0:
            raise ProtocolError("Boolean operator 'and' requires operands")
        result = True
        try:
            for v in operands:
                result = result and v.value
        except AttributeError:
            raise ProtocolError("Boolean operator 'and' requires its operands to be simple values")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "numpy.logical_and(" + ','.join(operands) + ")"  
        return expression 
    
class Or(AbstractExpression):
    """Boolean Or Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) == 0:
            raise ProtocolError("Boolean operator 'or' requires operands")
        result = False
        try:
            for v in operands:
                result = result or v.value
        except AttributeError:
            raise ProtocolError("Boolean operator 'or' requires its operands to be simple values")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "numpy.logical_or(" + ','.join(operands) + ")"  
        return expression 

class Xor(AbstractExpression):
    """Boolean Xor Operator"""
    def Evaluate(self, env):
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
        expression = "numpy.logical_xor(" + ','.join(operands) + ")"  
        return expression 

class Not(AbstractExpression):
    """Boolean Not Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 1:
            raise ProtocolError("Boolean operator 'not' requires 1 operand, not", len(self.children))
        try:
            result = not operands[0].value
        except AttributeError:
            raise ProtocolError("Boolean operator 'not' requires its operand to be a simple value")
        return V.Simple(result) 
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        if len(operands) == 1:
            expression = "numpy.logical_not(" + operands[0] + ")"
        else:
            raise ProtocolError("not operator requires one argument, not", len(operands))
        return expression 
    

    
class Eq(AbstractExpression):
    """Equality Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Equality operator requires 2 operands, not", len(self.children))
        try:
            result = operands[0].value == operands[1].value
        except AttributeError:
            raise ProtocolError("Equality operator requires its operands to be simple values")
        return V.Simple(result)
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = "numpy.equal(" + ','.join(operands) + ")"  
        return expression 
    
class Neq(AbstractExpression):
    """Not equal Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Not equal operator requires 2 operands, not", len(self.children))
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
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Less than operator requires 2 operands, not", len(self.children))
        try:
            result = operands[0].value < operands[1].value
        except AttributeError:
            raise ProtocolError("Less than operator requires its operands to be simple values")
        return V.Simple(result)       
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' < '.join(operands)  
        return expression 
     
    
class Gt(AbstractExpression):
    """Greater than Operator"""
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Greater than operator requires 2 operands, not", len(self.children))
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
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Less than or equal to operator requires 2 operands, not", len(self.children))
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
    def Evaluate(self, env):
        operands = self.EvaluateChildren(env)
        if len(self.children) != 2:
            raise ProtocolError("Greater than or equal to operator requires 2 operands, not", len(self.children))
        try:
            result = operands[0].value >= operands[1].value
        except AttributeError:
            raise ProtocolError("Greater than or equal to operator requires its operands to be simple values")
        return V.Simple(result) 
    
    def Compile(self):
        operands = [ "(" + child.Compile() + ")" for child in self.children]
        expression = ' >= '.join(operands)  
        return expression  