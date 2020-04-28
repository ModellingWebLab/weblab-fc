"""
Parse actions that can generate Python implementation objects
"""

import itertools
import math
import networkx
import os
from collections import deque
from contextlib import contextmanager

import pyparsing
import sympy
from cellmlmanip.model import DataDirectionFlow
from cellmlmanip.parser import UNIT_PREFIXES

from ..error_handling import ProtocolError, MissingVariableError
from ..language import expressions as E
from ..language import statements as S
from ..language import values as V
from ..locatable import Locatable
from ..simulations import model, modifiers, ranges, simulations
from .rdf import OXMETA_NS, PRED_IS, PRED_IS_VERSION_OF
from .rdf import create_rdf_node, get_variables_transitively, get_variables_that_are_version_of

# Magic state annotation
STATE_ANNOTATION = create_rdf_node((OXMETA_NS, 'state_variable'))

# Sympy symbol for "original_definition" construct
ORIGINAL_DEFINITION = sympy.Symbol('original_definition')

# Operators, sympy operators, etc.
OPERATORS = {'+': E.Plus, '-': E.Minus, '*': E.Times, '/': E.Divide, '^': E.Power,
             '==': E.Eq, '!=': E.Neq, '<': E.Lt, '>': E.Gt, '<=': E.Leq, '>=': E.Geq,
             'not': E.Not, '&&': E.And, '||': E.Or}
MATHML = {'log': E.Log, 'ln': E.Ln, 'exp': E.Exp, 'abs': E.Abs, 'ceiling': E.Ceiling,
          'floor': E.Floor, 'max': E.Max, 'min': E.Min, 'rem': E.Rem, 'root': E.Root,
          'power': E.Power,
          'plus': E.Plus, 'minus': E.Minus, 'times': E.Times, 'divide': E.Divide,
          'eq': E.Eq, 'neq': E.Neq, 'lt': E.Lt, 'gt': E.Gt, 'leq': E.Leq, 'geq': E.Geq,
          'not': E.Not, 'and': E.And, 'or': E.Or}
VALUES = {'true': E.Const(V.Simple(True)), 'false': E.Const(V.Simple(False)),
          'exponentiale': E.Const(V.Simple(math.e)),
          'infinity': E.Const(V.Simple(float('inf'))),
          'pi': E.Const(V.Simple(math.pi)), 'notanumber': E.Const(V.Simple(float('nan')))}

SYMPY_OPERATORS = {
    '+': sympy.Add,
    '-': lambda x, y: x - y,
    '*': sympy.Mul,
    '/': lambda x, y: x / y,
    '^': sympy.Pow,
    '==': sympy.Eq,
    '!=': sympy.Ne,
    '<': sympy.Lt,
    '>': sympy.Gt,
    '<=': sympy.Le,
    '>=': sympy.Ge,
    'not': sympy.Not,
    '&&': sympy.And,
    '||': sympy.Or,
}
SYMPY_MATHML = {
    E.Log: sympy.log,
    E.Ln: sympy.ln,
    E.Exp: sympy.exp,
    E.Abs: sympy.Abs,
    E.Ceiling: sympy.ceiling,
    E.Floor: sympy.floor,
    E.Max: sympy.Max,
    E.Min: sympy.Min,
    E.Rem: sympy.mod,
    E.Root: sympy.sqrt,
    E.Power: sympy.Pow,
    E.Plus: sympy.Add,
    E.Minus: lambda x, y: x - y,
    E.Times: sympy.Mul,
    E.Divide: lambda x, y: x / y,
    E.Eq: sympy.Eq,
    E.Neq: sympy.Ne,
    E.Lt: sympy.Lt,
    E.Gt: sympy.Gt,
    E.Leq: sympy.Le,
    E.Geq: sympy.Ge,
    E.Not: sympy.Not,
    E.And: sympy.And,
    E.Or: sympy.Or,
}


'''
    'arccos': sympy.acos,
    'arccosh': sympy.acosh,
    'arccot': sympy.acot,
    'arccoth': sympy.acoth,
    'arccsc': sympy.acsc,
    'arccsch': sympy.acsch,
    'arcsec': sympy.asec,
    'arcsech': sympy.asech,
    'arcsin': sympy.asin,
    'arcsinh': sympy.asinh,
    'arctan': sympy.atan,
    'arctanh': sympy.atanh,
    'cos': sympy.cos,
    'cosh': sympy.cosh,
    'cot': sympy.cot,
    'coth': sympy.coth,
    'csc': sympy.csc,
    'csch': sympy.csch,
    'exponentiale': sympy.E,
    'false': sympy.false,
    'infinity': sympy.oo,
    'notanumber': sympy.nan,
    'pi': sympy.pi,
    'sec': sympy.sec,
    'sech': sympy.sech,
    'sin': sympy.sin,
    'sinh': sympy.sinh,
    'tan': sympy.tan,
    'tanh': sympy.tanh,
    'true': sympy.true,
    'xor': sympy.Xor,
'''

source_file = ""  # Will be filled in by set_reference_source, e.g. in CompactSyntaxParser.try_parse


@contextmanager
def set_reference_source(reference_path):
    """Set ``reference_path`` as the base for resolving inputs in a ``with:`` block.

    Use like::

        with set_reference_source(source_file):
            parse_nested_protocol()
    """
    global source_file
    orig_source_file = source_file
    source_file = reference_path
    yield
    source_file = orig_source_file


class BaseAction(object):
    """Base class for all our parse actions.

    Parse actions are run on sections of protocols recognised by particular parser elements.

    This contains the code for allowing parsed protocol elements to be compared to lists in the test code.
    """

    def __init__(self, s, loc, tokens):
        self.tokens = tokens
        if isinstance(loc, str):
            # This instance is being created manually to implement syntactic sugar.
            self.source_location = loc
        else:
            self.source_location = "%s:%d:%d\t%s" % (
                source_file, pyparsing.lineno(loc, s), pyparsing.col(loc, s), pyparsing.line(loc, s))

    def __eq__(self, other):
        """Comparison of these parse results to another instance or a list."""
        if type(other) == type(self):
            return self.tokens == other.tokens
        elif isinstance(other, list):
            return self.tokens == other
        elif isinstance(other, str):
            return str(self.tokens) == other
        else:
            return False

    def __len__(self):
        """Get the length of the encapsulated token list."""
        if isinstance(self.tokens, str):
            length = 1
        else:
            length = len(self.tokens)
        return length

    def __getitem__(self, i):
        """Get the i'th encapsulated token."""
        assert not isinstance(self.tokens, str)
        return self.tokens[i]

    def __str__(self):
        if isinstance(self.tokens, str):
            detail = '[%s]' % self.tokens
        else:
            detail = str(self.tokens)
        return self.__class__.__name__ + detail

    def __repr__(self):
        return str(self)

    def get_children_expr(self):
        """Convert all sub-tokens to expr and return the list of elements."""
        return [tok.expr() for tok in self.tokens]

    def get_named_token_as_string(self, token_name, default=None):
        """Helper to get simple properties of this parse result."""
        value = self.tokens.get(token_name, default)
        if not isinstance(value, str) and value is not None:
            value = value[0]
        return value

    def delegate(self, action, tokens):
        """Create another parse action to process the given tokens for us."""
        if isinstance(action, str):
            action = globals()[action]
        return action('', self.source_location, tokens)

    def delegate_symbol(self, symbol, content=None):
        """
        Create a Symbol parse action, used e.g. for the ``null`` symbol in the protocol language.
        """
        if content is None:
            content = list()
        return self.delegate(Symbol(symbol), [content])

    def expr(self):
        """Updates location in parent locatable class and calls :meth:`_expr()`."""
        result = self._expr()
        if isinstance(result, Locatable):
            result.location = self.source_location
        return result

    def to_sympy(self, variable_generator, number_generator):
        """Convert this parse tree to a Sympy expression.

        May be implemented by subclasses if the operation makes sense, i.e. they represent (part of) an expression.

        :param variable_generator: Method to create expressions for symbols.
            Must have signature ``f(name) -> sympy.Basic``.
        :param number_generator: Method to create expressions for numbers with units.
            Must have signature ``f(value, unit) -> sympy.Basic`` where ``value`` is a ``float`` or ``str``.
        """
        raise NotImplementedError(str(type(self)))


class BaseGroupAction(BaseAction):
    """Base class for parse actions associated with a Group.
    This strips the extra nesting level in its __init__.
    """

    def __init__(self, s, loc, tokens):
        super(BaseGroupAction, self).__init__(s, loc, tokens[0])


class Trace(BaseGroupAction):
    """This wrapping action turns on tracing of the enclosed expression or nested protocol."""

    def _expr(self):
        wrapped_expr = self.tokens[0].expr()
        wrapped_expr.trace = True
        return wrapped_expr

######################################################################
# Post-processing language expressions
######################################################################


class Number(BaseGroupAction):
    """Parse action for numbers."""

    def __init__(self, s, loc, tokens):
        super(Number, self).__init__(s, loc, tokens)
        if len(tokens) == 2:
            # We have a units annotation (can be a name or a string ``units_of(variable)``)
            self._units = str(tokens[1])
        else:
            self._units = None

    def _expr(self):
        return E.Const(V.Simple(self.tokens))

    def to_sympy(self, variable_generator, number_generator):
        number = self.tokens
        if self._units is None:
            raise ValueError('Numbers in the model interface must have units attached; %s does not' % number)
        return number_generator(number, self._units)


class Variable(BaseGroupAction):
    """Parse action for variable references (identifiers)."""

    def _expr(self):
        var_name = self.tokens
        if var_name.startswith('MathML:'):
            actual_var = var_name[7:]
            if actual_var in MATHML:
                result = MATHML[actual_var]
            else:
                result = VALUES[actual_var]
        else:
            result = E.NameLookUp(var_name)
        return result

    def name(self):
        """Returns the name of the variable being referenced."""
        return str(self.tokens)

    def names(self):
        return [str(self.tokens)]

    def to_sympy(self, variable_generator, number_generator):
        return variable_generator(self.tokens)


class Operator(BaseGroupAction):
    """Parse action for most MathML operators that are represented as operators in the syntax."""

    def __init__(self, *args, **kwargs):
        super(Operator, self).__init__(*args)
        self.rightAssoc = kwargs.get('rightAssoc', False)

    def operator_operands(self):
        """Generator over (operator, operand) pairs."""
        it = iter(self.tokens[1:])
        while 1:
            try:
                operator = next(it)
                operand = next(it)
                yield (operator, operand)
            except StopIteration:
                break

    def _expr(self):
        if self.rightAssoc:
            # The only right-associative operators are also unary
            result = self.tokens[-1].expr()
            for operator in self.tokens[-2:-1:]:
                result = OPERATORS[operator](result)
        else:
            result = self.tokens[0].expr()
            for operator, operand in self.operator_operands():
                result = OPERATORS[operator](result, operand.expr())
        return result

    def to_sympy(self, variable_generator, number_generator):
        if self.rightAssoc:
            # The only right-associative operators are also unary
            result = self.tokens[-1].to_sympy(variable_generator, number_generator)
            for operator in self.tokens[-2:-1:]:
                if operator == '+':
                    pass
                elif operator == '-':
                    result = -result
                else:
                    raise NotImplementedError
        else:
            result = self.tokens[0].to_sympy(variable_generator, number_generator)
            for operator, operand in self.operator_operands():
                result = SYMPY_OPERATORS[operator](result, operand.to_sympy(variable_generator, number_generator))
        return result


class Wrap(BaseGroupAction):
    """Parse action for wrapped MathML operators."""

    def _expr(self):
        assert len(self.tokens) == 2
        operator_name = self.tokens[1]
        if operator_name.startswith('MathML:'):
            operator = MATHML[operator_name[7:]]
        else:
            operator = OPERATORS[operator_name]
        num_operands = int(self.tokens[0])
        return E.LambdaExpression.wrap(operator, num_operands)


class Piecewise(BaseGroupAction):
    """Parse action for if-then-else."""

    def _expr(self):
        if_, then_, else_ = self.get_children_expr()
        return E.If(if_, then_, else_)

    def to_sympy(self, variable_generator, number_generator):
        assert len(self.tokens) == 3
        if_, then_, else_ = [t.to_sympy(variable_generator, number_generator) for t in self.tokens]
        return sympy.Piecewise((then_, if_), (else_, True))


class MaybeTuple(BaseGroupAction):
    """Parse action for elements that may be grouped into a tuple, or might be a single item."""

    def _expr(self):
        assert len(self.tokens) > 0
        if len(self.tokens) > 1:
            # Tuple
            return self.delegate('Tuple', [self.tokens]).expr()
        else:
            # Single item
            return self.tokens[0].expr()  # should be list containing names

    def names(self):
        return list(map(str, self.tokens))


class Tuple(BaseGroupAction):
    """Parse action for tuples."""

    def _expr(self):
        child_expr = self.get_children_expr()
        return E.TupleExpression(*child_expr)


class Lambda(BaseGroupAction):
    """Parse action for lambda expressions."""

    def _expr(self):
        assert len(self.tokens) == 2
        param_list = self.tokens[0]
        body = self.tokens[1].expr()  # expr
        children = []
        default_params = []
        for param_decl in param_list:
            param_bvar = param_decl[0].names()  # names method
            if len(param_decl) == 1:  # No default given
                children.append(param_bvar)
                default_params.append(None)
            else:  # Default value case
                default_params.append(param_decl[1].expr().value)
                children.append(param_bvar)
        lambda_params = [[var for each in children for var in each]]
        if not isinstance(body, list):
            ret = S.Return(body)
            ret.location = body.location
            body = [ret]
        lambda_params.extend([body, default_params])
        return E.LambdaExpression(*lambda_params)


class FunctionCall(BaseGroupAction):
    """Parse action for function calls."""

    def _expr(self):
        assert len(self.tokens) == 2
        assert isinstance(self.tokens[0], Variable)
        func = self.tokens[0].expr()
        args = [t.expr() for t in self.tokens[1]]
        if hasattr(func, 'name'):
            if func.name == 'map':
                result = E.Map(*args)
            elif func.name == 'fold':
                result = E.Fold(*args)
            elif func.name == 'find':
                result = E.Find(*args)
            else:
                result = E.FunctionCall(func, args)
        elif not isinstance(func, E.NameLookUp):
            result = func(*args)
        else:
            result = E.FunctionCall(func, args)
        return result

    def to_sympy(self, variable_generator, number_generator):
        # Only supported for simple expressions (as used in the model interface)

        assert len(self.tokens) == 2
        assert isinstance(self.tokens[0], Variable)

        # Get callable that creates sympy object
        func = self.tokens[0].expr()
        try:
            func = SYMPY_MATHML[func]
        except KeyError:
            raise NotImplementedError('Sympy conversion not supported for: ' + str(func))

        # Parse operands, create and return
        return func(*[token.to_sympy(variable_generator, number_generator) for token in self.tokens[1]])


class _Symbol(BaseGroupAction):
    """Parse action for csymbols."""

    def __init__(self, s, loc, tokens, symbol):
        super(_Symbol, self).__init__(s, loc, tokens)
        self.symbol = symbol

    def _expr(self):
        if self.symbol == "null":
            return E.Const(V.Null())
        elif self.symbol == "defaultParameter":
            return E.Const(V.DefaultParameter())
        if isinstance(self.tokens, str):
            return E.Const(V.String(self.tokens))


def Symbol(symbol):
    """Wrapper around the _Symbol class."""
    def parse_action(s, loc, tokens):
        return _Symbol(s, loc, tokens, symbol)
    return parse_action


class Accessor(BaseGroupAction):
    """Parse action for accessors."""

    def _expr(self):
        if len(self.tokens) > 2:
            # Chained accessors, e.g. E.SHAPE.IS_ARRAY
            return self.delegate(
                'Accessor', [[self.delegate('Accessor', [self.tokens[:-1]]), self.tokens[-1]]]).expr()
        assert len(self.tokens) == 2
        object = self.tokens[0].expr()
        property = getattr(E.Accessor, self.tokens[1])
        return E.Accessor(object, property)


class Comprehension(BaseGroupAction):
    """Parse action for the comprehensions with array definitions."""

    def _expr(self):
        assert 2 <= len(self.tokens) <= 3
        parts = []
        if len(self.tokens) == 3:
            # There's an explicit dimension
            parts.append(self.tokens[0])
        range = self.tokens[-1]
        if len(range) == 2:
            # Add a step of 1
            range = [range[0], self.delegate('Number', ['1']), range[-1]]
        parts.extend(range)
        parts.append(self.delegate_symbol('string', self.tokens[-2]))  # The variable name
        return self.delegate('Tuple', [parts]).expr()


class Array(BaseGroupAction):
    """Parse action for creating arrays."""

    def _expr(self):
        entries = self.get_children_expr()
        if len(entries) > 1 and isinstance(self.tokens[1], Comprehension):
            # Array comprehension
            return E.NewArray(*entries, comprehension=True)
        else:
            return E.NewArray(*entries)


class View(BaseGroupAction):
    """Parse action for array views."""

    def _expr(self):
        assert 2 <= len(self.tokens)
        args = [self.tokens[0].expr()]
        null_token = self.delegate_symbol('null')
        for viewspec in self.tokens[1:]:
            tuple_tokens = []
            dimspec = None
            if 'dimspec' in viewspec:
                dimspec = viewspec['dimspec'][0]
                viewspec = viewspec[1:]
            tuple_tokens.extend(viewspec)
            if dimspec is not None:
                if len(tuple_tokens) == 1:
                    real_tuple_tokens = [dimspec, tuple_tokens[0], self.delegate('Number', ['0']), tuple_tokens[0]]
                elif len(tuple_tokens) == 2:
                    real_tuple_tokens = [dimspec, tuple_tokens[0], self.delegate('Number', ['1']), tuple_tokens[1]]
                else:
                    real_tuple_tokens = [dimspec, tuple_tokens[0], tuple_tokens[1], tuple_tokens[2]]
            else:
                real_tuple_tokens = tuple_tokens
            # Replace unspecified elements with csymbol-null
            for i, token in enumerate(real_tuple_tokens):
                if token == '' or token == '*':
                    real_tuple_tokens[i] = null_token
            args.append(self.delegate('Tuple', [real_tuple_tokens]).expr())
        return E.View(*args)


class Index(BaseGroupAction):
    """Parse action for index expressions."""

    def _expr(self):
        assert len(self.tokens) == 2
        index_tokens = self.tokens[1]
        assert 1 <= len(index_tokens)
        args = [self.tokens[0], index_tokens[0]]
        args.append(index_tokens.get('dim', self.delegate_symbol('defaultParameter')))
        args.append(index_tokens.get('shrink', [self.delegate_symbol('defaultParameter')])[0])
        if 'pad' in index_tokens:
            assert len(index_tokens['pad']) == 2
            args.extend(index_tokens['pad'])  # Pad direction & value
        args = [each.expr() for each in args]
        return E.Index(*args)

######################################################################
# Post-processing language statements
######################################################################


class Assignment(BaseGroupAction):
    """Parse action for both simple and tuple assignments."""

    def __init__(self, s, loc, tokens):
        super(Assignment, self).__init__(s, loc, tokens)
        if len(self.tokens) == 3:
            # This is an optional assignment
            self._optional = True
            self.tokens = self.tokens[1:]
        else:
            self._optional = False

    def _expr(self):
        assignee, value = self.get_children_expr()
        if isinstance(assignee, E.NameLookUp):
            var_list = [assignee.name]
        elif isinstance(assignee, E.TupleExpression):
            var_list = [child.name for child in assignee.children]
        args = [var_list, value]
        if self._optional:
            args.append(True)
        return S.Assign(*args)


class Return(BaseGroupAction):
    """Parse action for return statements."""

    def _expr(self):
        return S.Return(*self.get_children_expr())


class Assert(BaseGroupAction):
    """Parse action for assert statements."""

    def _expr(self):
        return S.Assert(*self.get_children_expr())


class FunctionDef(BaseGroupAction):
    """Parse action for function definitions, which are sugar for assignment of a lambda."""

    def _expr(self):
        assert len(self.tokens) == 3
        lambda_ = self.delegate('Lambda', [self.tokens[1:]])
        assign = self.delegate('Assignment', [[self.tokens[0], lambda_]])
        return assign.expr()


class StatementList(BaseGroupAction):
    """Parse action for lists of post-processing language statements."""

    def _expr(self):
        statements = self.get_children_expr()
        return statements

######################################################################
# Model interface section
######################################################################


class SetTimeUnits(BaseAction):
    """Parse action for specifying the units time should have in the model.

    ``independent var units <uname>``
    """
    def _expr(self):
        self.time_units = self.get_named_token_as_string('units')
        assert self.time_units is not None
        return self


class VariableReference:
    """
    Mixin providing properties for resolving references to variables specified via RDF terms (references to local
    variables do not use the VariableReference class).

    Used by input & output variable specifications, amongst others.

    Properties:

    ``prefixed_name``
        A 'prefix:local_name' string.
    ``ns_prefix``
        The namespace prefix part of ``prefixed_name``.
    ``local_name``
        The local name part of ``prefixed_name``.
    ``ns_uri``
        Once namespace prefixes have been resolved, the namespace URI corresponding to ``ns_prefix``.
    ``rdf_term``
        Once namespace prefixes have been resolved, the RDF term that annotates variable(s) we reference.

    """
    def _expr(self):
        self.set_name(self.get_named_token_as_string('name'))
        return self

    def set_name(self, name):
        """Set the name used for this reference. For use by subclasses."""
        self.prefixed_name = name
        self.ns_prefix, self.local_name = name.split(':', 1)
        self.ns_uri = None  # Will be set later using protocol's namespace mapping
        self.rdf_term = None  # Ditto

    def set_namespace(self, ns_uri):
        """Set the full namespace URI for this reference, and hence the RDF term."""
        self.ns_uri = ns_uri
        self.rdf_term = create_rdf_node((self.ns_uri, self.local_name))


class InputVariable(BaseGroupAction, VariableReference):
    """
    Parse action for input variables defined in the model interface.

    ``input <prefix:term> [units <uname>] [= <initial_value>]``

    Provides additional properties:

    ``units``
        Name of the units this variable is wanted in. Optional; default None.
    ``initial_value``
        Optional initial value for the variable; default None.
    """
    def _expr(self):
        super()._expr()
        self.units = self.get_named_token_as_string('units', default=None)
        self.initial_value = self.get_named_token_as_string('initial_value', default=None)
        return self


class OutputVariable(BaseGroupAction, VariableReference):
    """
    Parse action for output variables defined in the model interface.

    ``output <prefix:term> [units <uname>]``

    Provides additional properties:

    ``units``
        Name of the units this variable is wanted in. Optional; default None.
    """
    def _expr(self):
        super()._expr()
        self.units = self.get_named_token_as_string('units', default=None)
        return self


class OptionalVariable(BaseGroupAction, VariableReference):
    """Parse action for specifying optional variables in the model interface.

    ``optional <prefix:term> [default <simple_expr>]``

    Provides additional properties:

    ``default_expr``
        Optional string giving the default expression for the variable.
    """
    def __init__(self, s, loc, tokens):
        super().__init__(s, loc, tokens)

        self.default_expr = self.tokens.get('default', None)


class DeclareVariable(BaseGroupAction):
    """
    Parse action for local ``var`` declarations in the model interface.

    ``var <name> units <uname> [= <initial_value>]``

    These variables have no rdf term (but just a simple name).
    They can get their value from the initial value specified on the same line, or through a ``define`` statement.
    """

    def _expr(self):
        self.name = self.get_named_token_as_string('name')
        self.units = self.get_named_token_as_string('units')
        self.initial_value = self.get_named_token_as_string('initial_value', default=None)
        return self


class ClampVariable(BaseGroupAction, VariableReference):
    """Parse action for ``clamp`` declarations in the model interface.

    ``clamp <prefix:term> [to <simple_expr>]``

    With a 'to' clause this is implemented as syntactic sugar for :class:`ModelEquation`.

    Clamping to the initial value in the model is handled specially; we just record the variable reference.
    """
    def _expr(self):
        assert 1 <= len(self.tokens) <= 2
        name = self.tokens[0]
        if len(self.tokens) == 1:
            # Clamp to initial value, whatever that happens to evaluate as
            self.set_name(name.tokens)
            return self
        else:
            # Clamp to simple expression: delegated to ModelEquation using lhs = name, rhs = simple expression
            value = self.tokens[1]
            return self.delegate('ModelEquation', [[name, value]]).expr()


class ModelEquation(BaseGroupAction):
    """
    Parse action for ``define`` declarations in the model interface, that add or replace model variable's equations.

    ``define diff(<prefix:term>;<prefix:term>) | <prefix:term> = <simple_expr>``

    Properties:

    ``var``
        A :class:`Variable` indicating the variable this statement defines or modifies.
        Can be a simple name local to the model interface (see :class:`DefineVariable`), or a prefixed name.
    ``rhs``
        The new RHS, as a parse action tree initially. Call :meth:`to_sympy` to convert it.
    ``is_ode``
        True if the variable should be a state.
    ``bvar``
        ``None`` if this variable isn't a state, but a :class:`Variable` if it is, giving the 'time' variable.
    """
    def _expr(self):
        # Parse LHS
        if isinstance(self.tokens[0], Variable):
            # Assigning a normal variable
            assert len(self.tokens[0]) == 1
            self.var = self.tokens[0]
            self.bvar = None
        else:
            # Assigning an ODE
            assert len(self.tokens[0]) == 2
            self.var = self.tokens[0][0]
            self.bvar = self.tokens[0][1]
        self.is_ode = self.bvar is not None

        # Store RHS just as the tokens for now; converting to Sympy happens later
        self.rhs = self.tokens[1]

        return self

    def to_sympy(self, variable_generator, number_generator):
        """Convert this equation to Sympy.

        Typically this will be called via :meth:`ModelInterface.convert_equations_to_sympy` which sets up appropriate
        generators.

        :param variable_generator: a function to resolve a name reference within the equation to a model variable.
        :param number_generator: converts a number with units to a Sympy entity.
        """
        lhs = self.lhs_to_sympy(variable_generator, number_generator)
        rhs = self.rhs.to_sympy(variable_generator, number_generator)
        return sympy.Eq(lhs, rhs)

    def lhs_to_sympy(self, variable_generator, number_generator):
        """Converts only this equation's LHS to sympy."""

        lhs = self.var.to_sympy(variable_generator, number_generator)
        if self.is_ode:
            bvar = self.bvar.to_sympy(variable_generator, number_generator)
            lhs = sympy.Derivative(lhs, bvar, evaluate=False)
        return lhs


class Interpolate(BaseGroupAction):
    # Leaving old XML method in to document existing properties / tokens.
    # def _xml(self):
    #    assert len(self.tokens) == 4
    #    assert isinstance(self.tokens[0], str)
    #    file_path = self.delegate_symbol('string', self.tokens[0]).xml()
    #    assert isinstance(self.tokens[1], Variable)
    #    indep_var = self.tokens[1].xml()
    #    units = []
    #    for i in [2, 3]:
    #        assert isinstance(self.tokens[i], str)
    #        units.append(M.ci(self.tokens[i]))
    #    return M.apply(self.delegate_symbol('interpolate').xml(), file_path, indep_var, *units)
    pass


class UnitsConversion(BaseGroupAction):
    """
    Parse action for a units conversion rule.

    ``convert <uname> to <uname> by <simple_lambda_expr>``
    """

    def _expr(self):
        self.actual = self.get_named_token_as_string('actualDimensions')
        self.desired = self.get_named_token_as_string('desiredDimensions')
        self.rule = self.tokens[-1]


class ProtocolVariable():
    """
    Collates information about a (new or existing) model variable used in a protocol.

    Properties:

    ``name``
        A prefixed ontology term or a local variable name.
    ``short_name``
        A short name (no prefixes) that can be used for this variable.
    ``long_name``
        This variable's name plus any known aliases.
    ``input_terms``
        A list of rdf terms by which this variable is known as an input. This is typically zero or one term, but could
        end up being more if two ontology terms specify the same model variables. For inputs this is an error.
    ``output_terms``
        A list of rdf terms by which this variable is known as a scalar output. This is typically zero or one term, but
        multiple ontology terms may annotate the same output (as long as their units are consistent).
    ``vector_output_terms``
        A list of rdf terms by which this variable is known as a vector output. This is typically zero or one term.
    ``is_input``
        ``True`` iff the associated model variable is set as a model input by the protocol
    ``is_output``
        ``True`` iff the associated model variable is read as a model output by the protocol.
    ``is_optional``
        ``True`` iff this variable is allowed *not* to exist in the (original or modified) model.
    ``is_vector``
        ``True`` iff this variable is a vector output.
    ``is_local``
        ``True`` iff this is a local variable (i.e. defined within the protocol).
    ``units``
        The string name of the units specified for this variable (or ``None``).
    ``initial_value``
        An initial value (float) for this variable or ``None``, if given as part of an ``input`` clause.
    ``default_expr``
        An expression (``fc.language.AbstractExpression``) for this variable's default value. Set if given as part of an
        ``optional`` clause, else ``None``.
    ``equation``
        An equation for this variable, set using a ``define`` or a ``clamp`` clause.
    ``model_variable``
        A :class:`VariableDummy` instance that this protocol variable refers to (note that this may change during the
        lifetime of a :class:`ProtocolVariable`, e.g. through unit conversion.
    ``vector_variables``
        A set of :class:`VariableDummy` objects that this protocol variables refers to indirectly, e.g. if it was
        derived from an ontology term representing a category of variables.

    """
    def __init__(self, name):
        self.name = name
        self.short_name = name if ':' not in name else name[1 + name.index(':'):]
        self.long_name = name
        self._aliases = []
        self.input_terms = []
        self.output_terms = []
        self.is_input = False
        self.is_output = False
        self.is_optional = False
        self.is_vector = False
        self.is_local = False
        self.units = None
        self.initial_value = None
        self.default_expr = None
        self.equation = None
        self.model_variable = None
        self.vector_variables = []

    def update(self, name=None, input_term=None, output_term=None, is_optional=False, is_vector=False, is_local=False,
               units=None, initial_value=None, default_expr=None, equation=None, model_variable=None,
               vector_variables=None):
        """
        Merges new information into this :class:`ProtocolVariable`, raises a ProtocolError if new information is
        incompatible with what is already stored.
        """
        # Add name
        if name is not None and name != self.name and name not in self._aliases:
            self._aliases.append(name)
            self.long_name = 'name (aka ' + ', '.join(self._aliases) + ')'

        # Add rdf terms
        if input_term is not None and input_term not in self.input_terms:
            self.input_terms.append(input_term)
        if output_term is not None and output_term not in self.output_terms:
            self.output_terms.append(output_term)

        # Update type information
        self.is_input = bool(self.input_terms)
        self.is_output = bool(self.output_terms)
        self.is_optional = self.is_optional or is_optional
        self.is_vector = self.is_vector or is_vector
        self.is_local = self.is_local or is_local

        # Merge units
        if self.units is None:
            self.units = units
        elif units is not None and units != self.units:
            raise ProtocolError(f'Inconsistent units specified for {self.long_name} ({self.units} and {units}).')

        # Merge initial value
        if self.initial_value is None:
            self.initial_value = initial_value
        elif initial_value is not None and initial_value != self.initial_value:
            raise ProtocolError(f'Multiple initial values specified for {self.long_name} ({self.initial_value} and'
                                ' {initial_value}).')

        # Merge default expression
        if self.default_expr is None:
            self.default_expr = default_expr
        elif default_expr is not None and default_expr != self.default_expr:
            raise ProtocolError(f'Multiple default expressions specified for {self.long_name} ({self.default_expr} and'
                                ' {default_expr}).')

        # Merge equation
        if self.equation is None:
            self.equation = equation
        elif equation is not None and equation != self.equation:
            raise ProtocolError(f'Multiple equations specified for {self.long_name}.')

        # Set model variable
        if model_variable is not None:
            if self.model_variable is None:
                self.model_variable = model_variable
            else:
                assert model_variable == self.model_variable, 'ProtocolVariable representing multiple model variables'

        # Add vector variables
        if vector_variables:
            # At the moment this isn't used: vectors are only set post-modifications
            self.vector_variables.extend(vector_variables)

        # Check type information isn't conflicting (shouldn't be possible unless code/logic is wrong)
        assert not (self.is_local and (self.is_input or self.is_output)), f'{self.long_name} is local AND input/output'
        if self.is_vector:
            assert self.model_variable is None, f'{self.long_name} is a vector, but has a scalar variable'
            assert not self.is_input, f'{self.long_name} is an input AND a vector output'
            assert len(self.output_terms) == 1, f'{self.long_name} is a vector with multiple output terms'
        else:
            assert not self.vector_variables, f'{self.long_name} is not a vector, but has vector variables'

    def merge(self, pvar):
        """
        Merges the information from another protocol variable (``pvar``) into this one, assuming they both represent the
        same model variable: raises a ProtocolError if any conflicts are found.
        """
        if self.model_variable is None or self.model_variable is not pvar.model_variable:
            raise ValueError(
                'Merge() should only be used to merge protocol variables referencing the same model variable.')

        # Update list properties
        for term in pvar.input_terms:
            if term not in self.input_terms:
                self.input_terms.append(term)
        for term in pvar.output_terms:
            if term not in self.output_terms:
                self.output_terms.append(term)

        # Update optional indicator: If the same model variable is known by multiple rdf terms, they must _all_ be
        # optional for the variable to be optional.
        self.is_optional = self.is_optional and pvar.is_optional

        # Update the rest
        self.update(
            name=pvar.name,
            is_local=pvar.local,
            units=pvar.units,
            initial_value=pvar.initial_value,
            default_expr=pvar.default_expr,
            equation=pvar.equation,
            model_variable=pvar.model_variable,
            vector_variables=pvar.vector_variables,
        )


class ModelInterface(BaseGroupAction):
    """Parse action for the model interface section of a protocol.

    See https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Modelinterface for more on the syntax
    and semantics of the model interface.

    Includes helper methods for merging model interfaces, e.g. when one protocol imports another.

    Properties set after :meth:`_expr` has been called (and finalised after merging):

    ``time_units``
        Either None, or the name of the units to use for time.
    ``inputs``
        A list of :class:`InputVariable` objects specifying model variables that can be changed by the protocol.
    ``outputs``
        A list of :class:`OutputVariable` objects specifying model variables that can be read by the protocol.
    ``optional_decls``
        A list of :class:`OptionalVariable` objects specifying model variables referenced by the protocol
        that can be missing without causing an immediate error. This applies to variables declared as inputs or
        outputs, or referenced in new/changed equations. Errors may still arise later if a missing optional
        variable still ends up included in the generated model code, e.g. because no default clause or alternative
        was found.
    ``equations``
        A list of :class:`ModelEquation` objects specifying equations to be added to the model, possibly replacing
        existing equations defining the same variable(s).
        New definitions set with ``define`` end up here, as do statements such as ``clamp x to 1``.
    ``clamps``
        A list of :class:`ClampVariable` instances. These are created for e.g. ``clamp x``, while statements with an RHS
        such as ``clamp x to 1`` are treated as an alias of ``define x = 1`` and do not lead to creation of a
        ``ClampVariable`` object.
    ``local_var_declarations``
        A list of :class:`DeclareVariable` instances. These are created for ``var x`` statements.

    Properties set after :meth:`resolve_namespaces` has been called:

    ``ns_map``
        A dict mapping prefixes to URIs, as defined by the protocol.

    Properties set after :meth:`modify_model` has been called:

    ``units``
        A unit store.
    ``model``
        A model.
    ``time_variable``
        The model variable (``VariableDummy``) representing time.
    ``protocol_variables``
        A list of :class:`ProtocolVariable` objects.
    ``local_vars``
        A dict mapping names of local variables (created with ``var`` / ``DeclareVariable``) to model variables.
    ``magic_pvar``
        A :class:`ProtocolVariable` for the magic annotation ``oxmeta:state_variable``, or ``None``.

    """
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            # This is an empty instance not created by pyparsing. Fake the arguments pyparsing needs.
            args = ('', '', [[]])
        super().__init__(*args, **kwargs)
        self._time_units = []
        self.inputs = []
        self.outputs = []
        self.optional_decls = []
        self.equations = []
        self.clamps = []
        self.local_var_declarations = []

        # Initialise
        self._reinit()

    def _reinit(self):
        """(Re-)initialise this interface, so that it can be re-used."""

        self.ns_map = None
        self.units = None
        self.model = None
        self.time_variable = None
        self.protocol_variables = []
        self.local_vars = {}
        self.magic_pvar = None

    def _expr(self):
        actions = self.get_children_expr()
        self._time_units = [a for a in actions if isinstance(a, SetTimeUnits)]
        self.inputs = [a for a in actions if isinstance(a, InputVariable)]
        self.outputs = [a for a in actions if isinstance(a, OutputVariable)]
        self.optional_decls = [a for a in actions if isinstance(a, OptionalVariable)]
        self.equations = [a for a in actions if isinstance(a, ModelEquation)]
        self.clamps = [a for a in actions if isinstance(a, ClampVariable)]
        self.local_var_declarations = [a for a in actions if isinstance(a, DeclareVariable)]

        # Some basic semantics checking
        if len(self._time_units) > 1:
            raise ValueError('The units for time cannot be set multiple times')

        # (Re-)initialise this interface (so that cached interface objects can be re-used)
        self._reinit()

        return self

    def merge(self, interface):

        # only append unique entries
        def add_unique(list1, list2):
            for l in list2:
                if l not in list1:
                    list1.append(l)

        # append lists from interface to those already in this interface
        add_unique(self.inputs, interface.inputs)
        add_unique(self.outputs, interface.outputs)
        add_unique(self.optional_decls, interface.optional_decls)
        add_unique(self.equations, interface.equations)
        add_unique(self.clamps, interface.clamps)
        add_unique(self.local_var_declarations, interface.local_var_declarations)

        # need to be careful with time units
        # add from nested protocol if there are no time units in outer protocol
        # if outer and inner have time units these should be the same
        if not self._time_units:
            # only add if interface actually has an entry in _time_units
            if interface._time_units:
                self._time_units.append(interface._time_units[0])
        elif interface._time_units:
            if self.units.get_unit(self.time_units) != interface.units.get_unit(interface.time_units):
                raise ValueError('Mismatch in the units for time in nested protocols')

    @property
    def time_units(self):
        if self._time_units:
            return self._time_units[0].time_units
        return None

    def resolve_namespaces(self, ns_map):
        """Resolve namespace prefixes to full URIs for all parts of the interface.

        :param ns_map: mapping from NS prefix to URI.
        """
        self.ns_map = ns_map
        for item in itertools.chain(self.inputs, self.outputs, self.optional_decls, self.clamps):
            item.set_namespace(ns_map[item.ns_prefix])

    #######################################
    # Model manipulation and helper methods

    def modify_model(self, model, units):
        """Use the definitions in this interface to transform the provided model.

        This calls various internal helper methods to do the modifications, in an order orchestrated to follow the
        principle of least surprise for protocol authors. It attempts to produce results that most probably match what
        they expect to happen, without creating an inconsistent model.

        Key steps are:
        - Adding new variables and updating equations.
        - Clamping variables to their initial value.
        - Annotating the variables now comprising the state variable vector so they are recognised by
          the oxmeta:state_variable 'magic' ontology term.
        - Model variables and equations not needed to compute the requested outputs are removed.

        :param cellmlmanip.model.Model model: the model to manipulate
        :param cellmlmanip.units.UnitStore units: the protocol's unit store, for resolving unit references
        """
        self.model = model
        self.units = units

        # Models in FC must always have a time variable
        try:
            self.time_variable = self.model.get_free_variable()
        except ValueError:
            # TODO: Look for variable annotated as time instead?
            raise ProtocolError('Model must contain at least one ODE.')

        # Annotate all state variables with the magic `oxmeta:state_variable` term. This is done before unit conversion
        # so that annotations are transferred where needed. The original order in which state variables were defined is
        # stored.
        original_state_order = self._annotate_state_variables()

        # Collate information about variables mentioned in protocol.
        self._collate_variable_information()

        # Convert units of model variables, where needed.
        self._convert_time_unit_if_needed()
        self._convert_units()

        # Processes the variables used by the protocol, creating variables if needed and updating model equations.
        self._process_protocol_variables()

        # Process ``clamp`` statements with an RHS
        self._handle_clamping()

        # Process vector outputs
        self._process_transitive_variables()

        # Gather state variables, store vector ordering and add to ProtocolVariable if requested
        self._gather_state_variables(original_state_order)

        self._purge_unused_mathematics()

        # TODO: Any final consistency checks on the model?

    def _variable_generator(self, name):
        """Resolve a name reference within a model interface equation to a variable in the model.

        Used by :meth:`convert_equations_to_sympy`.

        :param str name: a name reference. If it contains a ':' then it is treated as a prefix:local_name ontology term
            reference. The prefix is looked up in our namespace URI map, and the variable then found with
            :meth:`cellmlmanip.model.Model.get_variable_by_ontology_term`. Otherwise we look for a variable defined
            within the protocol's model interface using :class:`DeclareVariable`.
        """
        # Reference to original definition: return a simple Sympy symbol to be substituted at a later stage.
        if name == 'original_definition':
            return ORIGINAL_DEFINITION

        # Annotation
        if ':' in name:
            prefix, local_name = name.split(':', 1)
            ns_uri = self.ns_map[prefix]
            try:
                return self.model.get_variable_by_ontology_term((ns_uri, local_name))
            except KeyError:
                raise MissingVariableError(name)

        # Local variable
        try:
            return self.local_vars[name]
        except KeyError:
            raise MissingVariableError(name)

    def _number_generator(self, value, units):
        """Convert a number with units in an equation to a :class:`cellmlmanip.model.NumberDummy`.

        Used by :meth:`convert_equations_to_sympy`.

        May raise a :class:`MissingVariableError` if the number's units are set with ``units_of(x)`` where ``x`` is a
        variable not (yet) known to the model.

        :param value: the numerical value
        :param units: the *name* of the units for this quantity. Will be looked up from the protocol's definitions.
            Alternatively, this can be a string ``units_of(variable)`` where ``variable`` is an ontology term for a
            variable in this model.
        """
        if units.startswith('units_of('):
            name = units[9:-1]
            # Get variable. This may trigger a MissingVariableError.
            var = self._variable_generator(name)
            units = var.units
        else:
            units = self.units.get_unit(units)
        return self.model.add_number(value, units)

    def _annotate_state_variables(self):
        """
        Annotate all state variables with the 'magic' oxmeta:state_variable term.

        This method also ensures that all state variables have an rdf identity (``var.rdf_identity``).
        :returns: A list containing all current state variables, in order of appearance in the model.
        """
        order = []
        for var in self.model.get_state_variables():
            self.model.add_cmeta_id(var)
            self.model.rdf.add((var.rdf_identity, PRED_IS_VERSION_OF, STATE_ANNOTATION))
            order.append(var.rdf_identity)
        return order

    def _collate_variable_information(self):
        """Collates information from different variable defining clauses, and stores it in ProtocolVariable objects."""

        # Temporary map from (protocol) names to ProtocolVariable objects
        name_to_pvar = {}

        def get(ref):
            try:
                pvar = name_to_pvar[ref.prefixed_name]
            except KeyError:
                pvar = ProtocolVariable(ref.prefixed_name)
                name_to_pvar[ref.prefixed_name] = pvar

                # Store 'state_variable' pvar, if used in protocol
                if ref.rdf_term == STATE_ANNOTATION:
                    self.magic_pvar = pvar

            return pvar

        # Add inputs
        #   input <prefix:term> [units <uname>] [= <initial_value>]
        for ref in self.inputs:
            pvar = get(ref)
            pvar.update(input_term=ref.rdf_term, units=ref.units, initial_value=ref.initial_value)

        # Add outputs and output categories
        #   output <prefix:term> [units <uname>]
        for ref in self.outputs:
            pvar = get(ref)
            pvar.update(output_term=ref.rdf_term, units=ref.units)

        # Add optional declarations
        #   optional <prefix:term> [default <simple_expr>]
        for ref in self.optional_decls:
            pvar = get(ref)
            pvar.update(is_optional=True, default_expr=ref.default_expr)

        # Add local variables
        #   var <name> units <uname> [= <initial_value>]
        for ref in self.local_var_declarations:
            # Local variable names must be unique, and can't even be re-used in imported/nested protocols
            if ref.name in name_to_pvar:
                raise ProtocolError(f'Variable "{ref.name}" was defined by more than one var statement.')

            # Create and store variable
            pvar = name_to_pvar[ref.name] = ProtocolVariable(ref.name)
            pvar.update(is_local=True, units=ref.units, initial_value=ref.initial_value)

        # Store equations from define and clamp statements
        for eq in self.equations:
            name = eq.var.name()
            try:
                pvar = name_to_pvar[name]
            except KeyError:
                # Variable not found: Still OK, as long as it refers to an existing model variable
                try:
                    self._variable_generator(name)
                except MissingVariableError:
                    raise ProtocolError(f'Define or clamp statement found for unknown variable: {name}.')

                # Create new protocol variable to store info
                pvar = name_to_pvar[name] = ProtocolVariable(name)

            # Store equation
            pvar.update(equation=eq)

        # Resolve references to model variables
        # If multiple references point to the same model variable, merge them
        var_to_pvar = {}
        aliases = []
        for pvar in name_to_pvar.values():
            try:
                pvar.model_variable = self._variable_generator(pvar.name)
            except MissingVariableError:
                # Could be optional, or vector output (which we can only determine post model modifications)
                continue

            # Check if another reference already points to this model variable
            try:
                partner = var_to_pvar[pvar.model_variable]
            except KeyError:
                var_to_pvar[pvar.model_variable] = pvar
                continue

            # Merge, and mark pvar for removal
            partner.merge(pvar)
            aliases.add(pvar)

        # Remove 'alias' references
        for pvar in aliases:
            del name_to_pvar[pvar.name]
        del aliases

        # Raise error if a variable is specified as an input using two different ontology terms
        for pvar in name_to_pvar.values():
            if len(pvar.input_terms) > 1:
                raise ProtocolVariable(
                    f'The model variable {pvar.model_variable} is specified as a protocol input by more than one'
                    f'ontology term {pvar.long_name}.')

        # Check against overdefinedness through clamp-to-initial-value plus an equation
        for ref in self.clamps:
            try:
                pvar = var_to_pvar[self.model.get_variable_by_ontology_term(ref.rdf_term)]
            except KeyError:
                continue
            if pvar.equation:
                raise ProtocolError(
                    f'The variable {pvar.long_name} is set by more than one clamp and/or define statement.')

        # Store all protocol variables
        self.protocol_variables = list(name_to_pvar.values())

    def _convert_time_unit_if_needed(self):
        """Check the units of the time variable and convert if needed."""

        # Try getting units from independent var clause
        units = self.time_units

        # Check if the units are set by an input or output instead or in addition to the independent var clause
        for pvar in self.protocol_variables:
            if pvar.model_variable is self.time_variable:
                if units is None:
                    units = pvar.units
                elif pvar.units is not None and pvar.units != units:
                    raise ProtocolError(f'Inconsistent units specified for the time variable {pvar.long_name}.')

        # Convert if required and possible
        if units is not None:
            units = self.units.get_unit(units)
            self.time_variable = self.model.convert_variable(self.time_variable, units, DataDirectionFlow.INPUT)

    def _convert_units(self):
        """Checks the units of model variables against protocol variables and converts them if needed."""
        # Note: this code assumes that the time variable has already been converted, so no provision is made to ensure
        # that self.time_variable still points at the correct variable, or that time is converted as an input.

        # Check and convert inputs
        for pvar in self.protocol_variables:
            # Get desired units
            if pvar.units is None:
                continue
            units = self.units.get_unit(pvar.units)

            # Convert model_variable units
            if (pvar.is_input or pvar.is_output) and pvar.model_variable is not None:
                if pvar.model_variable.units != units:
                    direction = DataDirectionFlow.INPUT if pvar.is_input else DataDirectionFlow.OUTPUT
                    pvar.model_variable = self.model.convert_variable(pvar.model_variable, units, direction)

    def _process_protocol_variables(self):
        """
        Processes the list of protocol variables and:

        - adds variables where needed;
        - updates model equations (e.g. set by defines and clamp-tos);
        - updates model initial values.

        To create a variable, it may be necessary to infer its units from an RHS specified by the user. This requires
        the RHS to be available as a sympy expression where all variables have known units. Because the RHS can include
        references to other variables yet to be created, these operations cannot be separated so must all be handled
        here at once.
        """
        # Create set of variables, and process them in one or more passes, until all are done or the situation is
        # unresolvable.
        todo = deque(self.protocol_variables)
        while todo:
            # Perform a single pass over the todo-variables, and check that at least one gets done
            done = False
            missing_variables = set()
            for i in range(len(todo)):
                pvar = todo.popleft()

                # RHS to set in this iteration
                rhs = None

                # Add variable if required
                if pvar.model_variable is None:

                    # Check if there's enough information to define the variable's RHS
                    if pvar.equation is None and pvar.initial_value is None and pvar.default_expr is None:
                        # No definition given for variable, this is OK if it's optional
                        if pvar.is_optional:
                            done = True
                            continue
                        # For output-only variables, we can't tell if this is OK yet (might be vector outputs)
                        elif pvar.is_output and not pvar.is_input:
                            done = True
                            continue
                        # But otherwise an error
                        elif pvar.is_local:
                            raise ProtocolError(f'No definition given for local variable {pvar.long_name}.')
                        else:
                            raise ProtocolError(f'No definition given for non-optional variable {pvar.long_name}.')

                    # Get units to create variable with
                    if pvar.units is not None:
                        units = self.units.get_unit(pvar.units)
                    else:
                        # No units specified, try getting from RHS
                        rhs = pvar.default_expr if pvar.equation is None else pvar.equation.rhs
                        if rhs is None:
                            # Not enough information to create, OK if optional, but otherwise an error
                            if pvar.is_optional:
                                done = True
                                continue
                            else:
                                raise ProtocolError(f'No units specified for non-optional variable {pvar.long_name}.')

                        # Try getting sympy RHS
                        # Note: This involves checking both variable dependencies, and dependencies on units of
                        # variables via numbers in ``units_of(..)``. So it's not enough at this point that all variables
                        # we depend on are resolved, we also need all variables mentioned in units_of() to be resolved.
                        # Both will raise a MissingVariableError if they can't be resolved.
                        try:
                            rhs = rhs.to_sympy(self._variable_generator, self._number_generator)
                        except MissingVariableError as e:
                            missing_variables.add(e.name)
                            # Unable to create at this time, but may be able to at a next pass
                            todo.append(pvar)
                            continue

                        # New variable, can't contain an `original_definition` in its RHS
                        if ORIGINAL_DEFINITION in rhs.atoms():
                            raise ProtocolError(
                                'The "original_definition" construct can only be used when redefining an existing'
                                ' variable. Found it when creating variable for {pvar.long_name}.')

                        # Get units for the variable to create from its RHS, fixing inconsistencies if required
                        units, rhs = self.units.evaluate_units_and_fix(rhs)

                        # If an LHS is provided the new variable could be a state variable; if so, we'll need to
                        # multiply the RHS units by the time units to get the variable units.
                        if pvar.equation is not None and pvar.equation.is_ode:
                            units *= self.time_variable.units

                    # Create variable, and annotate if possible
                    name = self.model.get_unique_name('protocol__' + pvar.short_name)
                    pvar.model_variable = self.model.add_variable(name, units)
                    rdf_terms = pvar.input_terms + pvar.output_terms
                    if rdf_terms:
                        self.model.add_cmeta_id(pvar.model_variable)
                        for rdf_term in rdf_terms:
                            self.model.rdf.add((pvar.model_variable.rdf_identity, PRED_IS, rdf_term))

                    # Store local variables
                    if pvar.is_local:
                        self.local_vars[pvar.name] = pvar.model_variable

                # At this point the model variable is guaranteed to exist (and be in the right units)

                # Add or replace equation, if required
                eq = self.model.get_definition(pvar.model_variable)
                if pvar.equation is not None or (pvar.default_expr is not None and eq is None):

                    # Get sympy RHS (or re-use one we just created)
                    if rhs is None:
                        rhs = pvar.default_expr if pvar.equation is None else pvar.equation.rhs
                        try:
                            rhs = rhs.to_sympy(self._variable_generator, self._number_generator)
                        except MissingVariableError as e:
                            missing_variables.add(e.name)
                            # Unable to create at this time, but may be able to at a next pass
                            todo.append(pvar)
                            continue

                    # Check if the RHS for this variable contains a reference to ``original_definition``
                    if ORIGINAL_DEFINITION in rhs.atoms():
                        if eq is None:
                            raise ProtocolError(
                                'The "original_definition" construct can only be used when redefining an existing'
                                ' variable. Found it when creating variable for {pvar.long_name}.')

                        # Create a new variable with the original equation for this variable, and replace the
                        # original_definition symbol in the equation.
                        org_def = self.model.add_variable(
                            name=self.model.get_unique_name('original_rhs_of__' + pvar.short_name),
                            units=self.units.evaluate_units(eq.lhs),
                        )
                        self.model.add_equation(sympy.Eq(org_def, eq.rhs))
                        rhs = rhs.xreplace({ORIGINAL_DEFINITION: org_def})

                    # Get sympy lhs
                    if pvar.equation is not None:
                        lhs = pvar.equation.lhs_to_sympy(self._variable_generator, self._number_generator)
                    else:
                        lhs = pvar.model_variable

                    # Setting as state variable? Then make sure an initial value exists
                    if lhs.is_Derivative and pvar.model_variable.initial_value is None and pvar.initial_value is None:
                        raise ProtocolError(f'The variable {pvar.long_name} is being set as a state variable but has no'
                                            ' initial value (this can be set using an `input` or `var` statement).')

                    # Create new equation
                    # If required, convert units within RHS to make it consistent and match the LHS units
                    new_eq = sympy.Eq(lhs, rhs)
                    new_eq = self.units.convert_expression_recursively(new_eq, None)

                    # Remove existing equation
                    if eq is not None:
                        self.model.remove_equation(eq)

                    # Add/remove state_variable annotation if needed
                    if eq is not None and eq.lhs.is_Derivative and not new_eq.lhs.is_Derivative:
                        self.model.rdf.remove((pvar.model_variable.rdf_identity, PRED_IS_VERSION_OF, STATE_ANNOTATION))
                    elif new_eq.lhs.is_Derivative:
                        self.model.add_cmeta_id(pvar.model_variable)
                        self.model.rdf.add((pvar.model_variable.rdf_identity, PRED_IS_VERSION_OF, STATE_ANNOTATION))

                    # Add or replace equation
                    self.model.add_equation(new_eq)
                    eq = new_eq

                # Set initial value, if required
                if pvar.initial_value is not None:

                    # Initial value used as short-hand for constant variable
                    # This is allowed if:
                    #   We're defining a new variable (CellML style!)
                    #   We're replacing a constant variable, that's not also set by a `define` or `clamp`
                    if eq is None or (not eq.lhs.is_Derivative and pvar.equation is None):
                        lhs = pvar.model_variable
                        rhs = self.model.add_number(pvar.initial_value, pvar.model_variable.units)
                        if eq is not None:
                            self.model.remove_equation(eq)
                        eq = sympy.Eq(lhs, rhs)
                        self.model.add_equation(eq)

                    # Initial value for state
                    elif eq.lhs.is_Derivative:
                        pvar.model_variable.initial_value = pvar.initial_value

                    # Overdefined model
                    else:
                        raise ProtocolError(
                            f'Initial value provided for {pvar.long_name}, which is not a state variable.')

                # Variable handled OK
                done = True

            if not done:
                # No changes in pass implies there are missing variables in the RHS of at least one variable.
                todo = ', '.join([pvar.name for pvar in todo])
                unknown = ', '.join(['"' + x + '"' for x in missing_variables])
                raise ProtocolError(
                    f'Unable to create or set equations for {todo}.'
                    f' References found to unknown variable(s): {unknown}.')

    def _handle_clamping(self):
        """Clamp requested variables to their initial values."""
        for clamp in self.clamps:
            var = self.model.get_variable_by_ontology_term(clamp.rdf_term)
            defn = self.model.get_definition(var)
            if var.initial_value is None:
                value = defn.rhs.evalf()  # TODO: What if there are variable references in the RHS?
            else:
                value = var.initial_value
            new_defn = sympy.Eq(var, self.model.add_number(value, var.units))
            if defn is not None:
                self.model.remove_equation(defn)
            self.model.add_equation(new_defn)

    def _process_transitive_variables(self):
        """
        Searches for transitive variables (for vector outputs), converts their units if needed, and updates the
        information stored in the ProtocolVariable objects.

        This is also the first time we can check that all outputs are specified correctly, so errors will be raised here
        if protocol variables are not found.
        """
        # TODO: We only check transitive variables for outputs that have not been resolved to model variables. At the
        # moment there's no checking that e.g. inputs don't have transitive variables, or that resolved outputs are not
        # somehow vector outputs at the same time.

        # Create a mapping from model variables to a ProtocolVariable that sets a unit, to check for clashes between
        # units set by scalar and vector annotations, or between two vector annotations.
        varmap = {}
        for pvar in self.protocol_variables:
            if pvar.units is not None and pvar.model_variable is not None:
                varmap[pvar.model_variable] = pvar

        # Check all outputs, see if unresolved ones are vector variables
        for pvar in self.protocol_variables:

            # Check that this is an unresolved output
            if pvar.is_output and pvar.model_variable is None:

                # Handle 'state_variable' annotation separately
                if pvar is self.magic_pvar:
                    continue

                # Get transitive variables
                variables = get_variables_transitively(self.model, pvar.output_terms[0])    # TODO: Ensure max 1 term
                if variables:

                    # Convert units
                    if pvar.units is not None:

                        # Check if any variables will cause a clash with previously set units
                        for var in variables:
                            # Get pvar that set a unit, or store this one as the unit-setting pvar
                            try:
                                other = varmap[var]
                            except KeyError:
                                varmap[var] = pvar
                                continue

                            # Check if the units match
                            if other.units != pvar.units:
                                if other.is_vector:
                                    raise ProtocolError(
                                        f'Conflicting units set for items that are both in vector output {pvar.name}'
                                        f' ({pvar.units}) and {other.name} ({other.units}).')
                                else:
                                    raise ProtocolError(
                                        f'Conflicting units set for item in vector output {pvar.name} ({pvar.units})'
                                        f' that is set individually as {other.long_name} ({other.units}).')

                        # Convert all variables
                        units = self.units.get_unit(pvar.units)
                        converted = []
                        for var in variables:
                            converted.append(self.model.convert_variable(var, units, DataDirectionFlow.OUTPUT))
                        variables = converted

                    # Order variables by display name
                    variables.sort(key=lambda var: self.model.get_display_name(var))

                    # Update ProtocolVariable object
                    pvar.update(is_vector=True, vector_variables=variables)

                elif not pvar.is_optional:
                    raise ProtocolError(f'No variable(s) found for non-optional output {pvar.long_name}.')

    def _gather_state_variables(self, original_state_order):
        """
        Gather all variables annotated as states and store them in the the ProtocolVariable for the
        ``oxmeta:state_variable annotation`` (if used by the protocol).

        :param original_state_order: An list containing the rdf identities of the variables as they originally appeared
            (before any model manipulation).
        """
        if self.magic_pvar is None:
            return

        # Gather state variables
        states = list(get_variables_that_are_version_of(self.model, STATE_ANNOTATION))

        # Create ordering, preserving original order as much as possible
        order = []
        todo = set([var.rdf_identity for var in states])
        for rdf_identity in original_state_order:
            try:
                todo.remove(rdf_identity)
            except KeyError:
                continue
            order.append(rdf_identity)
        order.extend(todo)
        order = {rdf_identity: i for i, rdf_identity in enumerate(order)}

        # Order states and store
        states.sort(key=lambda var: order[var.rdf_identity])
        self.magic_pvar.vector_variables = states
        self.magic_pvar.update(is_vector=True)

    def _purge_unused_mathematics(self):
        """Remove model equations and variables not needed for generating desired outputs."""

        # Create set of required variables
        required_variables = set()

        # Time is always needed, even if there are no state variables!
        required_variables.add(self.time_variable)

        # All protocol variables are required.
        for pvar in self.protocol_variables:
            if pvar.model_variable is not None:
                required_variables.add(pvar.model_variable)
            required_variables.update(pvar.vector_variables)

        # Add all variables used to compute the required variables.
        for variable in list(required_variables):
            required_variables.update(networkx.ancestors(self.model.graph, variable))

        # State variables don't have ancestors, but their derivative equations might. So loop over these and add any new
        # requirements. Do this iteratively in case the new dependencies on derivatives are introduced in the process.
        derivatives = self.model.get_derivatives()
        old_len = 0
        while old_len != len(required_variables):
            old_len = len(required_variables)
            for deriv in derivatives:
                if deriv.args[0] in required_variables:
                    required_variables.update(networkx.ancestors(self.model.graph, deriv))

        # Now figure out which variables *aren't* used
        all_variables = set(self.model.variables())
        unused_variables = all_variables - required_variables

        # Remove them and their definitions
        for variable in unused_variables:
            self.model.remove_variable(variable)

        # Add time back in to the graph if needed
        if self.time_variable not in self.model.graph.nodes:
            self.model.graph.add_node(self.time_variable, equation=None, variable_type='free')


######################################################################
# Simulation tasks section
######################################################################


class Range(BaseGroupAction):
    """Parse action for all the kinds of range supported."""

    def _expr(self):
        name = self.get_named_token_as_string('name', None)
        if 'uniform' in self.tokens:
            tokens = self.tokens['uniform'][0]
            start = tokens[0].expr()
            stop = tokens[-1].expr()
            if len(tokens) == 3:
                step = tokens[1].expr()
            else:
                step = E.Const(V.Simple(1))
            range_ = ranges.UniformRange(name, start, stop, step)
        elif 'vector' in self.tokens:
            expr = self.tokens['vector'][0].expr()
            range_ = ranges.VectorRange(name, expr)
        elif 'while' in self.tokens:
            cond = self.tokens['while'][0].expr()
            range_ = ranges.While(name, cond)
        return range_


class ModifierWhen(BaseGroupAction):
    """Parse action for the when part of modifiers."""

    def _expr(self):
        when = {'start': 'START_ONLY', 'each': 'EACH_LOOP', 'end': 'END_ONLY'}[self.tokens]
        return getattr(modifiers.AbstractModifier, when)


class Modifier(BaseGroupAction):
    """Parse action that generates all kinds of modifier."""

    def _expr(self):
        args = [self.tokens[0].expr()]
        detail = self.tokens[1]
        if 'set' in self.tokens[1]:
            modifier = modifiers.SetVariable
            args.append(detail[0])
            args.append(detail[1].expr())
        elif 'save' in self.tokens[1]:
            modifier = modifiers.SaveState
            args.append(detail[0])
        elif 'reset' in self.tokens[1]:
            modifier = modifiers.ResetState
            if len(detail) > 0:
                args.append(detail[0])
        return modifier(*args)


class Modifiers(BaseGroupAction):
    """Parse action for the modifiers collection."""

    def _expr(self):
        return self.get_children_expr()


class TimecourseSimulation(BaseGroupAction):

    def _expr(self):
        args = self.get_children_expr()
        return simulations.Timecourse(*args)


class NestedSimulation(BaseGroupAction):

    def _expr(self):
        args = [t.expr() for t in self.tokens[0:-1]]
        if len(args) == 1:
            # Add an empty modifiers element
            args.append(self.delegate('Modifiers', [[]]).expr())
        nested = self.tokens[-1][0]
        if isinstance(nested, (Simulation, NestedProtocol)):
            # Inline definition
            args.append(nested.expr())
        return simulations.Nested(args[2], args[0], args[1])


class OneStepSimulation(BaseGroupAction):
    # Leaving old XML method in to document existing properties / tokens.
    # def _xml(self):
    #    attrs = {}
    #    args = []
    #    if 'step' in self.tokens:
    #        attrs['step'] = str(self.tokens['step'][0])
    #    if 'modifiers' in self.tokens:
    #        args.append(self.tokens['modifiers'][0].xml())
    #    return P.oneStep(*args, **attrs)
    pass


class NestedProtocol(BaseGroupAction):
    def __init__(self, s, loc, tokens):
        self.trace = (tokens[0][-1] == '?')
        if self.trace:
            tokens[0] = tokens[0][:-1]
        super(NestedProtocol, self).__init__(s, loc, tokens)
        # Resolve any relative path to the nested protocol
        self.proto_path = os.path.join(os.path.dirname(source_file), self.tokens[0])

    def _expr(self):
        args = [self.proto_path]
        inputs = {}
        assert isinstance(self.tokens[1], StatementList)
        for assignment in self.tokens[1]:
            input_name = assignment.tokens[0].tokens
            input_value = assignment.tokens[1].expr()
            inputs[input_name] = input_value
        args.append(inputs)
        output_names, optional_flags = [], []
        for output in self.tokens[2:]:
            output_names.append(output[-1])
            optional_flags.append(len(output) == 2)
        args.extend([output_names, optional_flags])
        nested_proto = model.NestedProtocol(*args)
        result = simulations.OneStep(0)
        result.set_model(nested_proto)
        return result


class Simulation(BaseGroupAction):
    """Parse action for all kinds of simulation."""

    def _expr(self):
        sim = self.tokens[1].expr()
        sim.prefix = str(self.tokens[0])
        return sim


class Tasks(BaseGroupAction):
    """Parse action for a collection of simulation tasks."""

    def _expr(self):
        sims = self.get_children_expr()
        return sims

######################################################################
# Other protocol language constructs
######################################################################


class Inputs(BaseAction):
    """Parse action for the inputs section of a protocol."""

    def _expr(self):
        assert len(self.tokens) <= 1
        if len(self.tokens) == 1:  # Don't create an empty element
            return self.tokens[0].expr()


class Import(BaseGroupAction):
    """Parse action for protocol imports."""

    def _expr(self):
        assert len(self.tokens) >= 2
        set_inputs = {}
        if len(self.tokens) == 3:
            for set_input in self.tokens[2].tokens:
                name = set_input.tokens[0].tokens
                value_expr = set_input.tokens[1].expr()
                set_inputs[name] = value_expr
        return self.tokens[0], self.tokens[1], set_inputs


class UnitRef(BaseGroupAction):
    """Parse action for unit references within units definitions.

    Properties:

    ``units``
        Name of the units referenced.
    ``prefix``
        Optional SI prefix name, e.g. milli, deca. Defaults to ''.
    ``multiplier``
        Optional scalar multiplier; None if omitted.
    ``exponent``
        Optional exponent; None if omitted.
    ``offset``
        For defining units such as fahrenheit. Will raise an error if given.
    ``pint_expression``
        Pint syntax for this unit reference.
    """
    def _expr(self):
        if 'offset' in self.tokens:
            raise ValueError('Offset units are no longer supported')
        self.units = self.get_named_token_as_string('units')
        assert self.units is not None
        self.prefix = self.get_named_token_as_string('prefix', '')
        self.exponent = self.get_named_token_as_string('exponent')
        self.multiplier = self.get_named_token_as_string('multiplier')

        expr = self.units
        if self.prefix:
            expr = '({:e} * {})'.format(UNIT_PREFIXES[self.prefix], self.units)
        if self.exponent is not None:
            expr = '{}**{}'.format(expr, self.exponent)
        if self.multiplier is not None:
            expr = '{} * {}'.format(self.multiplier, expr)
        self.pint_expression = expr

        return self


class UnitsDef(BaseGroupAction):
    """Parse action for units definitions.

    Example definitions::

        my_units = 50 deci metre^2
        mV = milli volt
        uA_per_cm2 = micro ampere . centi metre^-2 "{/Symbol m}A/cm^2"
        A_per_F = ampere . farad^-1
        mM = milli mole . litre^-1 "{/Symbol m}M"

    Properties:

    ``name``
        The name of the units defined.
    ``description``
        Optional formatted description of the units, or None.
    ``unit_refs``
        List of UnitRef instances comprising the definition.
    ``pint_expression``
        Definition of these units in pint syntax.
    """
    def _expr(self):
        self.name = str(self.tokens[0])
        self.description = self.tokens.get('description')
        self.unit_refs = [t.expr() for t in self.tokens if isinstance(t, UnitRef)]
        self.pint_expression = ' * '.join(r.pint_expression for r in self.unit_refs)

        return self


class Units(BaseAction):
    """Parse action for the units definitions section."""
    def _expr(self):
        return self.get_children_expr()


class Library(BaseAction):
    """Parse action for the library section."""

    def _expr(self):
        if len(self.tokens) > 0:
            assert len(self.tokens) == 1
            return self.tokens[0].expr()


class PostProcessing(BaseAction):
    """Parse action for the post-processing section."""

    def _expr(self):
        if len(self.tokens) > 0:
            return self.delegate('StatementList', [self.tokens]).expr()


class Output(BaseGroupAction):
    """Parse action for a protocol output specification."""

    def _expr(self):
        output = {}
        if 'units' in self.tokens:
            output['units'] = str(self.tokens['units'])
        if 'name' in self.tokens:
            output['name'] = str(self.tokens['name'])
        if 'ref' in self.tokens:
            output['ref'] = str(self.tokens['ref'])
        if 'description' in self.tokens:
            output['description'] = str(self.tokens['description'])
        output['optional'] = 'optional' in self.tokens
        return output


class Outputs(BaseGroupAction):
    """Parse action for the outputs section."""

    def _expr(self):
        return self.get_children_expr()


class Plot(BaseGroupAction):
    """Parse action for simple plot specifications."""

    def _expr(self):
        using = self.tokens.get('using', '')
        if using:
            expected_num_tokens = 3
        else:
            expected_num_tokens = 2
        assert len(self.tokens) == expected_num_tokens, "Only a single plot curve is currently supported"
        curve = self.tokens[-1]
        key = curve.get('key', '')
        if key:
            curve = curve[:-1]
        assert len(curve) == 2, "Only a single y variable is currently supported"
        title = str(self.tokens[0])
        y, x = list(map(str, curve))
        plot = {'title': title, 'x': x, 'y': y}
        if key:
            plot['key'] = key
        if using:
            plot['using'] = using[0]
        return plot


class Plots(BaseGroupAction):
    """Parse action for the plots section."""

    def _expr(self):
        return self.get_children_expr()


class Protocol(BaseAction):
    """Parse action for a full protocol."""

    def _expr(self):
        d = {}
        d['imports'] = []
        for token in self.tokens:
            if isinstance(token, Inputs):
                d['inputs'] = token.expr()
            if isinstance(token, Import):
                d['imports'].append(token.expr())
            if isinstance(token, Library):
                d['library'] = token.expr()
            if isinstance(token, Units):
                d['units'] = token.expr()
            if isinstance(token, ModelInterface):
                d['model_interface'] = token.expr()
            if isinstance(token, Tasks):
                d['simulations'] = token.expr()
            if isinstance(token, PostProcessing):
                d['postprocessing'] = token.expr()
            if isinstance(token, Outputs):
                d['outputs'] = token.expr()
            if isinstance(token, Plots):
                d['plots'] = token.expr()

        if 'dox' in self.tokens:
            d['dox'] = self.tokens['dox'][0]

        # Scan for any declared namespaces
        ns_map = {}
        if 'namespace' in self.tokens:
            for prefix, uri in self.tokens['namespace']:
                ns_map[prefix] = uri
        d['ns_map'] = ns_map

        return d
