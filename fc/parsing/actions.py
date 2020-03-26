"""
Parse actions that can generate Python implementation objects
"""

import itertools
import math
import os

import pyparsing as p
import sympy
from cellmlmanip.model import DataDirectionFlow
from cellmlmanip.model import VariableDummy
from cellmlmanip.parser import UNIT_PREFIXES

from ..error_handling import ProtocolError
from ..language import expressions as E
from ..language import statements as S
from ..language import values as V
from ..locatable import Locatable
from ..simulations import model, modifiers, ranges, simulations
from .rdf import OXMETA_NS, PRED_IS, PRED_IS_VERSION_OF, create_rdf_node, get_variables_transitively


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

source_file = ""  # Will be filled in CompactSyntaxParser.try_parse


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
                source_file, p.lineno(loc, s), p.col(loc, s), p.line(loc, s))

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

    def get_attribute_dict(self, *attrNames):
        """Create an attribute dictionary from named parse results."""
        attrs = {}
        for key in attrNames:
            if key in self.tokens:
                attrs[key] = self.get_named_token_as_string(key)
        return attrs

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
        raise NotImplementedError


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
            # We have a units annotation
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
    """Mixin providing properties for resolving variable references.

    Used by input & output variable specifications, inter alia.

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
    def set_name(self, name):
        """Set the name used for this reference. For use by subclasses."""
        self.prefixed_name = name
        self.ns_prefix, self.local_name = name.split(':', 1)
        self.ns_uri = None  # Will be set later using protocol's namespace mapping
        self.rdf_term = None  # Ditto

    def _expr(self):
        self.set_name(self.get_named_token_as_string('name'))
        return self

    def set_namespace(self, ns_uri):
        """Set the full namespace URI for this reference, and hence the RDF term."""
        self.ns_uri = ns_uri
        self.rdf_term = create_rdf_node((self.ns_uri, self.local_name))

    @classmethod
    def create(cls, prefix, uri, local_name):
        """Helper method to create fake references for testing."""
        ref = cls()
        ref.prefixed_name = '{}:{}'.format(prefix, local_name)
        ref.ns_prefix = prefix
        ref.local_name = local_name
        ref.set_namespace(uri)
        return ref


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
        if 'default' in self.tokens:
            self.default_expr = s[self.tokens['default_start']:self.tokens['default_end']]
        else:
            self.default_expr = ''


class DeclareVariable(BaseGroupAction):
    # Leaving old XML method in to document existing properties.
    # def _xml(self):
    #    return P.declareNewVariable(**self.get_attribute_dict('name', 'units', 'initial_value'))
    pass


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
            self.set_name(name.tokens)
            return self
        else:
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

        Typically this will be called via :meth:`ModelInterface.sympy_equations` which sets up appropriate
        generators.

        :param variable_generator: a function to resolve a name reference within the equation to a model variable.
        :param number_generator: converts a number with units to a Sympy entity.
        """
        var = self.var.to_sympy(variable_generator, number_generator)
        if self.is_ode:
            bvar = self.bvar.to_sympy(variable_generator, number_generator)
            lhs = sympy.Derivative(var, bvar, evaluate=False)
        else:
            lhs = var
        rhs = self.rhs.to_sympy(variable_generator, number_generator)
        return sympy.Eq(lhs, rhs)


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
    # Leaving old XML method in to document existing properties / tokens.
    # def _xml(self):
    #    attrs = self.get_attribute_dict('desiredDimensions', 'actualDimensions')
    #    rule = self.tokens[-1].xml()
    #    return P.unitsConversionRule(rule, **attrs)
    pass


class ModelInterface(BaseGroupAction):
    """Parse action for the model interface section of a protocol.

    See https://chaste.cs.ox.ac.uk/trac/wiki/FunctionalCuration/ProtocolSyntax#Modelinterface for more on the syntax
    and semantics of the model interface.

    Includes helper methods for merging model interfaces, e.g. when one protocol imports another.

    Properties:

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
    ``sympy_equations``
        Once :meth:`modify_model` and :meth:`resolve_namespaces` have been called, this property gives Sympy versions of
        ``self.equations``.
    ``initial_values``
        Initial values for constants or state variables, defined by the inputs. Stored in a map from variable (as an
        RDF term) to value (as a float).
    ``vector_orderings``
        Used for consistent code generation of vector outputs.
    ``parameters``
        Once :meth:`modify_model` has been called, this property gives a list of those model inputs (as
        :class:`InputVariable` objects) that are constants (unless the protocol changes them while running).
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

        # Initialise
        self._reinit()

    def _reinit(self):
        """(Re-)initialise this interface, so that it can be re-used."""

        self._sympy_equations = None
        self.initial_values = {}
        self.units = None   # Will be the protocol's UnitStore
        self.model = None   # The model to be modified
        self.time_variable = None   # The model's time (free) variable
        self.ns_map = None  # Map NS prefixes to URIs, as defined by the protocol
        self.parameters = []
        self.vector_orderings = {}

    def _expr(self):
        actions = self.get_children_expr()
        self._time_units = [a for a in actions if isinstance(a, SetTimeUnits)]
        self.inputs = [a for a in actions if isinstance(a, InputVariable)]
        self.outputs = [a for a in actions if isinstance(a, OutputVariable)]
        self.optional_decls = [a for a in actions if isinstance(a, OptionalVariable)]
        self.equations = [a for a in actions if isinstance(a, ModelEquation)]
        self.clamps = [a for a in actions if isinstance(a, ClampVariable)]

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

    def modify_model(self, model, units):
        """Use the definitions in this interface to transform the provided model.

        This calls various internal helper methods to do the modifications, in an order orchestrated to
        follow the principle of least surprise for protocol authors. It attempts to produce results that
        most probably match what they expect to happen, without creating an inconsistent model.

        Key steps are:
        - Adding/checking variables defined as model inputs (setable by the protocol). See
          :meth:`_add_input_variables`.
        - Adding or replacing equations in the model's mathematics (:meth:`_add_or_replace_equations`).
        - Clamping variables to their initial value (:meth:`_handle_clamping`).
        - Annotating the variables now comprising the state variable vector so they are recognised by
          the oxmeta:state_variable 'magic' ontology term (:meth:`_annotate_state_variables`).
        - Units conversions are applied where appropriate on model inputs and outputs, and on added/changed
          equations, so the protocol sees quantities in the units it requested.
          See e.g. :meth:`_convert_time_if_needed`, :meth:`_convert_output_units`.
        - Model variables and equations not needed to compute the requested outputs are removed. See
          :meth:`_purge_unused_mathematics`.

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

        # Convert free variable units
        self._convert_time_if_needed()

        # Ensure inputs exist, adding them if needed, and storing initial values set by user.
        # This also performs unit conversion where needed for existing input variables.
        self._add_input_variables()

        # Now that all variables are in place, perform sanity checks on model-protocol combination before making any
        # modifications
        self._sanity_check()

        # TODO: Add variables defined with ``var`` statements

        # Process ``define`` statements, adding or replacing equations where needed,
        # and setting any initial values defined through inputs.
        self._add_or_replace_equations()

        # Process ``clamp`` statements
        self._handle_clamping()

        self._annotate_state_variables()

        # Convert units for output variables
        output_variables = self._convert_output_units()

        # Populate list of input parameters
        self._list_parameters()

        self._purge_unused_mathematics(output_variables)
        # TODO: Any final consistency checks on the model?

    def _sanity_check(self):
        """Perform initial sanity checks on the protocol interface."""

        # Variables can only appear as input or output once
        # If a variable appears as an input and an ouput, both must have the same units
        input_units = {}            # To check input vs output units
        output_units = {}           # To check input vs output units
        initial_values = set()      # To check against overdefinedness
        for ref in self.inputs:
            try:
                var = self.model.get_variable_by_ontology_term(ref.rdf_term)
            except KeyError:
                continue
            if var in input_units:
                raise ProtocolError('The variable ' + str(var) + ' was specified as an input twice.')
            input_units[var] = ref.units
            if ref.initial_value is not None:
                initial_values.add(var)
        for ref in self.outputs:
            try:
                var = self.model.get_variable_by_ontology_term(ref.rdf_term)
            except KeyError:
                continue
            if var in output_units:
                raise ProtocolError('The variable ' + str(var) + ' was specified as an output twice.')
            output_units[var] = ref.units
            units = input_units.get(var)
            if ref.units is not None and units is not None and ref.units != units:
                raise ProtocolError(
                    'The variable ' + str(var) + ' appears as input and output, but with different units.')

        # Check against overdefinedness: Variables cannot appear as an LHS of an equation more than once (e.g. used in
        # two defines, or used in a ``clamp x to 1`` and a define), and variables clamped to their current value can not
        # also be defined.
        seen = set()
        for ref in self.clamps:
            var = self.model.get_variable_by_ontology_term(ref.rdf_term)
            if var in seen:
                raise ProtocolError('The variable ' + str(var) + ' is set by multiple clamp statements.')
            seen.add(var)
        for eq in self.sympy_equations:
            var = eq.lhs.args[0] if eq.is_Derivative else eq.lhs
            if var in seen:
                raise ProtocolError(
                    'The variable ' + str(var) + ' is set by more than one clamp and/or define statement.')
            seen.add(var)

        # Note: Variables set with `define` may have an initial value (if they are defined through their derivatives),
        # this is checked later.

        # TODO: What about the `default` part of an `optional` statement?

    def _variable_generator(self, name):
        """Resolve a name reference within a model interface equation to a variable in the model.

        Used by :meth:`sympy_equations`.

        :param str name: a name reference. If it contains a ':' then it is treated as a prefix:local_name ontology term
            reference. The prefix is looked up in our namespace URI map, and the variable then found with
            :meth:`cellmlmanip.model.Model.get_variable_by_ontology_term`. Otherwise we look for a variable defined
            within the protocol's model interface using :class:`DeclareVariable`.
        """
        if ':' in name:
            prefix, local_name = name.split(':', 1)
            ns_uri = self.ns_map[prefix]
            return self.model.get_variable_by_ontology_term((ns_uri, local_name))
        else:
            # TODO: DeclareVariable not yet done
            raise NotImplementedError

    def _number_generator(self, value, units):
        """Convert a number with units in an equation to a :class:`cellmlmanip.model.NumberDummy`.

        Used by :meth:`sympy_equations`.

        :param value: the numerical value
        :param units: the *name* of the units for this quantity. Will be looked up from the protocol's definitions.
        """
        return self.model.add_number(value, self.units.get_unit(units))

    @property
    def sympy_equations(self):
        """The equations defined by the interface in Sympy form.

        Requires :meth:`modify_model` to have been called.
        """
        if self._sympy_equations is None:
            # Do the transformation
            self._sympy_equations = eqs = []
            for eq in self.equations:
                # TODO: Check whether lhs exists in the model; if not, and it is an output or optional, add it.
                # Units should be taken from the output spec (if present) or the RHS.
                # TODO: If there are variables on the RHS that don't exist, this is an error unless both the
                # missing variable and LHS are optional. In which case, just skip the equation (but log it).
                eqs.append(eq.to_sympy(self._variable_generator, self._number_generator))
        return self._sympy_equations

    #######################################
    # Helper methods for model manipulation

    def _convert_time_if_needed(self):
        """Units-convert the time variable if not in the protocol's units."""
        if self.time_units:
            time_units = self.units.get_unit(self.time_units)
            self.time_variable = self.model.convert_variable(self.time_variable, time_units, DataDirectionFlow.INPUT)

    def _add_input_variables(self):
        """Ensure requested input variables exist, unless they are optional.

        If the variable exists its units are checked and converted if needed.

        If it doesn't exist but is marked as optional, nothing is done.

        If it doesn't exist and is *not* optional, then it needs to be created here, which requires that
        units are given. If not, an error is raised.
        """
        for var in self.inputs:

            # Find variable symbol or create a new one
            variable = None
            try:
                variable = self.model.get_variable_by_ontology_term(var.rdf_term)
            except KeyError:
                # TODO: Check if variable is optional; skip if so. Add a helper method is_optional that
                # compares var.prefixed_name against self.optional_decls?
                optional = False
                if optional:
                    continue

                # Check units are given for new variable
                if var.units is None:
                    raise ProtocolError(
                        'Units must be specified for input variables not appearing in the model;'
                        ' none are given for ' + var.prefixed_name
                    )

                # Add the new input variable, with ontology annotation
                # TODO: Extract this into a helper method that DeclareVariable processing etc can also use
                name = self.model._get_unique_name('protocol__' + var.local_name)  # TODO: Make method public
                units = self.units.get_unit(var.units)
                variable = self.model.add_variable(name, units)
                self.model.add_cmeta_id(variable)
                self.model.rdf.add((variable.rdf_identity, PRED_IS, var.rdf_term))

            # Store initial value if given
            if var.initial_value is not None:
                self.initial_values[var.rdf_term] = var.initial_value

            # Maintain link to time variable, if needed
            is_time = variable is self.time_variable

            # Convert units if needed
            if var.units is not None:
                units = self.units.get_unit(var.units)
                if units != variable.units:
                    # print('Converting input ' + str(var.rdf_term) + ' to units ' + str(units))
                    variable = self.model.convert_variable(variable, units, DataDirectionFlow.INPUT)

                    # Update cached time variable
                    if is_time:
                        self.time_variable = variable

    def _add_or_replace_equations(self):
        """
        Modify the model by adding and/or replacing equations and setting initial values according to the specified
        inputs and define statements.
        """
        # Create map from model variables to initial values
        initial_values = {self.model.get_variable_by_ontology_term(k): v for k, v in self.initial_values.items()}

        # Apply all modifications (sympy_equations contains _only_ equations from define statements)
        for eq in self.sympy_equations:
            lhs = eq.lhs
            if lhs.is_Derivative:
                var = lhs.args[0]

                # Initial value must be set via inputs, or already be set (if this was already a state variable)
                if var.initial_value is None and var not in initial_values:
                    terms = '/'.join(str(x) for x in self.model.get_ontology_terms_by_variable(var))
                    raise ProtocolError(
                        'Variable {} is being set as a state variable but has no initial value (this can be set using'
                        '  an `input` statement)'.format(terms))
            else:
                var = lhs

                # Check that this variable isn't doubly defined
                if var in initial_values:
                    raise ProtocolError(
                        'Overdefined variable. An initial value was given for {} via an input statement, but a new'
                        ' equation was also set via a define statement.')

                # Unset initial value, in case it was a state variable previously
                var.initial_value = None

            # Remove existing equation if needed
            old_eq = self.model.get_definition(var)
            if old_eq is not None:
                self.model.remove_equation(old_eq)

            # Add new equation
            self.model.add_equation(eq)

        # Apply all initial values set in the inputs
        for var, value in initial_values.items():
            if self.model.is_state(var):
                # Set initial value
                var.initial_value = value
            else:
                # Replace equation
                old_eq = self.model.get_definition(var)
                if old_eq is not None:
                    self.model.remove_equation(old_eq)
                self.model.add_equation(sympy.Eq(var, self.model.add_number(value, var.units)))

        # TODO: Check that all/any _new_ derivatives are w.r.t. self.time_variable?

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

    def _annotate_state_variables(self):
        """Annotate all state variables with the 'magic' oxmeta:state_variable term."""
        state_annotation = create_rdf_node((OXMETA_NS, 'state_variable'))
        self.vector_orderings[state_annotation] = {}
        for i, state_var in enumerate(self.model.get_state_variables()):
            self.model.add_cmeta_id(state_var)
            self.model.rdf.add((state_var.rdf_identity, PRED_IS_VERSION_OF, state_annotation))
            self.vector_orderings[state_annotation][state_var.rdf_identity] = i

    def _convert_output_units(self):
        """Convert units for all outputs if needed.

        :return: The set of variables appearing in outputs, either directly or as part of a vector, in the desired
            units.
        """
        output_variables = set()
        for output in self.outputs:
            variables = get_variables_transitively(self.model, output.rdf_term)
            if output.units is not None:
                desired_units = self.units.get_unit(output.units)
                for i, variable in enumerate(variables):
                    variables[i] = self.model.convert_variable(
                        variable, desired_units, DataDirectionFlow.OUTPUT)
            output_variables.update(variables)
        return output_variables

    def _list_parameters(self):
        """Populates the list of model parameters (inputs with a constant RHS)."""

        for var in self.inputs:
            try:
                variable = self.model.get_variable_by_ontology_term(var.rdf_term)
            except KeyError:
                # Skip optional variables
                continue

            # Parameters are inputs that aren't states, and have a constant RHS
            eq = self.model.get_definition(variable)
            if isinstance(eq.lhs, VariableDummy) and len(eq.rhs.atoms(VariableDummy)) == 0:
                self.parameters.append(var)

    def _purge_unused_mathematics(self, output_variables):
        """Remove model equations and variables not needed for generating desired outputs.

        :param output_variables: the set of variables appearing in outputs, either directly or as part of a vector
        """
        import networkx as nx
        graph = self.model.graph
        required_variables = set(output_variables)

        # Time is always needed, even if there are no state variables!
        required_variables.add(self.time_variable)

        # Symbols used directly in equations computing outputs
        for variable in output_variables:
            required_variables.update(nx.ancestors(graph, variable))

        # Symbols used indirectly to compute state variables referenced in equations
        derivatives = self.model.get_derivatives()
        old_len = 0
        while old_len != len(required_variables):
            old_len = len(required_variables)
            for deriv in derivatives:
                if deriv.args[0] in required_variables:
                    required_variables.update(nx.ancestors(graph, deriv))
                    # And we also need time...
                    required_variables.add(deriv.args[1])

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
        attrs = self.get_attribute_dict('name', 'units')
        if 'uniform' in self.tokens:
            tokens = self.tokens['uniform'][0]
            start = tokens[0].expr()
            stop = tokens[-1].expr()
            if len(tokens) == 3:
                step = tokens[1].expr()
            else:
                step = E.Const(V.Simple(1))
            range_ = ranges.UniformRange(attrs['name'], start, stop, step)
        elif 'vector' in self.tokens:
            expr = self.tokens['vector'][0].expr()
            range_ = ranges.VectorRange(attrs['name'], expr)
        elif 'while' in self.tokens:
            cond = self.tokens['while'][0].expr()
            range_ = ranges.While(attrs['name'], cond)
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

    def _expr(self):
        args = []
        proto_file = self.tokens[0]
        proto_file = os.path.join(os.path.dirname(source_file), proto_file)
        args.append(proto_file)
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
    """Parse action for an output specification."""

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

