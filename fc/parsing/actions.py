"""
Parse actions that can generate Python implementation objects
"""

import itertools
import math
import os

import pyparsing as p
import sympy

import fc.language.expressions as E
import fc.language.statements as S
import fc.language.values as V
from fc.locatable import Locatable
from fc.simulations import model, modifiers, ranges, simulations


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
        Create a Symbol parse action, used e.g. for the ``null`` symbol in the protocol
        language.
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

    def to_sympy(self, symbol_generator, number_generator):
        """Convert this parse tree to a Sympy expression.

        May be implemented by subclasses if the operation makes sense, i.e. they represent (part of) an expression.

        :param symbol_generator: Method to create expressions for symbols.
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

    def to_sympy(self, symbol_generator, number_generator):
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

    def to_sympy(self, symbol_generator, number_generator):
        return symbol_generator(self.tokens)


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


class InputVariable(BaseGroupAction):
    """
    Parse action for input variables defined in the model interface.

    ``input <prefix:term> [units <uname>] [= <initial_value>]``
    """
    def _expr(self):
        name = self.prefixed_name = self.get_named_token_as_string('name')
        self.ns_prefix, self.local_name = name.split(':', 1)
        self.ns_uri = None  # Will be set later using protocol's namespace mapping
        self.units = self.get_named_token_as_string('units', default=None)
        self.initial_value = self.get_named_token_as_string('initial_value', default=None)
        return self


class OutputVariable(BaseGroupAction):
    """
    Parse action for output variables defined in the model interface.

    ``output <prefix:term> [units <uname>]``
    """
    def _expr(self):
        name = self.prefixed_name = self.get_named_token_as_string('name')
        self.ns_prefix, self.local_name = name.split(':', 1)
        self.ns_uri = None  # Will be set later using protocol's namespace mapping
        self.units = self.get_named_token_as_string('units', default=None)
        return self


class OptionalVariable(BaseGroupAction):
    """Parse action for specifying optional variables in the model interface.

    ``optional <prefix:term> [default <simple_expr>]``
    """
    def __init__(self, s, loc, tokens):
        super(OptionalVariable, self).__init__(s, loc, tokens)
        name = self.prefixed_name = self.get_named_token_as_string('name')
        self.ns_prefix, self.local_name = name.split(':', 1)
        self.ns_uri = None  # Will be set later using protocol's namespace mapping
        if 'default' in self.tokens:
            # Record the actual string making up the default expression
            self.default_expr = s[self.tokens['default_start']:self.tokens['default_end']]
        else:
            self.default_expr = ''


class DeclareVariable(BaseGroupAction):
    # Leaving old XML method in to document existing properties.
    # def _xml(self):
    #    return P.declareNewVariable(**self.get_attribute_dict('name', 'units', 'initial_value'))
    pass


class ClampVariable(BaseGroupAction):
    # Leaving old XML method in to document existing properties / tokens.
    # def _xml(self):
    #    assert 1 <= len(self.tokens) <= 2
    #    name = self.tokens[0]
    #    if len(self.tokens) == 1:
    #        value = name
    #    else:
    #        value = self.tokens[1]
    #    return self.delegate('ModelEquation', [[name, value]]).xml()
    pass

# TODO: Rename to class DefineVariable ?


class ModelEquation(BaseGroupAction):
    """
    Parse action for ``define`` declarations in the model interface, that
    add or replace model variable's equations.

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

    def to_sympy(self, symbol_generator, number_generator):
        """Convert this equation to Sympy.

        Notes for later implementation:
        * ``number_generator`` can be the same as the CellML parser, except using the protocol's UnitStore:
            ``lambda x, y: model.add_number(x, model.get_units(y))``
        * ``symbol_generator`` will be different. It will be set up by our parent ModelInterface,
            and needs to dereference all the name lookups we might encounter. Prefixed names will
            need to use the ns_map to get a (URI, local_name) pair and do ``model.get_symbol_by_ontology_term``.
            Plain names will need to check they've been defined in a ``DeclareVariable`` stanza, which should
            maintain a map from name to ``VariableDummy``.
        """
        var = self.var.to_sympy(symbol_generator, number_generator)
        if self.is_ode:
            bvar = self.bvar.to_sympy(symbol_generator, number_generator)
            lhs = sympy.Derivative(var, bvar, evaluate=False)
        else:
            lhs = var
        rhs = self.rhs.to_sympy(symbol_generator, number_generator)
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
        Either an empty list, or a singleton list specifying the units for time.
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
    ``sympy_equations``
        Once :meth:`associate_model` and :meth:`resolve_namespaces` have been called, this property gives Sympy
        versions of ``self.equations``.
    """
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            # This is an empty instance not created by pyparsing. Fake the arguments pyparsing needs.
            args = ('', '', [[]])
        super().__init__(*args, **kwargs)
        self.time_units = []
        self.inputs = []
        self.outputs = []
        self.optional_decls = []
        self.equations = []
        self._sympy_equations = None
        self._model = None
        self._ns_map = None

    def _expr(self):
        actions = self.get_children_expr()
        self.time_units = [a for a in actions if isinstance(a, SetTimeUnits)]
        self.inputs = [a for a in actions if isinstance(a, InputVariable)]
        self.outputs = [a for a in actions if isinstance(a, OutputVariable)]
        self.optional_decls = [a for a in actions if isinstance(a, OptionalVariable)]
        self.equations = [a for a in actions if isinstance(a, ModelEquation)]

        # Some basic semantics checking
        if len(self.time_units) > 1:
            raise ValueError('The units for time cannot be set multiple times')
        return self

    def resolve_namespaces(self, ns_map):
        """Resolve namespace prefixes to full URIs for all parts of the interface.

        :param ns_map: mapping from NS prefix to URI.
        """
        self._ns_map = ns_map
        for item in itertools.chain(self.inputs, self.outputs, self.optional_decls):
            item.ns_uri = ns_map[item.ns_prefix]

    def associate_model(self, model):
        """Tell this interface what model it is being used to manipulate."""
        self._model = model

    def _symbol_generator(self, name):
        if ':' in name:
            prefix, local_name = name.split(':', 1)
            ns_uri = self._ns_map[prefix]
            return self._model.get_symbol_by_ontology_term(ns_uri, local_name)
        else:
            # DeclareVariable not yet done
            raise NotImplementedError

    def _number_generator(self, value, units):
        # TODO: Use the protocol's UnitStore instead!
        return self._model.add_number(value, self._model.get_units(units))

    @property
    def sympy_equations(self):
        """The equations defined by the interface in Sympy form.

        Requires :meth:`associate_model` to have been called.
        """
        if self._sympy_equations is None:
            # Do the transformation
            self._sympy_equations = eqs = []
            for eq in self.equations:
                eqs.append(eq.to_sympy(self._symbol_generator, self._number_generator))
        return self._sympy_equations


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
    """Parse action for unit references within units definitions."""

    def get_value(self, token, negate=False):
        """Get a decent string representation of the value of the given numeric token.
        It may be a plain number, or it may be a simple expression which we have to evaluate.
        """
        format = "%.17g"
        result = str(token).strip()
        try:
            value = float(result)
        except ValueError:
            # Evaluation required; somewhat risky!
            value = eval(result)
            if negate:
                value = -value
            result = format % value
        else:
            # Just use the string representation in the protocol
            if negate:
                if result[0] == '-':
                    result = result[1:]
                else:
                    result = '-' + result
        return result

    # Leaving old XML method in to document existing properties / tokens.
    # def _xml(self):
    #    attrs = self.get_attribute_dict('prefix', 'units', 'exponent')
    #    if 'multiplier' in self.tokens:
    #        attrs['multiplier'] = self.GetValue(self.tokens['multiplier'][0])
    #    if 'offset' in self.tokens:
    #        attrs['offset'] = self.GetValue(self.tokens['offset'][0][1], self.tokens['offset'][0][0] == '-')
    #    return CELLML.unit(**attrs)


class UnitsDef(BaseGroupAction):
    """Parse action for units definitions."""

    # Leaving old XML method in to document existing properties / tokens.
    # def _xml(self):
    #    name = str(self.tokens[0])
    #    if 'description' in self.tokens:
    #        units_map[name] = str(self.tokens['description'])
    #    unit_refs = [t.xml() for t in self.tokens if isinstance(t, UnitRef)]
    #    return CELLML.units(*unit_refs, name=name)


class Units(BaseAction):
    """Parse action for the units definitions section."""

    # Leaving old XML method in to document existing properties / tokens.
    # def _xml(self):
    #    if len(self.tokens) > 0:
    #        return P.units(*self.get_children_XML())


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
            if isinstance(token, Library):
                d['library'] = token.expr()
            if isinstance(token, PostProcessing):
                d['postprocessing'] = token.expr()
            if isinstance(token, Import):
                d['imports'].append(token.expr())
            if isinstance(token, Tasks):
                d['simulations'] = token.expr()
            if isinstance(token, Inputs):
                d['inputs'] = token.expr()
            if isinstance(token, Outputs):
                d['outputs'] = token.expr()
            if isinstance(token, Plots):
                d['plots'] = token.expr()
            if isinstance(token, ModelInterface):
                d['model_interface'] = token.expr()

        if 'dox' in self.tokens:
            d['dox'] = self.tokens['dox'][0]

        # Scan for any declared namespaces
        ns_map = {}
        if 'namespace' in self.tokens:
            for prefix, uri in self.tokens['namespace']:
                ns_map[prefix] = uri
        d['ns_map'] = ns_map

        return d

