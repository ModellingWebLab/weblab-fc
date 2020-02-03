
import math
import os

import pyparsing as p

import fc.language.expressions as E
import fc.language.statements as S
import fc.language.values as V
import fc.simulations.model as Model
import fc.simulations.modifiers as Modifiers
import fc.simulations.ranges as Ranges
import fc.simulations.simulations as Simulations
from fc.locatable import Locatable

###############################################################
# Parse actions that can generate Python implementation objects
###############################################################


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


class Actions(object):
    """Container for parse actions."""
    source_file = ""  # Should be filled in by main parse method
    units_map = {}    # Will be cleared by main parse method

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
                    Actions.source_file, p.lineno(loc, s), p.col(loc, s), p.line(loc, s))

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

        def get_attribute_dict(self, *attrNames):
            """Create an attribute dictionary from named parse results."""
            attrs = {}
            for key in attrNames:
                if key in self.tokens:
                    value = self.tokens[key]
                    if not isinstance(value, str):
                        value = value[0]
                    attrs[key] = value
            return attrs

        def delegate(self, action, tokens):
            """Create another parse action to process the given tokens for us."""
            if isinstance(action, str):
                action = getattr(Actions, action)
            return action('', self.source_location, tokens)

        def delegate_symbol(self, symbol, content=None):
            """
            Create a Symbol parse action, used e.g. for the ``null`` symbol in the protocol
            language.
            """
            if content is None:
                content = list()
            return self.delegate(Actions.Symbol(symbol), [content])

        def expr(self):
            """Updates location in parent locatable class and calls :meth:`_expr()`."""
            result = self._expr()
            if isinstance(result, Locatable):
                result.location = self.source_location
            return result

    class BaseGroupAction(BaseAction):
        """Base class for parse actions associated with a Group.
        This strips the extra nesting level in its __init__.
        """

        def __init__(self, s, loc, tokens):
            super(Actions.BaseGroupAction, self).__init__(s, loc, tokens[0])

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
            super(Actions.Number, self).__init__(s, loc, tokens)
            if len(tokens) == 2:
                # We have a units annotation
                self._units = str(tokens[1])
            else:
                self._units = None

        def _expr(self):
            return E.Const(V.Simple(self.tokens))

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

    class Operator(BaseGroupAction):
        """Parse action for most MathML operators that are represented as operators in the syntax."""

        def __init__(self, *args, **kwargs):
            super(Actions.Operator, self).__init__(*args)
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
            assert isinstance(self.tokens[0], Actions.Variable)
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
            super(Actions._Symbol, self).__init__(s, loc, tokens)
            self.symbol = symbol

        def _expr(self):
            if self.symbol == "null":
                return E.Const(V.Null())
            elif self.symbol == "defaultParameter":
                return E.Const(V.DefaultParameter())
            if isinstance(self.tokens, str):
                return E.Const(V.String(self.tokens))

    @staticmethod
    def Symbol(symbol):
        """Wrapper around the _Symbol class."""
        def parse_action(s, loc, tokens):
            return Actions._Symbol(s, loc, tokens, symbol)
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
            if len(entries) > 1 and isinstance(self.tokens[1], Actions.Comprehension):
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
            super(Actions.Assignment, self).__init__(s, loc, tokens)
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
        # Leaving old XML method in to document existing properties.
        # def _xml(self):
        #     return P.setIndependentVariableUnits(**self.get_attribute_dict('units'))
        pass

    class InputVariable(BaseGroupAction):
        """
        Parse action for input variables defined in the model interface.
        """
        def _expr(self):
            # TODO: Will this always return 2 parts?
            ns, local_name = self.tokens['name'].split(':', 1)
            return {
                'type': 'InputVariable',
                'ns': ns,
                'local_name': local_name,
                'units': self.tokens.get('units', None),
                'initial_value': self.tokens.get('initial_value', None),
            }

    class OutputVariable(BaseGroupAction):
        """
        Parse action for output variables defined in the model interface.
        """
        def _expr(self):
            # TODO: Will this always return 2 parts?
            ns, local_name = self.tokens['name'].split(':', 1)
            return {
                'type': 'OutputVariable',
                'ns': ns,
                'local_name': local_name,
                'units': self.tokens.get('units', None),
            }

    class OptionalVariable(BaseGroupAction):
        def __init__(self, s, loc, tokens):
            super(Actions.OptionalVariable, self).__init__(s, loc, tokens)
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
        """
        def _expr(self):

            # Parse LHS
            var = None
            bvar = None
            if isinstance(self.tokens[0], Actions.Variable):
                # Assigning a normal variable
                assert len(self.tokens[0]) == 1
                var = self.tokens[0].names()[0]
            else:
                # Assigning an ODE
                assert len(self.tokens[0]) == 2
                var = self.tokens[0][0].names()[0]
                bvar = self.tokens[0][1].names()[0]

            # Store (don't parse) RHS
            rhs = self.tokens[1]

            return {
                'type': 'ModelEquation',
                'var': var,
                'bvar': bvar,
                'ode': bvar is not None,
                'rhs': rhs,
            }

    class Interpolate(BaseGroupAction):
        # Leaving old XML method in to document existing properties / tokens.
        # def _xml(self):
        #    assert len(self.tokens) == 4
        #    assert isinstance(self.tokens[0], str)
        #    file_path = self.delegate_symbol('string', self.tokens[0]).xml()
        #    assert isinstance(self.tokens[1], Actions.Variable)
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

        def _expr(self):
            #return self.get_children_expr()
            output = []
            handled = (
                Actions.OutputVariable,
                Actions.InputVariable,
                Actions.ModelEquation,
            )
            for action in self:
                if isinstance(action, handled):
                    output.append(action.expr())
                # TODO: Create objects for all parts of the model interface
            return output

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
                range_ = Ranges.UniformRange(attrs['name'], start, stop, step)
            elif 'vector' in self.tokens:
                expr = self.tokens['vector'][0].expr()
                range_ = Ranges.VectorRange(attrs['name'], expr)
            elif 'while' in self.tokens:
                cond = self.tokens['while'][0].expr()
                range_ = Ranges.While(attrs['name'], cond)
            return range_

    class ModifierWhen(BaseGroupAction):
        """Parse action for the when part of modifiers."""

        def _expr(self):
            when = {'start': 'START_ONLY', 'each': 'EACH_LOOP', 'end': 'END_ONLY'}[self.tokens]
            return getattr(Modifiers.AbstractModifier, when)

    class Modifier(BaseGroupAction):
        """Parse action that generates all kinds of modifier."""

        def _expr(self):
            args = [self.tokens[0].expr()]
            detail = self.tokens[1]
            if 'set' in self.tokens[1]:
                modifier = Modifiers.SetVariable
                args.append(detail[0])
                args.append(detail[1].expr())
            elif 'save' in self.tokens[1]:
                modifier = Modifiers.SaveState
                args.append(detail[0])
            elif 'reset' in self.tokens[1]:
                modifier = Modifiers.ResetState
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
            return Simulations.Timecourse(*args)

    class NestedSimulation(BaseGroupAction):

        def _expr(self):
            args = [t.expr() for t in self.tokens[0:-1]]
            if len(args) == 1:
                # Add an empty modifiers element
                args.append(self.delegate('Modifiers', [[]]).expr())
            nested = self.tokens[-1][0]
            if isinstance(nested, (Actions.Simulation, Actions.NestedProtocol)):
                # Inline definition
                args.append(nested.expr())
            return Simulations.Nested(args[2], args[0], args[1])

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
            super(Actions.NestedProtocol, self).__init__(s, loc, tokens)

        def _expr(self):
            args = []
            proto_file = self.tokens[0]
            import os
            proto_file = os.path.join(os.path.dirname(Actions.source_file), proto_file)
            args.append(proto_file)
            inputs = {}
            assert isinstance(self.tokens[1], Actions.StatementList)
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
            model = Model.NestedProtocol(*args)
            result = Simulations.OneStep(0)
            result.set_model(model)
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
        #        Actions.units_map[name] = str(self.tokens['description'])
        #    unit_refs = [t.xml() for t in self.tokens if isinstance(t, Actions.UnitRef)]
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
                if isinstance(token, Actions.Library):
                    d['library'] = token.expr()
                if isinstance(token, Actions.PostProcessing):
                    d['postprocessing'] = token.expr()
                if isinstance(token, Actions.Import):
                    d['imports'].append(token.expr())
                if isinstance(token, Actions.Tasks):
                    d['simulations'] = token.expr()
                if isinstance(token, Actions.Inputs):
                    d['inputs'] = token.expr()
                if isinstance(token, Actions.Outputs):
                    d['outputs'] = token.expr()
                if isinstance(token, Actions.Plots):
                    d['plots'] = token.expr()
                if isinstance(token, Actions.ModelInterface):
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

