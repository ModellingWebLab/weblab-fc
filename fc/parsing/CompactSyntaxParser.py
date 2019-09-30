
import math
import os
import sys

import pyparsing as p

import fc.language.expressions as E
import fc.language.statements as S
import fc.language.values as V
import fc.simulations.model as Model
import fc.simulations.modifiers as Modifiers
import fc.simulations.ranges as Ranges
import fc.simulations.simulations as Simulations
from fc.locatable import Locatable

__all__ = ['CompactSyntaxParser']

# Necessary for reasonable speed when using infixNotation
p.ParserElement.enablePackrat()


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
            """Create a csymbol parse action for producing part of our XML output."""
            if content is None:
                content = list()
            return self.delegate(Actions.Symbol(symbol), [content])

        def expr(self):
            """Updates location in parent locatable class and calls _expr method."""
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
        # Leaving old XML method in to document existing properties.
        # def _xml(self):
        #    return P.specifyInputVariable(**self.get_attribute_dict('name', 'units', 'initial_value'))
        pass

    class OutputVariable(BaseGroupAction):
        """
        Parse action for output variables defined in the model interface.
        """

        def _expr(self):
            ns, local_name = self.tokens['name'].split(':', 1)
            return {
                'type': 'OutputVariable',
                'ns': ns,
                'local_name': local_name,
                'unit': self.tokens.get('unit', None),
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

    class ModelEquation(BaseGroupAction):
        # Leaving old XML method in to document existing properties / tokens.
        # def _xml(self):
        #    assert len(self.tokens) == 2
        #    if isinstance(self.tokens[0], Actions.Variable):
        #        lhs = self.tokens[0].xml()
        #    else:
        #        # Assigning an ODE
        #        assert len(self.tokens[0]) == 2
        #        bvar = M.bvar(self.tokens[0][1].xml())
        #        lhs = self.AddLoc(M.apply(M.diff, bvar, self.tokens[0][0].xml()))
        #    rhs = self.tokens[1].xml()
        #    return P.addOrReplaceEquation(self.AddLoc(M.apply(M.eq, lhs, rhs)))
        pass

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
            # TODO: Create objects for all parts of the model interface
            #return self.get_children_expr()
            output = []
            for action in self:
                if isinstance(action, Actions.OutputVariable):
                    output.append(action.expr())
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


################################################################################
# Helper methods for defining parsers
################################################################################
def make_kw(keyword, suppress=True):
    """Helper function to create a parser for the given keyword."""
    kw = p.Keyword(keyword)
    if suppress:
        kw = kw.suppress()
    return kw


def adjacent(parser):
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

    def parseImpl(self, instring, loc, doActions=True):
        try:
            loc, tokens = self.expr._parse(instring, loc, doActions, callPreParse=True)
        except (p.ParseException, IndexError):
            if self.defaultValue is not self._optionalNotMatched:
                if self.expr.resultsName:
                    tokens = p.ParseResults([self.defaultValue])
                    tokens[self.expr.resultsName] = self.defaultValue
                else:
                    tokens = [self.defaultValue]
            else:
                tokens = []
        return loc, tokens


def optional_delimited_list(expr, delim):
    """Like delimitedList, but the list may be empty."""
    return p.delimitedList(expr, delim) | p.Empty()


def delimited_multi_list(elements, delimiter):
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
    rest = delimited_multi_list(elements[1:], delimiter)
    if unbounded:
        result = p.Forward()
        result << ((expr + delimiter + result) | expr | rest)
    else:
        if isinstance(rest, p.Empty):
            result = expr | rest
        else:
            result = (expr + delimiter + rest) | expr | rest
    return result


def unignore(parser):
    """Stop ignoring things in the given parser (and its children)."""
    for child in getattr(parser, 'exprs', []):
        unignore(child)
    if hasattr(parser, 'expr'):
        unignore(parser.expr)
    parser.ignoreExprs = []


def monkey_patch_pyparsing():
    """Monkey-patch some pyparsing methods to behave slightly differently."""

    def ignore(self, other):
        """Improved ignore that avoids ignoring self by accident."""
        if isinstance(other, p.Suppress):
            if other not in self.ignoreExprs and other != self:
                self.ignoreExprs.append(other.copy())
        else:
            self.ignoreExprs.append(p.Suppress(other.copy()))
        return self
    p.ParserElement.ignore = ignore

    def err_str(self):
        """Extended exception reporting that also prints the offending line with an error marker underneath."""
        return "%s (at char %d), (line:%d, col:%d):\n%s\n%s" % (self.msg, self.loc, self.lineno, self.column, self.line,
                                                                ' ' * (self.column - 1) + '^')
    p.ParseException.__str__ = err_str


monkey_patch_pyparsing()


class CompactSyntaxParser(object):
    """A parser for the compact textual syntax for protocols."""

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
    nl = p.Suppress(p.OneOrMore(Optional(comment) + p.LineEnd())
                    ).setName('Newline(s)')  # Any line can end with a comment
    obrace = (Optional(nl) + p.Suppress('{') + Optional(nl)).setName('{')
    cbrace = (Optional(nl) + p.Suppress('}') + Optional(nl)).setName('}')
    embedded_cbrace = (Optional(nl) + p.Suppress('}')).setName('}')

    # Identifiers
    nc_ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*').setName('ncIdent')
    c_ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*:[_a-zA-Z][_0-9a-zA-Z]*').setName('cIdent')
    ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*(:[_a-zA-Z][_0-9a-zA-Z]*)*').setName('Ident')
    nc_ident_as_var = nc_ident.copy().setParseAction(Actions.Variable)
    ident_as_var = ident.copy().setParseAction(Actions.Variable)

    # Numbers can be given in scientific notation, with an optional leading minus sign.
    # Within expressions they may also have units specified, e.g. in the model interface.
    units_ident = p.originalTextFor(p.Literal('units_of(') - adjacent(ident) + adjacent(p.Literal(')'))) | nc_ident
    units_annotation = p.Suppress('::') - units_ident("units")
    plain_number = p.Regex(r'-?[0-9]+((\.[0-9]+)?(e[-+]?[0-9]+)?)?').setName('Number')
    number = (plain_number + Optional(units_annotation)).setName('Number')

    # Used for descriptive text
    quoted_string = (p.QuotedString('"', escChar="\\") | p.QuotedString("'", escChar="\\")).setName('QuotedString')
    # This may become more specific in future
    quoted_uri = quoted_string.copy().setName('QuotedUri')

    # Expressions from the "post-processing" language
    #################################################

    # Expressions and statements must be constructed recursively
    expr = p.Forward().setName('Expression')
    stmt_list = p.Forward().setName('StatementList')

    # A vector written like 1:2:5 or 1:5 or A:B:C
    numeric_range = p.Group(expr + colon - expr + Optional(colon - expr))

    # Creating arrays
    dim_spec = Optional(expr + adjacent(dollar)) + nc_ident
    comprehension = p.Group(make_kw('for') - dim_spec + make_kw('in') - numeric_range).setParseAction(Actions.Comprehension)
    array = p.Group(osquare - expr + (p.OneOrMore(comprehension) | p.ZeroOrMore(comma - expr)) + csquare
                    ).setName('Array').setParseAction(Actions.Array)

    # Array views
    opt_expr = Optional(expr, default='')
    view_spec = p.Group(adjacent(osquare) - Optional(('*' | expr) + adjacent(dollar))('dimspec') +
                       opt_expr + Optional(colon - opt_expr + Optional(colon - opt_expr)) + csquare).setName('ViewSpec')

    # If-then-else
    if_expr = p.Group(make_kw('if') - expr + make_kw('then') - expr +
                     make_kw('else') - expr).setName('IfThenElse').setParseAction(Actions.Piecewise)

    # Lambda definitions
    param_decl = p.Group(nc_ident_as_var + Optional(eq + expr))
    param_list = p.Group(optional_delimited_list(param_decl, comma))
    lambda_expr = p.Group(make_kw('lambda') - param_list + ((colon - expr) | (obrace - stmt_list + embedded_cbrace))
                         ).setName('Lambda').setParseAction(Actions.Lambda)

    # Function calls
    # TODO: Allow lambdas, not just ident?
    arg_list = p.Group(optional_delimited_list(expr, comma))
    function_call = p.Group(ident_as_var + adjacent(oparen) - arg_list +
                           cparen).setName('FnCall').setParseAction(Actions.FunctionCall)

    # Tuples
    tuple = p.Group(oparen + expr + comma - optional_delimited_list(expr, comma) +
                    cparen).setName('Tuple').setParseAction(Actions.Tuple)

    # Accessors
    accessor = p.Combine(adjacent(p.Suppress('.')) -
                         p.oneOf('IS_SIMPLE_VALUE IS_ARRAY IS_STRING IS_TUPLE IS_FUNCTION IS_NULL IS_DEFAULT '
                                 'NUM_DIMS NUM_ELEMENTS SHAPE')).setName('Accessor')

    # Indexing
    pad = (make_kw('pad') + adjacent(colon) - expr + eq + expr).setResultsName('pad')
    shrink = (make_kw('shrink') + adjacent(colon) - expr).setResultsName('shrink')
    index_dim = expr.setResultsName('dim')
    index = p.Group(adjacent(p.Suppress('{')) - expr +
                    p.ZeroOrMore(comma - (pad | shrink | index_dim)) + p.Suppress('}')).setName('Index')

    # Special values
    null_value = p.Group(make_kw('null')).setName('Null').setParseAction(Actions.Symbol('null'))
    default_value = p.Group(make_kw('default')).setName('Default').setParseAction(Actions.Symbol('defaultParameter'))
    string_value = quoted_string.copy().setName('String').setParseAction(Actions.Symbol('string'))

    # Recognised MathML operators
    mathml_operators = set('''
        quotient rem max min root xor abs floor ceiling exp ln log
        sin cos tan
        sec csc cot
        sinh cosh tanh
        sech csch coth
        arcsin arccos arctan
        arccosh arccot arccoth
        arccsc arccsch arcsec
        arcsech arcsinh arctanh
        '''.split())

    # Wrapping MathML operators into lambdas
    mathml_operator = (
        p.oneOf('^ * / + - not == != <= >= < > && ||') |
        p.Combine('MathML:' + p.oneOf(' '.join(mathml_operators))))
    wrap = p.Group(
            p.Suppress('@') - adjacent(p.Word(p.nums)) + adjacent(colon) + mathml_operator
        ).setName('WrapMathML').setParseAction(Actions.Wrap)

    # Turning on tracing for debugging protocols
    trace = adjacent(p.Suppress('?'))

    # The main expression grammar.  Atoms are ordered according to rough speed of detecting mis-match.
    atom = (array | wrap | number.copy().setParseAction(Actions.Number) | string_value |
            if_expr | null_value | default_value | lambda_expr | function_call | ident_as_var | tuple).setName('Atom')
    expr <<= p.infixNotation(atom, [(accessor, 1, p.opAssoc.LEFT, Actions.Accessor),
                                    (view_spec, 1, p.opAssoc.LEFT, Actions.View),
                                    (index, 1, p.opAssoc.LEFT, Actions.Index),
                                    (trace, 1, p.opAssoc.LEFT, Actions.Trace),
                                    ('^', 2, p.opAssoc.LEFT, Actions.Operator),
                                    ('-', 1, p.opAssoc.RIGHT,
                                        lambda *args: Actions.Operator(*args, rightAssoc=True)),
                                    (p.oneOf('* /'), 2, p.opAssoc.LEFT, Actions.Operator),
                                    (p.oneOf('+ -'), 2, p.opAssoc.LEFT, Actions.Operator),
                                    (p.Keyword('not'), 1, p.opAssoc.RIGHT,
                                     lambda *args: Actions.Operator(*args, rightAssoc=True)),
                                    (p.oneOf('== != <= >= < >'), 2, p.opAssoc.LEFT, Actions.Operator),
                                    (p.oneOf('&& ||'), 2, p.opAssoc.LEFT, Actions.Operator)
                                    ])

    # Simpler expressions containing no arrays, functions, etc. Used in the model interface.
    simple_expr = p.Forward().setName('SimpleExpression')
    simple_if_expr = p.Group(make_kw('if') - simple_expr + make_kw('then') - simple_expr +
                           make_kw('else') - simple_expr).setName('SimpleIfThenElse').setParseAction(Actions.Piecewise)
    simple_arg_list = p.Group(optional_delimited_list(simple_expr, comma))
    simple_function_call = p.Group(ident_as_var + adjacent(oparen) - simple_arg_list +
                                 cparen).setName('SimpleFnCall').setParseAction(Actions.FunctionCall)
    simple_expr <<= p.infixNotation(number.copy().setParseAction(Actions.Number) |
                                   simple_if_expr | simple_function_call | ident_as_var,
                                   [('^', 2, p.opAssoc.LEFT, Actions.Operator),
                                    ('-', 1, p.opAssoc.RIGHT,
                                        lambda *args: Actions.Operator(*args, rightAssoc=True)),
                                    (p.oneOf('* /'), 2, p.opAssoc.LEFT, Actions.Operator),
                                    (p.oneOf('+ -'), 2, p.opAssoc.LEFT, Actions.Operator),
                                    (p.Keyword('not'), 1, p.opAssoc.RIGHT,
                                     lambda *args: Actions.Operator(*args, rightAssoc=True)),
                                    (p.oneOf('== != <= >= < >'), 2, p.opAssoc.LEFT, Actions.Operator),
                                    (p.oneOf('&& ||'), 2, p.opAssoc.LEFT, Actions.Operator)
                                    ])
    simple_param_list = p.Group(optional_delimited_list(p.Group(nc_ident_as_var), comma))
    simple_lambda_expr = p.Group(make_kw('lambda') - simple_param_list + colon -
                               expr).setName('SimpleLambda').setParseAction(Actions.Lambda)

    # Newlines in expressions may be escaped with a backslash
    expr.ignore('\\' + p.LineEnd())
    simple_expr.ignore('\\' + p.LineEnd())
    # Bare newlines are OK provided we started with a bracket.
    # However, it's quite hard to enforce that restriction.
    expr.ignore(p.Literal('\n'))
    simple_expr.ignore(p.Literal('\n'))
    # Embedded comments are also OK
    expr.ignore(comment)
    simple_expr.ignore(comment)
    # Avoid mayhem
    unignore(nl)

    # Statements from the "post-processing" language
    ################################################

    # Simple assignment (i.e. not to a tuple)
    simple_assign = p.Group(nc_ident_as_var + eq - expr).setName('SimpleAssign').setParseAction(Actions.Assignment)
    simple_assign_list = p.Group(optional_delimited_list(simple_assign, nl)).setParseAction(Actions.StatementList)

    # Assertions and function returns
    assert_stmt = p.Group(make_kw('assert') - expr).setName('AssertStmt').setParseAction(Actions.Assert)
    return_stmt = p.Group(make_kw('return') - p.delimitedList(expr)).setName('ReturnStmt').setParseAction(Actions.Return)

    # Full assignment, to a tuple of names or single name
    _idents = p.Group(p.delimitedList(nc_ident_as_var)).setParseAction(Actions.MaybeTuple)
    assign_stmt = p.Group(((make_kw('optional', suppress=False)("optional") + _idents) | _idents) + eq -
                         p.Group(p.delimitedList(expr)).setParseAction(Actions.MaybeTuple))   \
        .setName('AssignStmt').setParseAction(Actions.Assignment)

    # Function definition
    function_defn = p.Group(make_kw('def') - nc_ident_as_var + oparen + param_list + cparen -
                           ((colon - expr) | (obrace - stmt_list + Optional(nl) + p.Suppress('}')))
                           ).setName('FunctionDef').setParseAction(Actions.FunctionDef)

    stmt_list << p.Group(p.delimitedList(assert_stmt | return_stmt | function_defn | assign_stmt, nl))
    stmt_list.setParseAction(Actions.StatementList)

    # Miscellaneous constructs making up protocols
    ##############################################

    # Documentation (Markdown)
    documentation = p.Group(make_kw('documentation') - obrace - p.Regex("[^}]*") + cbrace)("dox")

    # Namespace declarations
    ns_decl = p.Group(make_kw('namespace') - nc_ident("prefix") + eq + quoted_uri("uri")).setName('NamespaceDecl')
    ns_decls = optional_delimited_list(ns_decl("namespace*"), nl)

    # Protocol input declarations, with default values
    inputs = (make_kw('inputs') - obrace - simple_assign_list + cbrace).setName('Inputs').setParseAction(Actions.Inputs)

    # Import statements
    import_stmt = p.Group(
        make_kw('import') -
        Optional(
            nc_ident +
            eq,
            default='') +
        quoted_uri +
        Optional(
            obrace -
            simple_assign_list +
            embedded_cbrace)).setName('Import').setParseAction(
                Actions.Import)
    imports = optional_delimited_list(import_stmt, nl).setName('Imports')

    # Library, globals defined using post-processing language.
    # Strictly speaking returns aren't allowed, but that gets picked up later.
    library = (make_kw('library') - obrace - Optional(stmt_list) +
               cbrace).setName('Library').setParseAction(Actions.Library)

    # Post-processing
    post_processing = (make_kw('post-processing') + obrace -
                      optional_delimited_list(assert_stmt | return_stmt | function_defn | assign_stmt, nl) +
                      cbrace).setName('PostProc').setParseAction(Actions.PostProcessing)

    # Units definitions
    si_prefix = p.oneOf('deka hecto kilo mega giga tera peta exa zetta yotta'
                       'deci centi milli micro nano pico femto atto zepto yocto')
    _num_or_expr = p.originalTextFor(plain_number | (oparen + expr + cparen))
    unit_ref = p.Group(Optional(_num_or_expr)("multiplier") + Optional(si_prefix)("prefix") + nc_ident("units") +
                      Optional(p.Suppress('^') + plain_number)("exponent") +
                      Optional(p.Group(p.oneOf('- +') + _num_or_expr))("offset")).setParseAction(Actions.UnitRef)
    units_def = p.Group(nc_ident + eq + p.delimitedList(unit_ref, '.') + Optional(quoted_string)("description")
                       ).setName('UnitsDefinition').setParseAction(Actions.UnitsDef)
    units = (make_kw('units') - obrace - optional_delimited_list(units_def, nl) + cbrace
             ).setName('Units').setParseAction(Actions.Units)

    # Model interface section
    #########################
    units_ref = make_kw('units') - nc_ident

    # Setting the units for the independent variable
    set_time_units = (make_kw('independent') - make_kw('var') - units_ref("units")).setParseAction(Actions.SetTimeUnits)
    # Input variables, with optional units and initial value
    input_variable = p.Group(
        make_kw('input') -
        c_ident("name") +
        Optional(units_ref)("units") +
        Optional(
            eq +
            plain_number)("initial_value")).setName('InputVariable').setParseAction(
        Actions.InputVariable)
    # Model outputs of interest, with optional units
    output_variable = p.Group(make_kw('output') - c_ident("name") + Optional(units_ref("units"))
                             ).setName('OutputVariable').setParseAction(Actions.OutputVariable)
    # Model variables (inputs, outputs, or just used in equations) that are allowed to be missing
    locator = p.Empty().leaveWhitespace().setParseAction(lambda s, l, t: l)
    var_default = make_kw('default') - locator("default_start") + simple_expr("default")
    optional_variable = p.Group(make_kw('optional') - c_ident("name") + Optional(var_default) + locator("default_end")
                               ).setName('OptionalVar').setParseAction(Actions.OptionalVariable)
    # New variables added to the model, with optional initial value
    new_variable = p.Group(
        make_kw('var') -
        nc_ident("name") +
        units_ref("units") +
        Optional(
            eq +
            plain_number)("initial_value")).setName('NewVariable').setParseAction(
        Actions.DeclareVariable)
    # Adding or replacing equations in the model
    clamp_variable = p.Group(make_kw('clamp') - ident_as_var + Optional(make_kw('to') - simple_expr)
                            ).setName('ClampVariable').setParseAction(Actions.ClampVariable)
    interpolate = p.Group(
        make_kw('interpolate') -
        oparen -
        quoted_string -
        comma -
        ident_as_var -
        comma -
        nc_ident -
        comma -
        nc_ident -
        cparen).setName('Interpolate').setParseAction(
        Actions.Interpolate)
    model_equation = p.Group(make_kw('define') -
                            (p.Group(make_kw('diff') +
                                     adjacent(oparen) -
                                     ident_as_var +
                                     p.Suppress(';') +
                                     ident_as_var +
                                     cparen) | ident_as_var) +
                            eq +
                            (interpolate | simple_expr)
                            ).setName('AddOrReplaceEquation').setParseAction(Actions.ModelEquation)
    # Units conversion rules
    units_conversion = p.Group(
        make_kw('convert') -
        nc_ident("actualDimensions") +
        make_kw('to') +
        nc_ident("desiredDimensions") +
        make_kw('by') -
        simple_lambda_expr).setName('UnitsConversion').setParseAction(
        Actions.UnitsConversion)

    model_interface = p.Group(make_kw('model') - make_kw('interface') - obrace -
                             Optional(set_time_units - nl) +
                             optional_delimited_list((input_variable | output_variable | optional_variable | new_variable |
                                                    clamp_variable | model_equation | units_conversion), nl) +
                             cbrace).setName('ModelInterface').setParseAction(Actions.ModelInterface)

    # Simulation definitions
    ########################

    # Ranges
    uniform_range = make_kw('uniform') + numeric_range
    vector_range = make_kw('vector') + expr
    while_range = make_kw('while') + expr
    range = p.Group(make_kw('range') + nc_ident("name") + units_ref("units") +
                    (uniform_range("uniform") | vector_range("vector") | while_range("while"))
                    ).setName('Range').setParseAction(Actions.Range)

    # Modifiers
    modifier_when = make_kw('at') - (make_kw('start', False) |
                                   (make_kw('each', False) - make_kw('loop')) |
                                   make_kw('end', False)).setParseAction(Actions.ModifierWhen)
    set_variable = make_kw('set') - ident + eq + expr
    save_state = make_kw('save') - make_kw('as') - nc_ident
    reset_state = make_kw('reset') - Optional(make_kw('to') + nc_ident)
    modifier = p.Group(modifier_when + p.Group(set_variable("set") | save_state("save") | reset_state("reset"))
                       ).setName('Modifier').setParseAction(Actions.Modifier)
    modifiers = p.Group(make_kw('modifiers') + obrace - optional_delimited_list(modifier, nl) + cbrace
                        ).setName('Modifiers').setParseAction(Actions.Modifiers)

    # The simulations themselves
    simulation = p.Forward().setName('Simulation')
    _select_output = p.Group(make_kw('select') - Optional(make_kw('optional', suppress=False)) -
                            make_kw('output') - nc_ident).setName('SelectOutput')
    nested_protocol = p.Group(make_kw('protocol') - quoted_uri + obrace +
                             simple_assign_list + Optional(nl) + optional_delimited_list(_select_output, nl) +
                             cbrace + Optional('?')).setName('NestedProtocol').setParseAction(Actions.NestedProtocol)
    timecourse_sim = p.Group(make_kw('timecourse') - obrace - range + Optional(nl + modifiers) + cbrace
                            ).setName('TimecourseSim').setParseAction(Actions.TimecourseSimulation)
    nested_sim = p.Group(make_kw('nested') - obrace - range + nl + Optional(modifiers) +
                        p.Group(make_kw('nests') + (simulation | nested_protocol | ident)) +
                        cbrace).setName('NestedSim').setParseAction(Actions.NestedSimulation)
    one_step_sim = p.Group(make_kw('oneStep') - Optional(p.originalTextFor(expr))("step") +
                         Optional(obrace - modifiers + cbrace)("modifiers")).setParseAction(Actions.OneStepSimulation)
    simulation << p.Group(make_kw('simulation') - Optional(nc_ident + eq, default='') +
                          (timecourse_sim | nested_sim | one_step_sim) -
                          Optional('?' + nl)).setParseAction(Actions.Simulation)

    tasks = p.Group(make_kw('tasks') + obrace - p.ZeroOrMore(simulation) +
                    cbrace).setName('Tasks').setParseAction(Actions.Tasks)

    # Output specifications
    #######################

    output_desc = Optional(quoted_string)("description")
    output_spec = p.Group(Optional(make_kw('optional', suppress=False))("optional") +
                         nc_ident("name") +
                         ((units_ref("units") +
                           output_desc) | (eq +
                                          ident("ref") +
                                          Optional(units_ref)("units") +
                                          output_desc))).setName('Output').setParseAction(Actions.Output)
    outputs = p.Group(make_kw('outputs') + obrace - optional_delimited_list(output_spec, nl) +
                      cbrace).setName('Outputs').setParseAction(Actions.Outputs)

    # Plot specifications
    #####################

    plot_curve = p.Group(p.delimitedList(nc_ident, ',') +
                        make_kw('against') - nc_ident +
                        Optional(make_kw('key') - nc_ident("key"))).setName('Curve')
    plot_using = (make_kw('using') - (make_kw('lines', suppress=False) |
                                    make_kw('points', suppress=False) |
                                    make_kw('linespoints', suppress=False)))("using")
    plot_spec = p.Group(make_kw('plot') - quoted_string + Optional(plot_using) - obrace +
                       plot_curve + p.ZeroOrMore(nl + plot_curve) + cbrace).setName('Plot').setParseAction(Actions.Plot)
    plots = p.Group(make_kw('plots') + obrace - p.ZeroOrMore(plot_spec) +
                    cbrace).setName('Plots').setParseAction(Actions.Plots)

    # Parsing a full protocol
    #########################

    protocol = p.And(
        list(map(Optional, [
            nl,
            documentation,
            ns_decls + nl,
            inputs,
            imports + nl,
            library,
            units,
            model_interface,
            tasks,
            post_processing,
            outputs,
            plots,
        ]))).setName('Protocol').setParseAction(Actions.Protocol)

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

    def try_parse(self, callable, *args, **kwargs):
        """
        Try calling the given parse command, increasing the stack depth limit
        if needed.
        """
        r = None  # Result
        while self._stack_depth_factor < 3:
            try:
                r = callable(*args, **kwargs)
            except RuntimeError as msg:
                print('Got RuntimeError:', msg, file=sys.stderr)
                self._stack_depth_factor += 0.5
                new_limit = int(
                    self._stack_depth_factor * self._original_stack_limit)
                print('Increasing recursion limit to', new_limit,
                      file=sys.stderr)
                sys.setrecursionlimit(new_limit)
            else:
                break  # Parsed OK
        if not r:
            raise RuntimeError("Failed to parse expression even with a recursion limit of %d; giving up!"
                               % (int(self._stack_depth_factor * self._original_stack_limit),))
        return r

################################################################################
# Parser debugging support
################################################################################

def get_named_grammars(obj=CompactSyntaxParser):
    """Get a list of all the named grammars in the given object."""
    grammars = []
    for parser in dir(obj):
        parser = getattr(obj, parser)
        if isinstance(parser, p.ParserElement):
            grammars.append(parser)
    return grammars


def enable_debug(grammars=None):
    """Enable debugging of our (named) grammars."""
    def display_loc(instring, loc):
        return " at loc " + str(loc) + "(%d,%d)" % (p.lineno(loc, instring), p.col(loc, instring))

    def success_debug_action(instring, startloc, endloc, expr, toks):
        print("Matched " + str(expr) + " -> " + str(toks.asList()) + display_loc(instring, endloc))

    def exception_debug_action(instring, loc, expr, exc):
        print("Exception raised:" + str(exc) + display_loc(instring, loc))

    for parser in grammars or get_named_grammars():
        parser.setDebugActions(None, success_debug_action, exception_debug_action)


def disable_debug(grammars=None):
    """Stop debugging our (named) grammars."""
    for parser in grammars or get_named_grammars():
        parser.setDebug(False)


class Debug(object):
    """A Python 2.6+ context manager that enables debugging just for the enclosed block."""

    def __init__(self, grammars=None):
        self._grammars = list(grammars or get_named_grammars())

    def __enter__(self):
        enable_debug(self._grammars)

    def __exit__(self, type, value, traceback):
        disable_debug(self._grammars)

