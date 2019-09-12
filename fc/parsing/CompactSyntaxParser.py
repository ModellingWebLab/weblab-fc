
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

        def GetChildrenExpr(self):
            """Convert all sub-tokens to expr and return the list of elements."""
            return [tok.expr() for tok in self.tokens]

        def GetAttributeDict(self, *attrNames):
            """Create an attribute dictionary from named parse results."""
            attrs = {}
            for key in attrNames:
                if key in self.tokens:
                    value = self.tokens[key]
                    if not isinstance(value, str):
                        value = value[0]
                    attrs[key] = value
            return attrs

        def Delegate(self, action, tokens):
            """Create another parse action to process the given tokens for us."""
            if isinstance(action, str):
                action = getattr(Actions, action)
            return action('', self.source_location, tokens)

        def DelegateSymbol(self, symbol, content=None):
            """Create a csymbol parse action for producing part of our XML output."""
            if content is None:
                content = list()
            return self.Delegate(Actions.Symbol(symbol), [content])

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
        # Map from operator symbols used to MathML element names
        OP_MAP = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'divide', '^': 'power',
                  '==': 'eq', '!=': 'neq', '<': 'lt', '>': 'gt', '<=': 'leq', '>=': 'geq',
                  'not': 'not', '&&': 'and', '||': 'or'}

        def __init__(self, *args, **kwargs):
            super(Actions.Operator, self).__init__(*args)
            self.rightAssoc = kwargs.get('rightAssoc', False)

        def OperatorOperands(self):
            """Generator over (operator, operand) pairs."""
            it = iter(self.tokens[1:])
            while 1:
                try:
                    operator = next(it)
                    operand = next(it)
                    yield (operator, operand)
                except StopIteration:
                    break

        def Operator(self, operator):
            """Get the MathML element for the given operator."""
            return getattr(M, self.OP_MAP[operator])

        def _expr(self):
            if self.rightAssoc:
                # The only right-associative operators are also unary
                result = self.tokens[-1].expr()
                for operator in self.tokens[-2:-1:]:
                    result = OPERATORS[operator](result)
            else:
                result = self.tokens[0].expr()
                for operator, operand in self.OperatorOperands():
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
            return E.LambdaExpression.Wrap(operator, num_operands)

    class Piecewise(BaseGroupAction):
        """Parse action for if-then-else."""

        def _expr(self):
            if_, then_, else_ = self.GetChildrenExpr()
            return E.If(if_, then_, else_)

    class MaybeTuple(BaseGroupAction):
        """Parse action for elements that may be grouped into a tuple, or might be a single item."""

        def _expr(self):
            assert len(self.tokens) > 0
            if len(self.tokens) > 1:
                # Tuple
                return self.Delegate('Tuple', [self.tokens]).expr()
            else:
                # Single item
                return self.tokens[0].expr()  # should be list containing names

        def names(self):
            return list(map(str, self.tokens))

    class Tuple(BaseGroupAction):
        """Parse action for tuples."""

        def _expr(self):
            child_expr = self.GetChildrenExpr()
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
                return self.Delegate(
                    'Accessor', [[self.Delegate('Accessor', [self.tokens[:-1]]), self.tokens[-1]]]).expr()
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
                range = [range[0], self.Delegate('Number', ['1']), range[-1]]
            parts.extend(range)
            parts.append(self.DelegateSymbol('string', self.tokens[-2]))  # The variable name
            return self.Delegate('Tuple', [parts]).expr()

    class Array(BaseGroupAction):
        """Parse action for creating arrays."""

        def _expr(self):
            entries = self.GetChildrenExpr()
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
            null_token = self.DelegateSymbol('null')
            for viewspec in self.tokens[1:]:
                tuple_tokens = []
                dimspec = None
                if 'dimspec' in viewspec:
                    dimspec = viewspec['dimspec'][0]
                    viewspec = viewspec[1:]
                tuple_tokens.extend(viewspec)
                if dimspec is not None:
                    if len(tuple_tokens) == 1:
                        real_tuple_tokens = [dimspec, tuple_tokens[0], self.Delegate('Number', ['0']), tuple_tokens[0]]
                    elif len(tuple_tokens) == 2:
                        real_tuple_tokens = [dimspec, tuple_tokens[0], self.Delegate('Number', ['1']), tuple_tokens[1]]
                    else:
                        real_tuple_tokens = [dimspec, tuple_tokens[0], tuple_tokens[1], tuple_tokens[2]]
                else:
                    real_tuple_tokens = tuple_tokens
                # Replace unspecified elements with csymbol-null
                for i, token in enumerate(real_tuple_tokens):
                    if token == '' or token == '*':
                        real_tuple_tokens[i] = null_token
                args.append(self.Delegate('Tuple', [real_tuple_tokens]).expr())
            return E.View(*args)

    class Index(BaseGroupAction):
        """Parse action for index expressions."""

        def _expr(self):
            assert len(self.tokens) == 2
            index_tokens = self.tokens[1]
            assert 1 <= len(index_tokens)
            args = [self.tokens[0], index_tokens[0]]
            args.append(index_tokens.get('dim', self.DelegateSymbol('defaultParameter')))
            args.append(index_tokens.get('shrink', [self.DelegateSymbol('defaultParameter')])[0])
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
            assignee, value = self.GetChildrenExpr()
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
            return S.Return(*self.GetChildrenExpr())

    class Assert(BaseGroupAction):
        """Parse action for assert statements."""

        def _expr(self):
            return S.Assert(*self.GetChildrenExpr())

    class FunctionDef(BaseGroupAction):
        """Parse action for function definitions, which are sugar for assignment of a lambda."""

        def _expr(self):
            assert len(self.tokens) == 3
            lambda_ = self.Delegate('Lambda', [self.tokens[1:]])
            assign = self.Delegate('Assignment', [[self.tokens[0], lambda_]])
            return assign.expr()

    class StatementList(BaseGroupAction):
        """Parse action for lists of post-processing language statements."""

        def _expr(self):
            statements = self.GetChildrenExpr()
            return statements

    ######################################################################
    # Model interface section
    ######################################################################

    class SetTimeUnits(BaseAction):
        # Leaving old XML method in to document existing properties.
        # def _xml(self):
        #     return P.setIndependentVariableUnits(**self.GetAttributeDict('units'))
        pass

    class InputVariable(BaseGroupAction):
        # Leaving old XML method in to document existing properties.
        # def _xml(self):
        #    return P.specifyInputVariable(**self.GetAttributeDict('name', 'units', 'initial_value'))
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
        #    return P.declareNewVariable(**self.GetAttributeDict('name', 'units', 'initial_value'))
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
        #    return self.Delegate('ModelEquation', [[name, value]]).xml()
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
        #    file_path = self.DelegateSymbol('string', self.tokens[0]).xml()
        #    assert isinstance(self.tokens[1], Actions.Variable)
        #    indep_var = self.tokens[1].xml()
        #    units = []
        #    for i in [2, 3]:
        #        assert isinstance(self.tokens[i], str)
        #        units.append(M.ci(self.tokens[i]))
        #    return M.apply(self.DelegateSymbol('interpolate').xml(), file_path, indep_var, *units)
        pass

    class UnitsConversion(BaseGroupAction):
        # Leaving old XML method in to document existing properties / tokens.
        # def _xml(self):
        #    attrs = self.GetAttributeDict('desiredDimensions', 'actualDimensions')
        #    rule = self.tokens[-1].xml()
        #    return P.unitsConversionRule(rule, **attrs)
        pass

    class ModelInterface(BaseGroupAction):

        def _expr(self):
            # TODO: Create objects for all parts of the model interface
            #return self.GetChildrenExpr()
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
            attrs = self.GetAttributeDict('name', 'units')
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
            return self.GetChildrenExpr()

    class TimecourseSimulation(BaseGroupAction):

        def _expr(self):
            args = self.GetChildrenExpr()
            return Simulations.Timecourse(*args)

    class NestedSimulation(BaseGroupAction):

        def _expr(self):
            args = [t.expr() for t in self.tokens[0:-1]]
            if len(args) == 1:
                # Add an empty modifiers element
                args.append(self.Delegate('Modifiers', [[]]).expr())
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
            sims = self.GetChildrenExpr()
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
        #    attrs = self.GetAttributeDict('prefix', 'units', 'exponent')
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
        #        return P.units(*self.GetChildrenXml())

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
                return self.Delegate('StatementList', [self.tokens]).expr()

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
            return self.GetChildrenExpr()

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
            return self.GetChildrenExpr()

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
    """Stop ignoring things in the given parser (and its children)."""
    for child in getattr(parser, 'exprs', []):
        UnIgnore(child)
    if hasattr(parser, 'expr'):
        UnIgnore(parser.expr)
    parser.ignoreExprs = []


def MonkeyPatch():
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


MonkeyPatch()


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
    ncIdent = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*').setName('ncIdent')
    cIdent = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*:[_a-zA-Z][_0-9a-zA-Z]*').setName('cIdent')
    ident = p.Regex('[_a-zA-Z][_0-9a-zA-Z]*(:[_a-zA-Z][_0-9a-zA-Z]*)*').setName('Ident')
    ncIdentAsVar = ncIdent.copy().setParseAction(Actions.Variable)
    identAsVar = ident.copy().setParseAction(Actions.Variable)

    # Numbers can be given in scientific notation, with an optional leading minus sign.
    # Within expressions they may also have units specified, e.g. in the model interface.
    unitsIdent = p.originalTextFor(p.Literal('units_of(') - Adjacent(ident) + Adjacent(p.Literal(')'))) | ncIdent
    unitsAnnotation = p.Suppress('::') - unitsIdent("units")
    plainNumber = p.Regex(r'-?[0-9]+((\.[0-9]+)?(e[-+]?[0-9]+)?)?').setName('Number')
    number = (plainNumber + Optional(unitsAnnotation)).setName('Number')

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
    numericRange = p.Group(expr + colon - expr + Optional(colon - expr))

    # Creating arrays
    dimSpec = Optional(expr + Adjacent(dollar)) + ncIdent
    comprehension = p.Group(MakeKw('for') - dimSpec + MakeKw('in') - numericRange).setParseAction(Actions.Comprehension)
    array = p.Group(osquare - expr + (p.OneOrMore(comprehension) | p.ZeroOrMore(comma - expr)) + csquare
                    ).setName('Array').setParseAction(Actions.Array)

    # Array views
    optExpr = Optional(expr, default='')
    viewSpec = p.Group(Adjacent(osquare) - Optional(('*' | expr) + Adjacent(dollar))('dimspec') +
                       optExpr + Optional(colon - optExpr + Optional(colon - optExpr)) + csquare).setName('ViewSpec')

    # If-then-else
    ifExpr = p.Group(MakeKw('if') - expr + MakeKw('then') - expr +
                     MakeKw('else') - expr).setName('IfThenElse').setParseAction(Actions.Piecewise)

    # Lambda definitions
    paramDecl = p.Group(ncIdentAsVar + Optional(eq + expr))
    paramList = p.Group(OptionalDelimitedList(paramDecl, comma))
    lambdaExpr = p.Group(MakeKw('lambda') - paramList + ((colon - expr) | (obrace - stmtList + embedded_cbrace))
                         ).setName('Lambda').setParseAction(Actions.Lambda)

    # Function calls
    # TODO: Allow lambdas, not just ident?
    argList = p.Group(OptionalDelimitedList(expr, comma))
    functionCall = p.Group(identAsVar + Adjacent(oparen) - argList +
                           cparen).setName('FnCall').setParseAction(Actions.FunctionCall)

    # Tuples
    tuple = p.Group(oparen + expr + comma - OptionalDelimitedList(expr, comma) +
                    cparen).setName('Tuple').setParseAction(Actions.Tuple)

    # Accessors
    accessor = p.Combine(Adjacent(p.Suppress('.')) -
                         p.oneOf('IS_SIMPLE_VALUE IS_ARRAY IS_STRING IS_TUPLE IS_FUNCTION IS_NULL IS_DEFAULT '
                                 'NUM_DIMS NUM_ELEMENTS SHAPE')).setName('Accessor')

    # Indexing
    pad = (MakeKw('pad') + Adjacent(colon) - expr + eq + expr).setResultsName('pad')
    shrink = (MakeKw('shrink') + Adjacent(colon) - expr).setResultsName('shrink')
    index_dim = expr.setResultsName('dim')
    index = p.Group(Adjacent(p.Suppress('{')) - expr +
                    p.ZeroOrMore(comma - (pad | shrink | index_dim)) + p.Suppress('}')).setName('Index')

    # Special values
    nullValue = p.Group(MakeKw('null')).setName('Null').setParseAction(Actions.Symbol('null'))
    defaultValue = p.Group(MakeKw('default')).setName('Default').setParseAction(Actions.Symbol('defaultParameter'))
    stringValue = quotedString.copy().setName('String').setParseAction(Actions.Symbol('string'))

    # Recognised MathML operators
    mathmlOperators = set('''quotient rem max min root xor abs floor ceiling exp ln log
                             sin cos tan   sec csc cot   sinh cosh tanh   sech csch coth
                             arcsin arccos arctan   arccosh arccot arccoth
                             arccsc arccsch arcsec   arcsech arcsinh arctanh'''.split())

    # Wrapping MathML operators into lambdas
    mathmlOperator = (p.oneOf('^ * / + - not == != <= >= < > && ||') |
                      p.Combine('MathML:' + p.oneOf(' '.join(mathmlOperators))))
    wrap = p.Group(p.Suppress('@') - Adjacent(p.Word(p.nums)) + Adjacent(colon) + mathmlOperator
                   ).setName('WrapMathML').setParseAction(Actions.Wrap)

    # Turning on tracing for debugging protocols
    trace = Adjacent(p.Suppress('?'))

    # The main expression grammar.  Atoms are ordered according to rough speed of detecting mis-match.
    atom = (array | wrap | number.copy().setParseAction(Actions.Number) | stringValue |
            ifExpr | nullValue | defaultValue | lambdaExpr | functionCall | identAsVar | tuple).setName('Atom')
    expr <<= p.infixNotation(atom, [(accessor, 1, p.opAssoc.LEFT, Actions.Accessor),
                                    (viewSpec, 1, p.opAssoc.LEFT, Actions.View),
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
    simpleExpr = p.Forward().setName('SimpleExpression')
    simpleIfExpr = p.Group(MakeKw('if') - simpleExpr + MakeKw('then') - simpleExpr +
                           MakeKw('else') - simpleExpr).setName('SimpleIfThenElse').setParseAction(Actions.Piecewise)
    simpleArgList = p.Group(OptionalDelimitedList(simpleExpr, comma))
    simpleFunctionCall = p.Group(identAsVar + Adjacent(oparen) - simpleArgList +
                                 cparen).setName('SimpleFnCall').setParseAction(Actions.FunctionCall)
    simpleExpr <<= p.infixNotation(number.copy().setParseAction(Actions.Number) |
                                   simpleIfExpr | simpleFunctionCall | identAsVar,
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
    simpleParamList = p.Group(OptionalDelimitedList(p.Group(ncIdentAsVar), comma))
    simpleLambdaExpr = p.Group(MakeKw('lambda') - simpleParamList + colon -
                               expr).setName('SimpleLambda').setParseAction(Actions.Lambda)

    # Newlines in expressions may be escaped with a backslash
    expr.ignore('\\' + p.LineEnd())
    simpleExpr.ignore('\\' + p.LineEnd())
    # Bare newlines are OK provided we started with a bracket.
    # However, it's quite hard to enforce that restriction.
    expr.ignore(p.Literal('\n'))
    simpleExpr.ignore(p.Literal('\n'))
    # Embedded comments are also OK
    expr.ignore(comment)
    simpleExpr.ignore(comment)
    # Avoid mayhem
    UnIgnore(nl)

    # Statements from the "post-processing" language
    ################################################

    # Simple assignment (i.e. not to a tuple)
    simpleAssign = p.Group(ncIdentAsVar + eq - expr).setName('SimpleAssign').setParseAction(Actions.Assignment)
    simpleAssignList = p.Group(OptionalDelimitedList(simpleAssign, nl)).setParseAction(Actions.StatementList)

    # Assertions and function returns
    assertStmt = p.Group(MakeKw('assert') - expr).setName('AssertStmt').setParseAction(Actions.Assert)
    returnStmt = p.Group(MakeKw('return') - p.delimitedList(expr)).setName('ReturnStmt').setParseAction(Actions.Return)

    # Full assignment, to a tuple of names or single name
    _idents = p.Group(p.delimitedList(ncIdentAsVar)).setParseAction(Actions.MaybeTuple)
    assignStmt = p.Group(((MakeKw('optional', suppress=False)("optional") + _idents) | _idents) + eq -
                         p.Group(p.delimitedList(expr)).setParseAction(Actions.MaybeTuple))   \
        .setName('AssignStmt').setParseAction(Actions.Assignment)

    # Function definition
    functionDefn = p.Group(MakeKw('def') - ncIdentAsVar + oparen + paramList + cparen -
                           ((colon - expr) | (obrace - stmtList + Optional(nl) + p.Suppress('}')))
                           ).setName('FunctionDef').setParseAction(Actions.FunctionDef)

    stmtList << p.Group(p.delimitedList(assertStmt | returnStmt | functionDefn | assignStmt, nl))
    stmtList.setParseAction(Actions.StatementList)

    # Miscellaneous constructs making up protocols
    ##############################################

    # Documentation (Markdown)
    documentation = p.Group(MakeKw('documentation') - obrace - p.Regex("[^}]*") + cbrace)("dox")

    # Namespace declarations
    nsDecl = p.Group(MakeKw('namespace') - ncIdent("prefix") + eq + quotedUri("uri")).setName('NamespaceDecl')
    nsDecls = OptionalDelimitedList(nsDecl("namespace*"), nl)

    # Protocol input declarations, with default values
    inputs = (MakeKw('inputs') - obrace - simpleAssignList + cbrace).setName('Inputs').setParseAction(Actions.Inputs)

    # Import statements
    importStmt = p.Group(
        MakeKw('import') -
        Optional(
            ncIdent +
            eq,
            default='') +
        quotedUri +
        Optional(
            obrace -
            simpleAssignList +
            embedded_cbrace)).setName('Import').setParseAction(
                Actions.Import)
    imports = OptionalDelimitedList(importStmt, nl).setName('Imports')

    # Library, globals defined using post-processing language.
    # Strictly speaking returns aren't allowed, but that gets picked up later.
    library = (MakeKw('library') - obrace - Optional(stmtList) +
               cbrace).setName('Library').setParseAction(Actions.Library)

    # Post-processing
    postProcessing = (MakeKw('post-processing') + obrace -
                      OptionalDelimitedList(assertStmt | returnStmt | functionDefn | assignStmt, nl) +
                      cbrace).setName('PostProc').setParseAction(Actions.PostProcessing)

    # Units definitions
    siPrefix = p.oneOf('deka hecto kilo mega giga tera peta exa zetta yotta'
                       'deci centi milli micro nano pico femto atto zepto yocto')
    _num_or_expr = p.originalTextFor(plainNumber | (oparen + expr + cparen))
    unitRef = p.Group(Optional(_num_or_expr)("multiplier") + Optional(siPrefix)("prefix") + ncIdent("units") +
                      Optional(p.Suppress('^') + plainNumber)("exponent") +
                      Optional(p.Group(p.oneOf('- +') + _num_or_expr))("offset")).setParseAction(Actions.UnitRef)
    unitsDef = p.Group(ncIdent + eq + p.delimitedList(unitRef, '.') + Optional(quotedString)("description")
                       ).setName('UnitsDefinition').setParseAction(Actions.UnitsDef)
    units = (MakeKw('units') - obrace - OptionalDelimitedList(unitsDef, nl) + cbrace
             ).setName('Units').setParseAction(Actions.Units)

    # Model interface section
    #########################
    unitsRef = MakeKw('units') - ncIdent

    # Setting the units for the independent variable
    setTimeUnits = (MakeKw('independent') - MakeKw('var') - unitsRef("units")).setParseAction(Actions.SetTimeUnits)
    # Input variables, with optional units and initial value
    inputVariable = p.Group(
        MakeKw('input') -
        cIdent("name") +
        Optional(unitsRef)("units") +
        Optional(
            eq +
            plainNumber)("initial_value")).setName('InputVariable').setParseAction(
        Actions.InputVariable)
    # Model outputs of interest, with optional units
    outputVariable = p.Group(MakeKw('output') - cIdent("name") + Optional(unitsRef("units"))
                             ).setName('OutputVariable').setParseAction(Actions.OutputVariable)
    # Model variables (inputs, outputs, or just used in equations) that are allowed to be missing
    locator = p.Empty().leaveWhitespace().setParseAction(lambda s, l, t: l)
    varDefault = MakeKw('default') - locator("default_start") + simpleExpr("default")
    optionalVariable = p.Group(MakeKw('optional') - cIdent("name") + Optional(varDefault) + locator("default_end")
                               ).setName('OptionalVar').setParseAction(Actions.OptionalVariable)
    # New variables added to the model, with optional initial value
    newVariable = p.Group(
        MakeKw('var') -
        ncIdent("name") +
        unitsRef("units") +
        Optional(
            eq +
            plainNumber)("initial_value")).setName('NewVariable').setParseAction(
        Actions.DeclareVariable)
    # Adding or replacing equations in the model
    clampVariable = p.Group(MakeKw('clamp') - identAsVar + Optional(MakeKw('to') - simpleExpr)
                            ).setName('ClampVariable').setParseAction(Actions.ClampVariable)
    interpolate = p.Group(
        MakeKw('interpolate') -
        oparen -
        quotedString -
        comma -
        identAsVar -
        comma -
        ncIdent -
        comma -
        ncIdent -
        cparen).setName('Interpolate').setParseAction(
        Actions.Interpolate)
    modelEquation = p.Group(MakeKw('define') -
                            (p.Group(MakeKw('diff') +
                                     Adjacent(oparen) -
                                     identAsVar +
                                     p.Suppress(';') +
                                     identAsVar +
                                     cparen) | identAsVar) +
                            eq +
                            (interpolate | simpleExpr)
                            ).setName('AddOrReplaceEquation').setParseAction(Actions.ModelEquation)
    # Units conversion rules
    unitsConversion = p.Group(
        MakeKw('convert') -
        ncIdent("actualDimensions") +
        MakeKw('to') +
        ncIdent("desiredDimensions") +
        MakeKw('by') -
        simpleLambdaExpr).setName('UnitsConversion').setParseAction(
        Actions.UnitsConversion)

    modelInterface = p.Group(MakeKw('model') - MakeKw('interface') - obrace -
                             Optional(setTimeUnits - nl) +
                             OptionalDelimitedList((inputVariable | outputVariable | optionalVariable | newVariable |
                                                    clampVariable | modelEquation | unitsConversion), nl) +
                             cbrace).setName('ModelInterface').setParseAction(Actions.ModelInterface)

    # Simulation definitions
    ########################

    # Ranges
    uniformRange = MakeKw('uniform') + numericRange
    vectorRange = MakeKw('vector') + expr
    whileRange = MakeKw('while') + expr
    range = p.Group(MakeKw('range') + ncIdent("name") + unitsRef("units") +
                    (uniformRange("uniform") | vectorRange("vector") | whileRange("while"))
                    ).setName('Range').setParseAction(Actions.Range)

    # Modifiers
    modifierWhen = MakeKw('at') - (MakeKw('start', False) |
                                   (MakeKw('each', False) - MakeKw('loop')) |
                                   MakeKw('end', False)).setParseAction(Actions.ModifierWhen)
    setVariable = MakeKw('set') - ident + eq + expr
    saveState = MakeKw('save') - MakeKw('as') - ncIdent
    resetState = MakeKw('reset') - Optional(MakeKw('to') + ncIdent)
    modifier = p.Group(modifierWhen + p.Group(setVariable("set") | saveState("save") | resetState("reset"))
                       ).setName('Modifier').setParseAction(Actions.Modifier)
    modifiers = p.Group(MakeKw('modifiers') + obrace - OptionalDelimitedList(modifier, nl) + cbrace
                        ).setName('Modifiers').setParseAction(Actions.Modifiers)

    # The simulations themselves
    simulation = p.Forward().setName('Simulation')
    _selectOutput = p.Group(MakeKw('select') - Optional(MakeKw('optional', suppress=False)) -
                            MakeKw('output') - ncIdent).setName('SelectOutput')
    nestedProtocol = p.Group(MakeKw('protocol') - quotedUri + obrace +
                             simpleAssignList + Optional(nl) + OptionalDelimitedList(_selectOutput, nl) +
                             cbrace + Optional('?')).setName('NestedProtocol').setParseAction(Actions.NestedProtocol)
    timecourseSim = p.Group(MakeKw('timecourse') - obrace - range + Optional(nl + modifiers) + cbrace
                            ).setName('TimecourseSim').setParseAction(Actions.TimecourseSimulation)
    nestedSim = p.Group(MakeKw('nested') - obrace - range + nl + Optional(modifiers) +
                        p.Group(MakeKw('nests') + (simulation | nestedProtocol | ident)) +
                        cbrace).setName('NestedSim').setParseAction(Actions.NestedSimulation)
    oneStepSim = p.Group(MakeKw('oneStep') - Optional(p.originalTextFor(expr))("step") +
                         Optional(obrace - modifiers + cbrace)("modifiers")).setParseAction(Actions.OneStepSimulation)
    simulation << p.Group(MakeKw('simulation') - Optional(ncIdent + eq, default='') +
                          (timecourseSim | nestedSim | oneStepSim) -
                          Optional('?' + nl)).setParseAction(Actions.Simulation)

    tasks = p.Group(MakeKw('tasks') + obrace - p.ZeroOrMore(simulation) +
                    cbrace).setName('Tasks').setParseAction(Actions.Tasks)

    # Output specifications
    #######################

    outputDesc = Optional(quotedString)("description")
    outputSpec = p.Group(Optional(MakeKw('optional', suppress=False))("optional") +
                         ncIdent("name") +
                         ((unitsRef("units") +
                           outputDesc) | (eq +
                                          ident("ref") +
                                          Optional(unitsRef)("units") +
                                          outputDesc))).setName('Output').setParseAction(Actions.Output)
    outputs = p.Group(MakeKw('outputs') + obrace - OptionalDelimitedList(outputSpec, nl) +
                      cbrace).setName('Outputs').setParseAction(Actions.Outputs)

    # Plot specifications
    #####################

    plotCurve = p.Group(p.delimitedList(ncIdent, ',') +
                        MakeKw('against') - ncIdent +
                        Optional(MakeKw('key') - ncIdent("key"))).setName('Curve')
    plotUsing = (MakeKw('using') - (MakeKw('lines', suppress=False) |
                                    MakeKw('points', suppress=False) |
                                    MakeKw('linespoints', suppress=False)))("using")
    plotSpec = p.Group(MakeKw('plot') - quotedString + Optional(plotUsing) - obrace +
                       plotCurve + p.ZeroOrMore(nl + plotCurve) + cbrace).setName('Plot').setParseAction(Actions.Plot)
    plots = p.Group(MakeKw('plots') + obrace - p.ZeroOrMore(plotSpec) +
                    cbrace).setName('Plots').setParseAction(Actions.Plots)

    # Parsing a full protocol
    #########################

    protocol = p.And(
        list(map(Optional, [
            nl,
            documentation,
            nsDecls + nl,
            inputs,
            imports + nl,
            library,
            units,
            modelInterface,
            tasks,
            postProcessing,
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

    # def ParseFile(self, filename, xmlGenerator=None):
    #    """Main entry point for parsing a single protocol file; returns an ElementTree."""
    #    Actions.source_file = filename
    #    Actions.units_map = {}
    #    if xmlGenerator is None:
    #        xmlGenerator = self._Try(self.protocol.parseFile, filename, parseAll=True)[0]
    #    xml = xmlGenerator.xml()
    #    xml.base = filename
    #    return ET.ElementTree(xml)

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
        return " at loc " + str(loc) + "(%d,%d)" % (p.lineno(loc, instring), p.col(loc, instring))

    def SuccessDebugAction(instring, startloc, endloc, expr, toks):
        print("Matched " + str(expr) + " -> " + str(toks.asList()) + DisplayLoc(instring, endloc))

    def ExceptionDebugAction(instring, loc, expr, exc):
        print("Exception raised:" + str(exc) + DisplayLoc(instring, loc))

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

