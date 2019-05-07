
import numpy as np

from .abstract import AbstractExpression
from ...error_handling import ProtocolError
from .. import values as V


class Const(AbstractExpression):
    """Class for constant value as expression."""

    def __init__(self, value):
        super(Const, self).__init__()
        self.value = value

    def interpret(self, env):
        return self.value

    def compile(self, array_context=True):
        """If this is a simple value, return a string containing the value."""
        try:
            return str(self.value.value)
        except AttributeError:
            raise NotImplementedError

    def get_used_variables(self):
        return set()


class FunctionCall(AbstractExpression):
    """Used to call a function in the protocol language."""

    def __init__(self, function_or_name, children):
        super(FunctionCall, self).__init__()
        if isinstance(function_or_name, str):
            self.function = NameLookUp(function_or_name)
        else:
            self.function = function_or_name
        self.children = children

    def interpret(self, env):
        actual_params = self.evaluate_children(env)
        function = self.function.evaluate(env)
        if not isinstance(function, V.LambdaClosure):
            raise ProtocolError(function, "is not a function")
        return function.evaluate(env, actual_params)


class If(AbstractExpression):
    """If, then, else statement in protocol language."""

    def __init__(self, test_expr, then_expr, else_expr):
        super(If, self).__init__()
        self.test_expr = test_expr
        self.then_expr = then_expr
        self.else_expr = else_expr

    def interpret(self, env):
        test = self.test_expr.evaluate(env)
        if not hasattr(test, 'value'):
            raise ProtocolError("The test in an if expression must be a simple value or 0-d array, not", test)
        if test.value:
            result = self.then_expr.evaluate(env)
        else:
            result = self.else_expr.evaluate(env)
        return result

    def get_used_variables(self):
        result = self.test_expr.get_used_variables()
        result |= self.then_expr.get_used_variables()
        result |= self.else_expr.get_used_variables()
        return result

    def compile(self, array_context=True):
        test = self.test_expr.compile()
        then = self.then_expr.compile()
        else_ = self.else_expr.compile()
        if array_context:
            return '___np.where(%s,%s,%s)' % (test, then, else_)
        else:
            return '(%s if %s else %s)' % (then, test, else_)


class NameLookUp(AbstractExpression):
    """Used to look up a name for a given environment"""

    # This is used to replace colons when turning a name into a valid Python identifier
    PREFIXED_NAME = '__PPM__'

    @staticmethod
    def pythonize_name(name):
        if ':' in name:
            return NameLookUp.PREFIXED_NAME + name.replace(':', NameLookUp.PREFIXED_NAME)
        return name

    def __init__(self, name):
        assert self.PREFIXED_NAME not in name, "Choice of variable name breaks an implementation assumption"
        super(NameLookUp, self).__init__()
        self.name = name

    def interpret(self, env):
        return env.look_up(self.name)

    def compile(self, array_context=True):
        return self.pythonize_name(self.name)

    def get_used_variables(self):
        return {self.name}


class TupleExpression(AbstractExpression):
    """Expression that returns a protocol language tuple when evaluated."""

    def __init__(self, *children):
        super(TupleExpression, self).__init__(*children)
        if len(self.children) < 1:
            raise ProtocolError("Empty tuple expressions are not allowed")

    def interpret(self, env):
        return V.Tuple(*self.evaluate_children(env))


class LambdaExpression(AbstractExpression):
    """Expression for function in protocol language."""

    def __init__(self, formal_parameters, body, default_parameters=None):
        super(LambdaExpression, self).__init__()
        self.formal_parameters = formal_parameters
        self.body = body
        self.default_parameters = default_parameters

    def interpret(self, env):
        return V.LambdaClosure(env, self.formal_parameters, self.body, self.default_parameters)

    @staticmethod
    def wrap(operator, num_params):
        parameters = []
        look_up_list = []
        for i in range(num_params):
            parameters.append("___" + str(i))
            look_up_list.append(NameLookUp("___" + str(i)))
        from ..statements import Return
        body = [Return(operator(*look_up_list))]
        function = LambdaExpression(parameters, body)
        return function


class Accessor(AbstractExpression):
    """Expression that reports type of protocol language value."""
    IS_SIMPLE_VALUE = 0
    IS_ARRAY = 1
    IS_STRING = 2
    IS_FUNCTION = 3
    IS_TUPLE = 4
    IS_NULL = 5
    IS_DEFAULT = 6
    NUM_DIMS = 7
    NUM_ELEMENTS = 8
    SHAPE = 9

    def __init__(self, variable_expr, attribute):
        super(Accessor, self).__init__()
        self.variable_expr = variable_expr
        self.attribute = attribute

    def interpret(self, env):
        variable = self.variable_expr.evaluate(env)
        if self.attribute == self.IS_SIMPLE_VALUE:
            result = hasattr(variable, 'value')
        elif self.attribute == self.IS_ARRAY:
            result = hasattr(variable, 'array')
        elif self.attribute == self.IS_STRING:
            result = isinstance(variable, V.String)
        elif self.attribute == self.IS_FUNCTION:
            result = isinstance(variable, V.LambdaClosure)
        elif self.attribute == self.IS_TUPLE:
            result = isinstance(variable, V.Tuple)
        elif self.attribute == self.IS_NULL:
            result = isinstance(variable, V.Null)
        elif self.attribute == self.IS_DEFAULT:
            result = isinstance(variable, V.DefaultParameter)
        elif self.attribute == self.NUM_DIMS:
            try:
                result = variable.array.ndim
            except AttributeError:
                raise ProtocolError("Cannot get number of dimensions of non-array", variable)
        elif self.attribute == 8:
            try:
                result = variable.array.size
            except AttributeError:
                raise ProtocolError("Cannot get number of elements of non-array", variable)
        elif self.attribute == 9:
            try:
                result = V.Array(np.array(variable.array.shape))
            except AttributeError:
                raise ProtocolError("Cannot get shape of non-array", variable)
        if isinstance(result, V.Array):
            return result
        else:
            return V.Simple(result)

