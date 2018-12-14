
from . import values as V
from ..utility import locatable
from ..utility.error_handling import ProtocolError


class AbstractStatement(locatable.Locatable):
    """Base class for statements in the protocol language."""

    def Evaluate(self, env):
        raise NotImplementedError


class Assign(AbstractStatement):
    """Assign statements in the protocol language."""

    def __init__(self, names, rhs, optional=False):
        super(Assign, self).__init__()
        self.names = names
        self.rhs = rhs
        self.optional = optional

    def Evaluate(self, env):
        try:
            results = self.rhs.Evaluate(env)
        except Exception:
            if not self.optional:
                raise
            return V.Null()
        if len(self.names) > 1:
            if not isinstance(results, V.Tuple):
                raise ProtocolError("When assigning multiple names the value to assign must be a tuple.")
            env.DefineNames(self.names, results.values)
        else:
            env.DefineName(self.names[0], results)
        return V.Null()


class Assert(AbstractStatement):
    def __init__(self, expr):
        super(Assert, self).__init__()
        self.expr = expr

    def Evaluate(self, env):
        result = self.expr.Evaluate(env)
        try:
            if not result.value:
                raise ProtocolError("Assertion failed.")
        except AttributeError:
            raise ProtocolError("Assertion did not yield a Simple value or 0-d Array.")
        return V.Null()


class Return(AbstractStatement):
    def __init__(self, *parameters):
        super(Return, self).__init__()
        self.parameters = parameters

    def Evaluate(self, env):
        results = [expr.Evaluate(env) for expr in self.parameters]
        if len(results) == 0:
            return V.Null()
        elif len(results) == 1:
            return results[0]
        else:
            return V.Tuple(*results)

    def Compile(self, env):
        if len(self.parameters) == 1:
            from .expressions import Const
            if isinstance(self.parameters[0], Const):
                raise NotImplementedError
            expression = self.parameters[0].Compile()
        else:
            raise NotImplementedError
        return expression
