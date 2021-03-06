"""
Protocol language statements: statements that can appear in certain parts of a protocol, e.g. assignments and
assertions.
"""

from . import expressions as E
from . import values as V
from .. import locatable
from ..error_handling import ProtocolError


class AbstractStatement(locatable.Locatable):
    """Base class for statements in the protocol language."""

    def evaluate(self, env):
        raise NotImplementedError


class Assign(AbstractStatement):
    """Assign statements in the protocol language."""

    def __init__(self, names, rhs, optional=False):
        super(Assign, self).__init__()
        self.names = names
        self.rhs = rhs
        self.optional = optional

    def evaluate(self, env):
        try:
            results = self.rhs.evaluate(env)
        except Exception:
            if not self.optional:
                raise
            return V.Null()
        if len(self.names) > 1:
            if not isinstance(results, V.Tuple):
                raise ProtocolError("When assigning multiple names the value to assign must be a tuple.")
            env.define_names(self.names, results.values)
        else:
            env.define_name(self.names[0], results)
        return V.Null()


class Assert(AbstractStatement):
    """
    Assertion used within a protocol.
    """
    def __init__(self, expr):
        super(Assert, self).__init__()
        self.expr = expr

    def evaluate(self, env):
        result = self.expr.evaluate(env)
        try:
            ok = result.value
        except AttributeError:
            raise ProtocolError("Assertion did not yield a Simple value or 0-d Array.")
        if not ok:
            # TODO: Add more conditional types here (see #170)
            if isinstance(self.expr, E.Eq):
                operands = self.expr.evaluate_children(env)
                lhs = operands[0].value
                rhs = operands[1].value
                raise ProtocolError(f'Assertion failed: {lhs} != {rhs}.')
            raise ProtocolError("Assertion failed.")
        return V.Null()


class Return(AbstractStatement):
    def __init__(self, *parameters):
        super(Return, self).__init__()
        self.parameters = parameters

    def evaluate(self, env):
        results = [expr.evaluate(env) for expr in self.parameters]
        if len(results) == 0:
            return V.Null()
        elif len(results) == 1:
            return results[0]
        else:
            return V.Tuple(*results)

    def compile(self, env):
        if len(self.parameters) == 1:
            from .expressions import Const
            if isinstance(self.parameters[0], Const):
                raise NotImplementedError
            expression = self.parameters[0].compile()
        else:
            raise NotImplementedError
        return expression
