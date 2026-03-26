"""Trig functions for FC."""
import math

from .abstract import AbstractExpression
from ...error_handling import ProtocolError
from .. import values as V


class TrigExpression(AbstractExpression):
    """Base class for (univariate) trigonometric expressions."""
    _op = None
    _ex = None

    def __init__(self, *children):
        super(TrigExpression, self).__init__(*children)
        if len(self.children) != 1:
            raise ProtocolError(f"Operator '{self._op}' requires one operand, not {len(self.children)}.")

    def interpret(self, env):
        operands = self.evaluate_children(env)
        try:
            result = self._ex(operands[0].value)
        except (AttributeError, TypeError):
            raise ProtocolError("Trigonometric operator requires a number as its operand")
        except ValueError as e:
            raise ProtocolError(f"Invalid input to trigonometric operator: {e}.")
        return V.Simple(result)

    def compile(self, array_context=True):
        operands = ["(" + child.compile(array_context) + ")" for child in self.children]
        expression = f"{self._op}({operands[0]})"
        return expression


class Sin(TrigExpression):
    """Sine."""
    _op = 'sin'
    _ex = math.sin


class Cos(TrigExpression):
    """Cosine."""
    _op = 'cos'
    _ex = math.cos


class Tan(TrigExpression):
    """Tangent."""
    _op = 'tan'
    _ex = math.tan


class ArcSin(TrigExpression):
    """Inverse sine."""
    _op = 'asin'
    _ex = math.asin


class ArcCos(TrigExpression):
    """Inverse cosine."""
    _op = 'acos'
    _ex = math.acos


class ArcTan(TrigExpression):
    """Inverse tangent."""
    _op = 'atan'
    _ex = math.atan


class Sinh(TrigExpression):
    """Hyperbolic sine."""
    _op = 'sinh'
    _ex = math.sinh


class Cosh(TrigExpression):
    """Hyperbolic cosine."""
    _op = 'cosh'
    _ex = math.cosh


class Tanh(TrigExpression):
    """Hyperbolic tangent."""
    _op = 'tanh'
    _ex = math.tanh


class ArcSinh(TrigExpression):
    """Inverse hyperbolic sine."""
    _op = 'asinh'
    _ex = math.asinh


class ArcCosh(TrigExpression):
    """Inverse hyperbolic cosine."""
    _op = 'acosh'
    _ex = math.acosh


class ArcTanh(TrigExpression):
    """Inverse hyperbolic tangent."""
    _op = 'atanh'
    _ex = math.atanh


class Sec(TrigExpression):
    """Secant."""
    _op = 'sec'

    def _ex(self, a):
        return 1 / math.cos(a)


class Csc(TrigExpression):
    """Cosecant."""
    _op = 'csc'

    def _ex(self, a):
        return 1 / math.sin(a)


class Cot(TrigExpression):
    """Cotangent."""
    _op = 'cot'

    def _ex(self, a):
        return 1 / math.tan(a)


class ArcSec(TrigExpression):
    """Inverse secant."""
    _op = 'asec'

    def _ex(self, a):
        return math.acos(1 / a)


class ArcCsc(TrigExpression):
    """Inverse cosecant."""
    _op = 'acsc'

    def _ex(self, a):
        return math.asin(1 / a)


class ArcCot(TrigExpression):
    """Inverse cotangent."""
    _op = 'acot'

    def _ex(self, a):
        return math.atan(1 / a)


class Sech(TrigExpression):
    """Hyperbolic secant."""
    _op = 'sech'

    def _ex(self, a):
        return 1 / math.cosh(a)


class Csch(TrigExpression):
    """Hyperbolic cosecant."""
    _op = 'csch'

    def _ex(self, a):
        return 1 / math.sinh(a)


class Coth(TrigExpression):
    """Hyperbolic cotangent."""
    _op = 'coth'

    def _ex(self, a):
        return 1 / math.tanh(a)


class ArcSech(TrigExpression):
    """Inverse hyperbolic secant."""
    _op = 'asech'

    def _ex(self, a):
        return math.acosh(1 / a)


class ArcCsch(TrigExpression):
    """Inverse hyperbolic cosecant."""
    _op = 'acsch'

    def _ex(self, a):
        return math.asinh(1 / a)


class ArcCoth(TrigExpression):
    """Inverse hyperbolic cotangent."""
    _op = 'acoth'

    def _ex(self, a):
        return math.atanh(1 / a)

