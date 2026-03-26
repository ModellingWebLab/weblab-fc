"""
Expression types available in the Functional Curation protocol language.

Importing this package makes all expression classes defined in submodules available within its scope.
Users will thus typically do:
    import fc.language.expressions as E
and access `E.Const` etc.
"""

# Import submodules and make the expressions they define available locally.

from .abstract import AbstractExpression
from .array import Find, Fold, Index, Map, NewArray, View
from .general import Accessor, Const, FunctionCall, If, LambdaExpression, NameLookUp, TupleExpression
from .maths import (
    Abs, And, Ceiling, Divide, Eq, Exp, Floor, Geq, Gt, Leq, Ln, Log, Lt, Max, Min, Minus, Neq, Not, Or, Plus, Power,
    Rem, Root, Times, Xor
)
from .trig import (
    ArcCos, ArcCosh, ArcCot, ArcCoth, ArcCsc, ArcCsch, ArcSec, ArcSech, ArcSin, ArcSinh, ArcTan, ArcTanh, Cos, Cosh,
    Cot, Coth, Csc, Csch, Sec, Sech, Sin, Sinh, Tan, Tanh, TrigExpression
)

__all__ = [
    "AbstractExpression",
    "Abs",
    "And",
    "Accessor",
    "Ceiling",
    "Const",
    "Divide",
    "Eq",
    "Exp",
    "Floor",
    "FunctionCall",
    "Geq",
    "Gt",
    "If",
    "Leq",
    "NameLookUp",
    "Ln",
    "Log",
    "Lt",
    "Max",
    "Min",
    "Minus",
    "Neq",
    "Not",
    "Or",
    "Plus",
    "Power",
    "Rem",
    "Root",
    "Times",
    "ArcCos",
    "ArcCosh",
    "ArcCot",
    "ArcCoth",
    "ArcCsc",
    "ArcCsch",
    "ArcSech",
    "ArcSec",
    "ArcSin",
    "ArcSinh",
    "ArcTan",
    "ArcTanh",
    "Cos",
    "Cosh",
    "Cot",
    "Coth",
    "Csc",
    "Csch",
    "Sec",
    "Sech",
    "Sin",
    "Sinh",
    "Tan",
    "Tanh",
    "TrigExpression",
    "Find",
    "Fold",
    "Index",
    "Map",
    "NewArray",
    "TupleExpression",
    "View",
    "LambdaExpression",
    "Xor",
]


def N(number):
    """A convenience expression constructor for defining constant numbers."""
    from .. import values

    return Const(values.Simple(number))
