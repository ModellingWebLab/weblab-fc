
"""
Expression types available in the Functional Curation protocol language.

Importing this package makes all expression classes defined in submodules available within its scope.
Users will thus typically do:
    import fc.language.expressions as E
and access `E.Const` etc.
"""

# Import submodules and make the expressions they define available locally.

from .abstract import *
from .general import *
from .maths import *
from .array import *


def N(number):
    """A convenience expression constructor for defining constant numbers."""
    return Const(V.Simple(number))
