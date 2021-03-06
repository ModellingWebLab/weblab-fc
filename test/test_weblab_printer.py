#
# Tests conversion of sympy expressions to weblab cython code.
#
import logging
import math
import pytest
import sympy as sp

import fc.code_generation as cg


# Show more logging output
logging.getLogger().setLevel(logging.INFO)


class TestWebLabPrinter(object):

    @pytest.fixture(scope="class")
    def x(self):
        return sp.symbols('x')

    @pytest.fixture(scope="class")
    def y(self):
        return sp.symbols('y')

    @pytest.fixture(scope="class")
    def z(self):
        return sp.symbols('z')

    @pytest.fixture(scope="class")
    def p(self):
        return cg.WebLabPrinter()

    def test_numbers(self, p, x):
        # Number types
        assert p.doprint(1) == '1'                  # int
        assert p.doprint(1.2) == '1.2'              # float, short format
        assert p.doprint(math.pi) == '3.141592653589793'  # float, long format
        assert p.doprint(1.436432635636e-123) == '1.436432635636e-123'
        assert p.doprint(x - x) == '0'              # Zero
        assert p.doprint(x / x) == '1'              # One
        assert p.doprint(-x / x) == '-1'            # Negative one
        assert p.doprint(5 * (x / x)) == '5'        # Sympy integer
        assert p.doprint(5.5 * (x / x)) == '5.5'        # Sympy float
        assert p.doprint(sp.Rational(5, 7)) == '5 / 7'  # Sympy rational

        # Special numbers
        assert p.doprint(sp.pi) == 'math.pi'
        assert p.doprint(sp.E) == 'math.e'

    def test_symbols(self, p, x, y):
        # Symbols
        assert p.doprint(x) == 'x'

        # Derivatives
        assert p.doprint(sp.Derivative(x, y)) == 'Derivative(x, y)'

        # Symbol function
        def symbol_function(symbol):
            return symbol.name.upper()

        q = cg.WebLabPrinter(symbol_function)
        assert q.doprint(x) == 'X'
        assert q.doprint(sp.Derivative(x, y)) == p.doprint(sp.Derivative(x, y))

        # Derivative function
        def derivative_function(deriv):
            a = deriv.expr
            b = deriv.variables[0]
            return 'd' + a.name + '/' + 'd' + b.name.upper()

        q = cg.WebLabPrinter(derivative_function=derivative_function)
        assert q.doprint(sp.Derivative(x, y)) == 'dx/dY'
        assert q.doprint(x) == p.doprint(x)

        # Both
        q = cg.WebLabPrinter(symbol_function, derivative_function)
        assert q.doprint(x) == 'X'
        assert q.doprint(sp.Derivative(x, y)) == 'dx/dY'

    def test_addition(self, p, x, y, z):

        # Addition and subtraction
        assert p.doprint(x + y) == 'x + y'
        assert p.doprint(x + y + z) == 'x + y + z'
        assert p.doprint(x - y) == 'x - y'
        assert p.doprint(2 + z) == '2 + z'
        assert p.doprint(z + 2) == '2 + z'
        assert p.doprint(z - 2) == '-2 + z'
        assert p.doprint(2 - z) == '2 - z'
        assert p.doprint(-x) == '-x'
        assert p.doprint(-x - 2) == '-2 - x'

    def test_multiplication(self, p, x, y, z):

        # Multiplication and division
        assert p.doprint(x * y) == 'x * y'
        assert p.doprint(x * y * z) == 'x * y * z'
        assert p.doprint(x / y) == 'x / y'
        assert p.doprint(2 * z) == '2 * z'
        assert p.doprint(z * 5) == '5 * z'
        assert p.doprint(4 / z) == '4 / z'
        assert p.doprint(z / 3) == 'z / 3'
        assert p.doprint(1 / x) == '1 / x'  # Uses pow
        assert p.doprint(1 / (x * y)) == '1 / (x * y)'
        assert p.doprint(1 / -(x * y)) == '-1 / (x * y)'
        assert p.doprint(x + (y + z)) == 'x + y + z'
        assert p.doprint(x * (y + z)) == 'x * (y + z)'
        assert p.doprint(x * y * z) == 'x * y * z'
        assert p.doprint(x + y > x * z), 'x + y > x * z'
        assert p.doprint(x**2 + y**2) == 'x**2 + y**2'
        assert p.doprint(x**2 + 3 * y**2) == 'x**2 + 3 * y**2'
        assert p.doprint(x**(2 + y**2)) == 'x**(2 + y**2)'
        assert p.doprint(x**(2 + 3 * y**2)) == 'x**(2 + 3 * y**2)'
        assert p.doprint(x**-1 * y**-1) == '1 / (x * y)'
        assert p.doprint(x / y / z) == 'x / (y * z)'
        assert p.doprint(x / y * z) == 'x * z / y'
        assert p.doprint(x / (y * z)) == 'x / (y * z)'
        assert p.doprint(x * y**(-2 / (3 * x / x))) == 'x / y**(2 / 3)'

        # Sympy issue #14160
        d = sp.Mul(
            -2,
            x,
            sp.Pow(sp.Mul(y, y, evaluate=False), -1, evaluate=False),
            evaluate=False
        )
        assert p.doprint(d) == '-2 * x / (y * y)'

    def test_powers(self, p, x, y, z):

        # Powers and square roots
        assert p.doprint(sp.sqrt(2)) == 'math.sqrt(2)'
        assert p.doprint(1 / sp.sqrt(2)) == 'math.sqrt(2) / 2'
        assert p.doprint(
            sp.Mul(1, 1 / sp.sqrt(2))) == 'math.sqrt(2) / 2'
        assert p.doprint(sp.sqrt(x)) == 'math.sqrt(x)'
        assert p.doprint(1 / sp.sqrt(x)) == '1 / math.sqrt(x)'
        assert p.doprint(x**(x / (2 * x))) == 'math.sqrt(x)'
        assert p.doprint(x**(x / (-2 * x))) == '1 / math.sqrt(x)'
        assert p.doprint(x**-1) == '1 / x'
        assert p.doprint(x**0.5) == 'x**0.5'
        assert p.doprint(x**-0.5) == 'x**(-0.5)'
        assert p.doprint(x**(1 + y)) == 'x**(1 + y)'
        assert p.doprint(x**-(1 + y)) == 'x**(-1 - y)'
        assert p.doprint((x + z)**-(1 + y)) == '(x + z)**(-1 - y)'
        assert p.doprint(x**-2) == 'x**(-2)'
        assert p.doprint(x**3.2) == 'x**3.2'

    def test_trig_functions(self, p, x):

        # Trig functions
        assert p.doprint(sp.acos(x)) == 'math.acos(x)'
        assert p.doprint(sp.acosh(x)) == 'math.acosh(x)'
        assert p.doprint(sp.asin(x)) == 'math.asin(x)'
        assert p.doprint(sp.asinh(x)) == 'math.asinh(x)'
        assert p.doprint(sp.atan(x)) == 'math.atan(x)'
        assert p.doprint(sp.atanh(x)) == 'math.atanh(x)'
        assert p.doprint(sp.ceiling(x)) == 'math.ceil(x)'
        assert p.doprint(sp.cos(x)) == 'math.cos(x)'
        assert p.doprint(sp.cosh(x)) == 'math.cosh(x)'
        assert p.doprint(sp.Function('_exp')(x)) == 'math.exp(x)'
        assert p.doprint(sp.factorial(x)) == 'math.factorial(x)'
        assert p.doprint(sp.floor(x)) == 'math.floor(x)'
        assert p.doprint(sp.log(x)) == 'math.log(x)'
        assert p.doprint(sp.sin(x)) == 'math.sin(x)'
        assert p.doprint(sp.sinh(x)) == 'math.sinh(x)'
        assert p.doprint(sp.tan(x)) == 'math.tan(x)'
        assert p.doprint(sp.tanh(x)) == 'math.tanh(x)'

    def test_conditions(self, p, x, y, z):

        # Conditions
        assert p.doprint(sp.Eq(x, y)) == 'x == y'
        assert p.doprint(sp.Eq(x, sp.Eq(y, z))) == 'x == (y == z)'
        assert p.doprint(sp.Eq(sp.Eq(x, y), z)) == '(x == y) == z'
        assert p.doprint(sp.Ne(x, y)) == 'x != y'
        assert p.doprint(sp.Gt(x, y)) == 'x > y'
        assert p.doprint(sp.Lt(x, y)) == 'x < y'
        assert p.doprint(sp.Ge(x, y)) == 'x >= y'
        assert p.doprint(sp.Le(x, y)) == 'x <= y'
        e = sp.Eq(sp.Eq(x, 3), sp.Eq(y, 5))
        assert p.doprint(e) == '(x == 3) == (y == 5)'

    def test_boolean_logic(self, p, x, y, z):

        # Boolean logic
        assert p.doprint(True) == 'True'
        assert p.doprint(False) == 'False'
        assert p.doprint(sp.Eq(x, x)) == 'True'
        assert p.doprint(sp.Ne(x, x)) == 'False'
        assert (
            p.doprint(sp.And(sp.Eq(x, y), sp.Eq(x, z))) == 'x == y and x == z')
        assert (
            p.doprint(sp.And(sp.Eq(x, y), sp.Eq(x, z), sp.Eq(x, 2))) ==
            'x == 2 and x == y and x == z')
        assert p.doprint(sp.Or(sp.Eq(x, y), sp.Eq(x, z))) == 'x == y or x == z'
        assert (
            p.doprint(sp.Or(sp.Eq(x, y), sp.Eq(x, z), sp.Eq(x, 2))) ==
            'x == 2 or x == y or x == z')
        a, b, c = x > 2, x > y, x > z
        assert p.doprint(a & b) == 'x > 2 and x > y'
        # 1 or (0 and 0) = 1 = 1 or 0 and 0 -- and binds stronger
        # (1 or 0) and 0 = 0
        assert p.doprint(a | (b & c)) == 'x > 2 or x > y and x > z'
        assert p.doprint((a | b) & c) == 'x > z and (x > 2 or x > y)'

    def test_piecewise_expressions(self, p, x):

        # Piecewise expressions
        e = sp.Piecewise((0, x > 0), (1, True))
        assert p.doprint(e) == '((0) if (x > 0) else (1))'
        e = sp.Piecewise((0, x > 0), (1, x > 1), (2, True))
        assert (
            p.doprint(e) == '((0) if (x > 0) else ((1) if (x > 1) else (2)))')
        e = sp.Piecewise((0, x > 0), (1, x > 1), (2, True), (3, x > 3))
        assert (
            p.doprint(e) == '((0) if (x > 0) else ((1) if (x > 1) else (2)))')
        # Sympy filters out False statements
        e = sp.Piecewise(
            (0, x > 0), (1, x != x), (2, True), (3, x > 3),
            evaluate=False)
        assert p.doprint(e) == '((0) if (x > 0) else (2))'

    def test_long_expression(self, p, x, y, z):

        # Longer expressions
        assert (
            p.doprint((x + y) / (2 + z / sp.log(x - y))) ==
            '(x + y) / (2 + z / math.log(x - y))')
        assert p.doprint((y + sp.sin(x))**-1) == '1 / (y + math.sin(x))'

    def test_unsupported_sympy_items(self, p, x):

        # Unsupported sympy item
        e = sp.Matrix()
        with pytest.raises(ValueError):
            p.doprint(e)

        # Unsupported sympy function
        e = sp.gamma(x)
        with pytest.raises(ValueError):
            p.doprint(e)

    def test_abs(self, p, x, y):
        assert p.doprint(sp.Abs(x + y)) == 'abs(x + y)'
        assert p.doprint(sp.Abs(3.2, evaluate=False)) == 'abs(3.2)'
        assert p.doprint(sp.Abs(-3, evaluate=False)) == 'abs(-3)'
