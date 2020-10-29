"""
Test trigonometric math expressions using simple values.
"""
import pytest
import sympy as sp

import fc.language.expressions as E
import fc.language.values as V
import fc.environment as Env


@pytest.fixture
def env():
    return Env.Environment()


def test_sin(env):
    fnc = E.Sin(E.Const(V.Simple(3)))
    ref = 0.1411200081
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_cos(env):
    fnc = E.Cos(E.Const(V.Simple(3)))
    ref = -0.9899924966
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_tan(env):
    fnc = E.Tan(E.Const(V.Simple(3)))
    ref = -0.1425465431
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_asin(env):
    fnc = E.ArcSin(E.Const(V.Simple(0.3)))
    ref = 0.304692654
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_acos(env):
    fnc = E.ArcCos(E.Const(V.Simple(0.3)))
    ref = 1.266103673
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_atan(env):
    fnc = E.ArcTan(E.Const(V.Simple(0.3)))
    ref = 0.2914567945
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_sinh(env):
    fnc = E.Sinh(E.Const(V.Simple(3)))
    ref = 10.01787493
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_cosh(env):
    fnc = E.Cosh(E.Const(V.Simple(3)))
    ref = 10.06766200
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_tanh(env):
    fnc = E.Tanh(E.Const(V.Simple(3)))
    ref = 0.9950547537
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_asinh(env):
    fnc = E.ArcSinh(E.Const(V.Simple(3)))
    ref = 1.818446459
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_acosh(env):
    fnc = E.ArcCosh(E.Const(V.Simple(3)))
    ref = 1.762747174
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_atanh(env):
    fnc = E.ArcTanh(E.Const(V.Simple(0.3)))
    ref = 0.3095196042
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_sec(env):
    fnc = E.Sec(E.Const(V.Simple(3)))
    ref = -1.010108666
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_csc(env):
    fnc = E.Csc(E.Const(V.Simple(3)))
    ref = 7.086167396
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_cot(env):
    fnc = E.Cot(E.Const(V.Simple(3)))
    ref = -7.015252551
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_asec(env):
    fnc = E.ArcSec(E.Const(V.Simple(3)))
    ref = 1.230959417
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_acsc(env):
    fnc = E.ArcCsc(E.Const(V.Simple(3)))
    ref = 0.3398369095
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_acot(env):
    fnc = E.ArcCot(E.Const(V.Simple(3)))
    ref = 0.3217505544
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_sech(env):
    fnc = E.Sech(E.Const(V.Simple(3)))
    ref = 0.09932792742
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_csch(env):
    fnc = E.Csch(E.Const(V.Simple(3)))
    ref = 0.09982156967
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_coth(env):
    fnc = E.Coth(E.Const(V.Simple(3)))
    ref = 1.004969823
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_asech(env):
    fnc = E.ArcSech(E.Const(V.Simple(0.3)))
    ref = 1.873820243
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_acsch(env):
    fnc = E.ArcCsch(E.Const(V.Simple(0.3)))
    ref = 1.918896472
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)


def test_acoth(env):
    fnc = E.ArcCoth(E.Const(V.Simple(3)))
    ref = 0.3465735903
    assert fnc.evaluate(env).value == pytest.approx(ref)
    assert sp.sympify(fnc.compile()).evalf() == pytest.approx(ref)

