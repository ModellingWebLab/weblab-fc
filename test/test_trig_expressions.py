"""
Test trigonometric math expressions using simple values.
"""
import pytest

import fc.language.expressions as E
import fc.language.values as V
import fc.environment as Env


def test_sin():
    env = Env.Environment()
    assert E.Sin(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(0.1411200081)


def test_cos():
    env = Env.Environment()
    assert E.Cos(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(-0.9899924966)


def test_tan():
    env = Env.Environment()
    assert E.Tan(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(-0.1425465431)


def test_asin():
    env = Env.Environment()
    assert E.ArcSin(E.Const(V.Simple(0.3))).evaluate(env).value == pytest.approx(0.304692654)


def test_acos():
    env = Env.Environment()
    assert E.ArcCos(E.Const(V.Simple(0.3))).evaluate(env).value == pytest.approx(1.266103673)


def test_atan():
    env = Env.Environment()
    assert E.ArcTan(E.Const(V.Simple(0.3))).evaluate(env).value == pytest.approx(0.2914567945)


def test_sinh():
    env = Env.Environment()
    assert E.Sinh(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(10.01787493)


def test_cosh():
    env = Env.Environment()
    assert E.Cosh(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(10.06766200)


def test_tanh():
    env = Env.Environment()
    assert E.Tanh(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(0.9950547537)


def test_asinh():
    env = Env.Environment()
    assert E.ArcSinh(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(1.818446459)


def test_acosh():
    env = Env.Environment()
    assert E.ArcCosh(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(1.762747174)


def test_atanh():
    env = Env.Environment()
    assert E.ArcTanh(E.Const(V.Simple(0.3))).evaluate(env).value == pytest.approx(0.3095196042)


def test_sec():
    env = Env.Environment()
    assert E.Sec(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(-1.010108666)


def test_csc():
    env = Env.Environment()
    assert E.Csc(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(7.086167396)


def test_cot():
    env = Env.Environment()
    assert E.Cot(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(-7.015252551)


def test_asec():
    env = Env.Environment()
    assert E.ArcSec(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(1.230959417)


def test_acsc():
    env = Env.Environment()
    assert E.ArcCsc(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(0.3398369095)


def test_acot():
    env = Env.Environment()
    assert E.ArcCot(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(0.3217505544)


def test_sech():
    env = Env.Environment()
    assert E.Sech(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(0.09932792742)


def test_csch():
    env = Env.Environment()
    assert E.Csch(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(0.09982156967)


def test_coth():
    env = Env.Environment()
    assert E.Coth(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(1.004969823)


def test_asech():
    env = Env.Environment()
    assert E.ArcSech(E.Const(V.Simple(0.3))).evaluate(env).value == pytest.approx(1.873820243)


def test_acsch():
    env = Env.Environment()
    assert E.ArcCsch(E.Const(V.Simple(0.3))).evaluate(env).value == pytest.approx(1.918896472)


def test_acoth():
    env = Env.Environment()
    assert E.ArcCoth(E.Const(V.Simple(3))).evaluate(env).value == pytest.approx(0.3465735903)

