"""
Test relations (eq, neq, lt, gt, lte, gte) using simple values.
"""
import fc.language.expressions as E
import fc.language.values as V


def test_eq():
    assert not E.Eq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert E.Eq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_neq():
    assert E.Neq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert not E.Neq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_lt():
    assert not E.Lt(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert E.Lt(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value
    assert not E.Lt(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_gt():
    assert E.Gt(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert not E.Gt(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value
    assert not E.Gt(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_leq():
    assert not E.Leq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert E.Leq(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value
    assert E.Leq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_geq():
    assert E.Geq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert not E.Geq(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value
    assert E.Geq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value

