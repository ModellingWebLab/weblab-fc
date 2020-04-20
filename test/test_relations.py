"""
Test relations (eq, neq, lt, gt, lte, gte) using simple values.
"""
import fc.language.expressions as E
import fc.language.values as V


def test_eq(self):
    assert not E.Eq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert E.Eq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_neq(self):
    assert E.Neq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert not E.Neq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_lt(self):
    assert not E.Lt(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert E.Lt(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value
    assert not E.Lt(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_gt(self):
    assert E.Gt(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert not E.Gt(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value
    assert not E.Gt(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_leq(self):
    assert not E.Leq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert E.Leq(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value
    assert E.Leq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value


def test_geq(self):
    assert E.Geq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value
    assert not E.Geq(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value
    assert E.Geq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value

