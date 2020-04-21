"""Tests logic using simple values. Tests and, or, xor, not."""

import fc.language.expressions as E
import fc.language.values as V


def test_and():
    assert E.And(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value == 0
    assert E.And(E.Const(V.Simple(1)), E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value == 1


def test_or():
    assert E.Or(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value == 1
    assert E.Or(E.Const(V.Simple(0)), E.Const(V.Simple(0)), E.Const(V.Simple(0))).evaluate({}).value == 0


def test_xor():
    assert E.Xor(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value == 1
    assert E.Xor(E.Const(V.Simple(0)), E.Const(V.Simple(0))).evaluate({}).value == 0
    assert E.Xor(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value == 0


def test_not():
    assert E.Not(E.Const(V.Simple(1))).evaluate({}).value == 0
    assert E.Not(E.Const(V.Simple(3))).evaluate({}).value == 0
    assert E.Not(E.Const(V.Simple(0))).evaluate({}).value == 1

