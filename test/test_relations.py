
import unittest

import fc.language.expressions as E
import fc.language.values as V


class TestRelations(unittest.TestCase):
    """Test relations (eq, neq, lt, gt, lte, gte) using simple values."""

    def test_eq(self):
        self.assertFalse(E.Eq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value)
        self.assertTrue(E.Eq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value)

    def test_neq(self):
        self.assertTrue(E.Neq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value)
        self.assertFalse(E.Neq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value)

    def test_lt(self):
        self.assertFalse(E.Lt(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value)
        self.assertTrue(E.Lt(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value)
        self.assertFalse(E.Lt(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value)

    def test_gt(self):
        self.assertTrue(E.Gt(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value)
        self.assertFalse(E.Gt(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value)
        self.assertFalse(E.Gt(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value)

    def test_leq(self):
        self.assertFalse(E.Leq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value)
        self.assertTrue(E.Leq(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value)
        self.assertTrue(E.Leq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value)

    def test_geq(self):
        self.assertTrue(E.Geq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value)
        self.assertFalse(E.Geq(E.Const(V.Simple(0)), E.Const(V.Simple(1))).evaluate({}).value)
        self.assertTrue(E.Geq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value)
