
import unittest

import fc.language.expressions as E
import fc.language.values as V


class TestRelations(unittest.TestCase):
    """Test relations (eq, neq, lt, gt, lte, gte) using simple values."""

    def testEq(self):
        self.assertFalse(E.Eq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).Evaluate({}).value)
        self.assertTrue(E.Eq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).Evaluate({}).value)

    def testNeq(self):
        self.assertTrue(E.Neq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).Evaluate({}).value)
        self.assertFalse(E.Neq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).Evaluate({}).value)

    def testLt(self):
        self.assertFalse(E.Lt(E.Const(V.Simple(1)), E.Const(V.Simple(0))).Evaluate({}).value)
        self.assertTrue(E.Lt(E.Const(V.Simple(0)), E.Const(V.Simple(1))).Evaluate({}).value)
        self.assertFalse(E.Lt(E.Const(V.Simple(1)), E.Const(V.Simple(1))).Evaluate({}).value)

    def testGt(self):
        self.assertTrue(E.Gt(E.Const(V.Simple(1)), E.Const(V.Simple(0))).Evaluate({}).value)
        self.assertFalse(E.Gt(E.Const(V.Simple(0)), E.Const(V.Simple(1))).Evaluate({}).value)
        self.assertFalse(E.Gt(E.Const(V.Simple(1)), E.Const(V.Simple(1))).Evaluate({}).value)

    def testLeq(self):
        self.assertFalse(E.Leq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).Evaluate({}).value)
        self.assertTrue(E.Leq(E.Const(V.Simple(0)), E.Const(V.Simple(1))).Evaluate({}).value)
        self.assertTrue(E.Leq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).Evaluate({}).value)

    def testGeq(self):
        self.assertTrue(E.Geq(E.Const(V.Simple(1)), E.Const(V.Simple(0))).Evaluate({}).value)
        self.assertFalse(E.Geq(E.Const(V.Simple(0)), E.Const(V.Simple(1))).Evaluate({}).value)
        self.assertTrue(E.Geq(E.Const(V.Simple(1)), E.Const(V.Simple(1))).Evaluate({}).value)
