
import unittest

import fc.language.expressions as E
import fc.language.values as V


class TestLogicOperators(unittest.TestCase):
    """Tests logic using simple values. Tests and, or, xor, not."""

    def testAnd(self):
        self.assertEqual(E.And(E.Const(V.Simple(1)), E.Const(V.Simple(0))).Evaluate({}).value, 0)
        self.assertEqual(E.And(E.Const(V.Simple(1)), E.Const(V.Simple(1)), E.Const(V.Simple(1))).Evaluate({}).value, 1)

    def testOr(self):
        self.assertEqual(E.Or(E.Const(V.Simple(1)), E.Const(V.Simple(0))).Evaluate({}).value, 1)
        self.assertEqual(E.Or(E.Const(V.Simple(0)), E.Const(V.Simple(0)), E.Const(V.Simple(0))).Evaluate({}).value, 0)

    def testXor(self):
        self.assertEqual(E.Xor(E.Const(V.Simple(1)), E.Const(V.Simple(0))).Evaluate({}).value, 1)
        self.assertEqual(E.Xor(E.Const(V.Simple(0)), E.Const(V.Simple(0))).Evaluate({}).value, 0)
        self.assertEqual(E.Xor(E.Const(V.Simple(1)), E.Const(V.Simple(1))).Evaluate({}).value, 0)

    def testNot(self):
        self.assertEqual(E.Not(E.Const(V.Simple(1))).Evaluate({}).value, 0)
        self.assertEqual(E.Not(E.Const(V.Simple(3))).Evaluate({}).value, 0)
        self.assertEqual(E.Not(E.Const(V.Simple(0))).Evaluate({}).value, 1)
