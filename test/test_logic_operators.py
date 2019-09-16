
import unittest

import fc.language.expressions as E
import fc.language.values as V


class TestLogicOperators(unittest.TestCase):
    """Tests logic using simple values. Tests and, or, xor, not."""

    def test_and(self):
        self.assertEqual(E.And(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value, 0)
        self.assertEqual(E.And(E.Const(V.Simple(1)), E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value, 1)

    def test_or(self):
        self.assertEqual(E.Or(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value, 1)
        self.assertEqual(E.Or(E.Const(V.Simple(0)), E.Const(V.Simple(0)), E.Const(V.Simple(0))).evaluate({}).value, 0)

    def test_xor(self):
        self.assertEqual(E.Xor(E.Const(V.Simple(1)), E.Const(V.Simple(0))).evaluate({}).value, 1)
        self.assertEqual(E.Xor(E.Const(V.Simple(0)), E.Const(V.Simple(0))).evaluate({}).value, 0)
        self.assertEqual(E.Xor(E.Const(V.Simple(1)), E.Const(V.Simple(1))).evaluate({}).value, 0)

    def test_not(self):
        self.assertEqual(E.Not(E.Const(V.Simple(1))).evaluate({}).value, 0)
        self.assertEqual(E.Not(E.Const(V.Simple(3))).evaluate({}).value, 0)
        self.assertEqual(E.Not(E.Const(V.Simple(0))).evaluate({}).value, 1)
