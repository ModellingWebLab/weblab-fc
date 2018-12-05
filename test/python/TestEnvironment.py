
import numpy as np
import unittest

import fc.language.values as V
import fc.utility.environment as Env
from fc.utility.error_handling import ProtocolError


class TestEnvironment(unittest.TestCase):
    """Test environment and delegations and associated functions."""

    def testDefiningNames(self):
        env = Env.Environment()
        one = V.Simple(1)
        env.DefineName("one", one)
        self.assertEqual(env.LookUp("one"), one)
        self.assertEqual(len(env), 1)
        self.assertEqual(env.DefinedNames(), ["one"])
        self.assertRaises(ProtocolError, env.DefineName, "one", 1)  # value must be a value type
        two = V.Simple(2)
        self.assertRaises(ProtocolError, env.DefineName, "one", two)  # already defined
        self.assertRaises(ProtocolError, env.OverwriteDefinition, "one", two)  # don't have permission to overwrite
        names = ["two", "three", "four"]
        values = [V.Simple(2), V.Simple(3), V.Simple(4)]
        env.DefineNames(names, values)
        self.assertRaises(ProtocolError, env.DefineNames, names, values)  # already defined
        self.assertEqual(len(env), 4)
        for i, name in enumerate(names):
            self.assertEqual((env.LookUp(names[i])), values[i])
        env2 = Env.Environment()
        env2.Merge(env)
        fresh1 = env.FreshIdent()
        fresh2 = env2.FreshIdent()
        self.assertNotEqual(fresh1, fresh2)
        self.assertEqual(sorted(env.DefinedNames()), sorted(env2.DefinedNames()))
        env.Clear()
        self.assertEqual(len(env), 0)
        env.DefineName("one", one)
        self.assertEqual(env.LookUp("one"), one)

    def testOverwritingEnv(self):
        env = Env.Environment()
        one = V.Simple(1)
        env.DefineName("one", one)
        self.assertEqual(env.LookUp("one"), one)
        two = V.Simple(2)
        self.assertRaises(ProtocolError, env.DefineName, "one", two)  # already defined
        self.assertRaises(ProtocolError, env.OverwriteDefinition, "one", two)  # don't have permission to overwrite
        self.assertRaises(ProtocolError, env.Remove, "one")  # don't have permission to overwrite
        env.allowOverwrite = True
        env.OverwriteDefinition("one", two)
        self.assertEqual(env.LookUp("one"), two)
        env.Remove("one")
        self.assertRaises(KeyError, env.LookUp, "one")  # item was removed
        env.DefineName("one", one)
        self.assertEqual(env.LookUp("one"), one)
        self.assertRaises(ProtocolError, env.Remove, "three")  # never added

    def testDelegation(self):
        root_env = Env.Environment()
        middle_env = Env.Environment(delegatee=root_env)
        top_env = Env.Environment()
        top_env.SetDelegateeEnv(middle_env)

        name = "name"
        value = V.Simple(123.4)
        root_env.DefineName(name, value)
        self.assertEqual(top_env.LookUp(name), value)

        value2 = V.Simple(432.1)
        middle_env.DefineName(name, value2)
        self.assertEqual(top_env.LookUp(name), value2)

        value3 = V.Simple(6.5)
        top_env.DefineName(name, value3)
        self.assertEqual(top_env.LookUp(name), value3)

    def testPrefixedDelegation(self):
        root_env = Env.Environment()
        env_a = Env.Environment()
        env_b = Env.Environment()
        env_a.DefineName("A", V.Simple(1))
        env_b.DefineName("B", V.Simple(2))
        root_env.SetDelegateeEnv(env_a, 'a')
        root_env.SetDelegateeEnv(env_b, 'b')

        self.assertRaises(ProtocolError, env_a.DefineName, 'a:b', V.Simple(1))
        self.assertEqual(root_env.LookUp('a:A').value, 1)
        self.assertEqual(root_env.LookUp('b:B').value, 2)
        self.assertRaises(KeyError, root_env.LookUp, 'a:n')
        self.assertRaises(KeyError, root_env.LookUp, 'c:c')

        env_aa = Env.Environment(delegatee=env_a)
        env_aa.DefineName('A', V.Simple(3))
        env_a.SetDelegateeEnv(env_aa, 'a')
        self.assertEqual(env_a.LookUp('a:A').value, 3)
        self.assertEqual(root_env.LookUp('a:a:A').value, 3)

    def testEvaluateExprAndStmt(self):
        # basic mathml function
        env = Env.Environment()
        expr_str = 'MathML:max(100, 115, 98)'
        self.assertEqual(env.EvaluateExpr(expr_str, env).value, 115)

        # array creation
        arr = V.Array(np.arange(10))
        env.DefineName('arr', arr)
        expr_str = 'arr[1:2:10]'
        predicted = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_almost_equal(predicted, env.EvaluateExpr(expr_str, env).array)

        # view of an array
        view_arr = V.Array(np.arange(10))
        env.DefineName('view_arr', view_arr)
        expr_str = 'view_arr[4]'
        predicted = np.array(4)
        np.testing.assert_array_almost_equal(env.EvaluateExpr(expr_str, env).array, predicted)

        # statement list
        stmt_str = 'z = lambda a: a+2\nassert z(2) == 4'
        env.EvaluateStatement(stmt_str, env)  # assertion built into list, no extra test needed
