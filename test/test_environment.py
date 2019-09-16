
import numpy as np
import unittest

import fc.language.values as V
import fc.environment as Env
from fc.error_handling import ProtocolError


class TestEnvironment(unittest.TestCase):
    """Test environment and delegations and associated functions."""

    def test_defining_names(self):
        env = Env.Environment()
        one = V.Simple(1)
        env.define_name("one", one)
        self.assertEqual(env.look_up("one"), one)
        self.assertEqual(len(env), 1)
        self.assertEqual(env.defined_names(), ["one"])
        self.assertRaises(ProtocolError, env.define_name, "one", 1)  # value must be a value type
        two = V.Simple(2)
        self.assertRaises(ProtocolError, env.define_name, "one", two)  # already defined
        self.assertRaises(ProtocolError, env.overwrite_definition, "one", two)  # don't have permission to overwrite
        names = ["two", "three", "four"]
        values = [V.Simple(2), V.Simple(3), V.Simple(4)]
        env.define_names(names, values)
        self.assertRaises(ProtocolError, env.define_names, names, values)  # already defined
        self.assertEqual(len(env), 4)
        for i, name in enumerate(names):
            self.assertEqual((env.look_up(names[i])), values[i])
        env2 = Env.Environment()
        env2.merge(env)
        fresh1 = env.fresh_ident()
        fresh2 = env2.fresh_ident()
        self.assertNotEqual(fresh1, fresh2)
        self.assertEqual(sorted(env.defined_names()), sorted(env2.defined_names()))
        env.clear()
        self.assertEqual(len(env), 0)
        env.define_name("one", one)
        self.assertEqual(env.look_up("one"), one)

    def test_overwriting_env(self):
        env = Env.Environment()
        one = V.Simple(1)
        env.define_name("one", one)
        self.assertEqual(env.look_up("one"), one)
        two = V.Simple(2)
        self.assertRaises(ProtocolError, env.define_name, "one", two)  # already defined
        self.assertRaises(ProtocolError, env.overwrite_definition, "one", two)  # don't have permission to overwrite
        self.assertRaises(ProtocolError, env.remove, "one")  # don't have permission to overwrite
        env.allow_overwrite = True
        env.overwrite_definition("one", two)
        self.assertEqual(env.look_up("one"), two)
        env.remove("one")
        self.assertRaises(KeyError, env.look_up, "one")  # item was removed
        env.define_name("one", one)
        self.assertEqual(env.look_up("one"), one)
        self.assertRaises(ProtocolError, env.remove, "three")  # never added

    def test_delegation(self):
        root_env = Env.Environment()
        middle_env = Env.Environment(delegatee=root_env)
        top_env = Env.Environment()
        top_env.set_delegatee_env(middle_env)

        name = "name"
        value = V.Simple(123.4)
        root_env.define_name(name, value)
        self.assertEqual(top_env.look_up(name), value)

        value2 = V.Simple(432.1)
        middle_env.define_name(name, value2)
        self.assertEqual(top_env.look_up(name), value2)

        value3 = V.Simple(6.5)
        top_env.define_name(name, value3)
        self.assertEqual(top_env.look_up(name), value3)

    def test_prefixed_delegation(self):
        root_env = Env.Environment()
        env_a = Env.Environment()
        env_b = Env.Environment()
        env_a.define_name("A", V.Simple(1))
        env_b.define_name("B", V.Simple(2))
        root_env.set_delegatee_env(env_a, 'a')
        root_env.set_delegatee_env(env_b, 'b')

        self.assertRaises(ProtocolError, env_a.define_name, 'a:b', V.Simple(1))
        self.assertEqual(root_env.look_up('a:A').value, 1)
        self.assertEqual(root_env.look_up('b:B').value, 2)
        self.assertRaises(KeyError, root_env.look_up, 'a:n')
        self.assertRaises(KeyError, root_env.look_up, 'c:c')

        env_aa = Env.Environment(delegatee=env_a)
        env_aa.define_name('A', V.Simple(3))
        env_a.set_delegatee_env(env_aa, 'a')
        self.assertEqual(env_a.look_up('a:A').value, 3)
        self.assertEqual(root_env.look_up('a:a:A').value, 3)

    def test_evaluate_expr_and_stmt(self):
        # basic mathml function
        env = Env.Environment()
        expr_str = 'MathML:max(100, 115, 98)'
        self.assertEqual(env.evaluate_expr(expr_str, env).value, 115)

        # array creation
        arr = V.Array(np.arange(10))
        env.define_name('arr', arr)
        expr_str = 'arr[1:2:10]'
        predicted = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_almost_equal(predicted, env.evaluate_expr(expr_str, env).array)

        # view of an array
        view_arr = V.Array(np.arange(10))
        env.define_name('view_arr', view_arr)
        expr_str = 'view_arr[4]'
        predicted = np.array(4)
        np.testing.assert_array_almost_equal(env.evaluate_expr(expr_str, env).array, predicted)

        # statement list
        stmt_str = 'z = lambda a: a+2\nassert z(2) == 4'
        env.evaluate_statement(stmt_str, env)  # assertion built into list, no extra test needed
