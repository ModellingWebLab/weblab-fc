"""Copyright (c) 2005-2015, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import unittest

import fc.language.values as V
import fc.utility.environment as Env
from fc.utility.error_handling import ProtocolError


class TestEnvironment(unittest.TestCase):
    """Test environment and delegations and associated functions."""
    def TestDefiningNames(self):
        env = Env.Environment()
        one = V.Simple(1)
        env.DefineName("one", one)
        self.assertEqual(env.LookUp("one"), one)
        self.assertEqual(len(env), 1)
        self.assertEqual(env.DefinedNames(), ["one"])
        self.assertRaises(ProtocolError, env.DefineName, "one", 1) # value must be a value type
        two = V.Simple(2)
        self.assertRaises(ProtocolError, env.DefineName, "one", two) # already defined
        self.assertRaises(ProtocolError, env.OverwriteDefinition, "one", two) # don't have permission to overwrite
        names = ["two", "three", "four"]
        values = [V.Simple(2), V.Simple(3), V.Simple(4)]
        env.DefineNames(names,values)
        self.assertRaises(ProtocolError, env.DefineNames, names, values) # already defined
        self.assertEqual(len(env), 4)
        for i,name in enumerate(names):
            self.assertEqual((env.LookUp(names[i])),values[i])
        env2 = Env.Environment()
        env2.Merge(env)
        fresh1 = env.FreshIdent()
        fresh2 = env2.FreshIdent()
        self.assertNotEqual(fresh1, fresh2)
        self.assertEqual(sorted(env.DefinedNames()), sorted(env2.DefinedNames()))
        env.Clear()
        self.assertEqual(len(env),0)
        env.DefineName("one", one)
        self.assertEqual(env.LookUp("one"), one)

    def TestOverwritingEnv(self):
        env = Env.Environment()
        one = V.Simple(1)
        env.DefineName("one", one)
        self.assertEqual(env.LookUp("one"), one)
        two = V.Simple(2)
        self.assertRaises(ProtocolError, env.DefineName, "one", two) # already defined
        self.assertRaises(ProtocolError, env.OverwriteDefinition, "one", two) # don't have permission to overwrite
        self.assertRaises(ProtocolError, env.Remove, "one") # don't have permission to overwrite 
        env.allowOverwrite = True
        env.OverwriteDefinition("one", two)
        self.assertEqual(env.LookUp("one"), two)
        env.Remove("one")
        self.assertRaises(KeyError, env.LookUp, "one") # item was removed
        env.DefineName("one",one)
        self.assertEqual(env.LookUp("one"), one)
        self.assertRaises(ProtocolError, env.Remove, "three") # never added

    def TestDelegation(self):
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

    def TestPrefixedDelegation(self):
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

    def TestEvaluateExprAndStmt(self):
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
        env.EvaluateStatement(stmt_str, env) # assertion built into list, no extra test needed
