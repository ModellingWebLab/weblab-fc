"""Copyright (c) 2005-2013, University of Oxford.
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

import unittest
import sys

# Import the module to test
import Environment as E
import Values as V

from ErrorHandling import ProtocolError

class TestEnvironment(unittest.TestCase):
    def TestDefiningNames(self):
        env = E.Environment()
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
        env2 = E.Environment()
        env2.Merge(env)
        fresh1 = env.FreshIdent()
        fresh2 = env2.FreshIdent()
        self.assertNotEqual(fresh1, fresh2)
        self.assertEqual(sorted(env.DefinedNames()),sorted(env2.DefinedNames()))
        env.Clear()
        self.assertEqual(len(env),0)
        env.DefineName("one", one)
        self.assertEqual(env.LookUp("one"), one)
                
    def TestOverwritingEnv(self):
        env = E.Environment()
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
        self.assertRaises(ProtocolError, env.LookUp, "one") # item was removed
        env.DefineName("one",one)
        self.assertEqual(env.LookUp("one"), one)
        self.assertRaises(ProtocolError, env.Remove, "three") # never added

"""void TestOverwritingEnv() throw (Exception)
    {

        AbstractValuePtr p_val2 = CV(432.1);
        TS_ASSERT_THROWS_CONTAINS(env.DefineName(name, p_val2, ""),
                                  "Name " + name + " is already defined and may not be re-bound.");
        env.OverwriteDefinition(name, p_val2, "");
        TS_ASSERT_EQUALS(env.Lookup(name), p_val2);
        TS_ASSERT_THROWS_CONTAINS(env.OverwriteDefinition("name2", p_val2, ""),
                                  "Name name2 is not defined and may not be overwritten.");

        env.RemoveDefinition(name, "");
        TS_ASSERT_THROWS_CONTAINS(env.Lookup(name), "Name " + name + " is not defined in this environment.");
        env.DefineName(name, p_val, "");
        TS_ASSERT_EQUALS(env.Lookup(name), p_val);
        TS_ASSERT_THROWS_CONTAINS(env.RemoveDefinition("name2", ""),
                                  "Name name2 is not defined and may not be removed.");
    }

class TestEnvironment : public CxxTest::TestSuite
{
public:
    void TestDefiningNames() throw (Exception)
    {
        TS_ASSERT_THROWS_CONTAINS(env.Lookup(env.FreshIdent()), " is not defined in this environment.");

        // Test debug tracing of environments
        DebugProto::TraceEnv(env);

    }
"""