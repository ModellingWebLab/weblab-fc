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

try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
from Modifiers import AbstractModifier
import Environment as Env
import Expressions as E
import MathExpressions as M
from Model import TestOdeModel
import Modifiers
import numpy as np
import Protocol
from ErrorHandling import ProtocolError
import Ranges
import Simulations
import Values as V

def N(number):
    return M.Const(V.Simple(number))

class TestModelSimulation(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""
    def TestSimpleODE(self):
        # using range made in python
        a = 5
        model = TestOdeModel(a)
        for t in range(10):
            if t > 0:
                model.Simulate(t)
            self.assertEqual(model.GetOutputs().LookUp('a').value, a)
            self.assertAlmostEqual(model.GetOutputs().LookUp('y').value, t*a)
        
    def TestUniformRange(self): 
        a = 5
        model = TestOdeModel(a)
        range_ = Ranges.UniformRange('count', N(0), N(10), N(1))
        time_sim = Simulations.Timecourse(range_)
        time_sim.Initialise()       
        time_sim.SetModel(model)
        results = time_sim.Run()
        np.testing.assert_array_almost_equal(results.LookUp('a').array, np.array([5]*11))
        np.testing.assert_array_almost_equal(results.LookUp('y').array, np.array([t*5 for t in range(11)]))   
     
    def TestVectorRange(self):    
        a = 5
        model = TestOdeModel(a)
        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)
        time_sim.Initialise()       
        time_sim.SetModel(model)
        results = time_sim.Run()
        np.testing.assert_array_almost_equal(results.LookUp('a').array, np.array([5]*11))
        np.testing.assert_array_almost_equal(results.LookUp('y').array, np.array([t*5 for t in range(11)]))   
        
    def TestNestedSimulations(self):
        a = 5
        model = TestOdeModel(a)
        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)
        time_sim.Initialise()       
        time_sim.SetModel(model)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.Initialise()       
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(50, 101, 5), np.arange(100, 151, 5), np.arange(150, 201, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)
        
        # test while loop
        
    def TestReset(self):                
        a = 5
        model = TestOdeModel(a)
        when = AbstractModifier.START_ONLY
        modifier = Modifiers.ResetState(when)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_, modifiers=[modifier])
        time_sim.Initialise()       

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.Initialise() 
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)
        
        # reset at each loop with modifier on nested simul, should be same result as above
        a = 5
        model = TestOdeModel(a)
        when = AbstractModifier.EACH_LOOP
        modifier = Modifiers.ResetState(when)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)
        time_sim.Initialise()       
          
        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_, modifiers=[modifier])
        nested_sim.Initialise() 
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)
        
    def TestSaveAndReset(self): 
        # save state and reset using save state
        a = 5
        model = TestOdeModel(a)
        save_modifier = Modifiers.SaveState(AbstractModifier.START_ONLY, 'start')
        reset_modifier = Modifiers.ResetState(AbstractModifier.EACH_LOOP, 'start')
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)     
        time_sim.Initialise()         

        range_ = Ranges.VectorRange('count', V.Array(np.array([1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_, modifiers=[save_modifier, reset_modifier])
        nested_sim.Initialise() 
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)
        
        # save state and reset using save state
        a = 5
        model = TestOdeModel(a)
        save_modifier = Modifiers.SaveState(AbstractModifier.END_ONLY, 'start')
        reset_modifier = Modifiers.ResetState(AbstractModifier.EACH_LOOP, 'start')
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        initial_time_sim = Simulations.Timecourse(range_, modifiers = [save_modifier])
        initial_time_sim.Initialise()       
        initial_time_sim.SetModel(model)
        initial_time_sim.Run()
        
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        inner_time_sim = Simulations.Timecourse(range_)
        inner_time_sim.Initialise()       
        range_ = Ranges.VectorRange('range', V.Array(np.array([1, 2, 3])))
        nested_sim = Simulations.Nested(inner_time_sim, range_, modifiers=[reset_modifier])
        nested_sim.Initialise() 
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(50, 101, 5), np.arange(50, 101, 5), np.arange(50, 101, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)
        
    def TestSetVariable(self): 
        # set variable
        a = 5
        model = TestOdeModel(a)
        modifier = Modifiers.SetVariable(AbstractModifier.START_ONLY, 'oxmeta:leakage_current', N(1))
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)       
        time_sim.Initialise()       

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_, modifiers=[modifier])
        nested_sim.Initialise() 
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 11), np.arange(10, 21), np.arange(20, 31), np.arange(30, 41)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)
        
    def TestSetWithRange(self):
        a = 5
        model = TestOdeModel(a)
        set_modifier = Modifiers.SetVariable(AbstractModifier.START_ONLY, 'oxmeta:leakage_current', E.NameLookUp('count'))
        reset_modifier = Modifiers.ResetState(AbstractModifier.START_ONLY)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3])))
        time_sim = Simulations.Timecourse(range_, modifiers=[set_modifier, reset_modifier])
        time_sim.Initialise()       

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.Initialise() 
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 6, 9]])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)
         
    def TestWhiles(self):
        a = 10
        model = TestOdeModel(a)
        while_range = Ranges.While('while', M.Lt(E.NameLookUp('while'), N(10)))
        time_sim = Simulations.Timecourse(while_range)
        time_sim.SetModel(model)
        time_sim.Initialise()
        results = time_sim.Run()
        predicted = np.arange(0, 100, 10)
        actual = results.LookUp('y').array
        np.testing.assert_array_almost_equal(predicted, actual)
        
    def TestPyCmlLuoRudy(self):
        import tempfile, subprocess, sys, imp
        import subprocess
        dir = tempfile.mkdtemp()
        test_while = 'projects/FunctionalCuration/test/protocols/compact/test_while_loop.txt'
        xml_file = subprocess.check_output(['python', 'projects/FunctionalCuration/src/proto/parsing/CompactSyntaxParser.py', test_while, dir])
        xml_file = xml_file.strip()
        class_name = 'GeneratedModel'
        code = subprocess.check_output(['./python/pycml/translate.py', '-t', 'Python', '-p', '--Wu', '--protocol=' + xml_file, 'projects/FunctionalCuration/cellml/luo_rudy_1991.cellml', '-c', class_name, '-o', '-'])
        module = imp.new_module(class_name)
        exec code in module.__dict__
        for name in module.__dict__.keys():
            if name.startswith(class_name):
                model = getattr(module, name)()
        proto = Protocol.Protocol(test_while)
        proto.SetModel(model)
        proto.SetInput('num_iters', N(10))
        proto.Run()
        
#     def TestPyCmlLuoRudyStringModel(self):
#         # shorter test after properly implementing set model into protocol
#         proto_file = 'projects/FunctionalCuration/test/protocols/compact/test_while_loop.txt'
#         proto = Protocol.Protocol(proto_file)
#         model = 'luo_rudy_1991.cellml'
#         print 'proto', proto
#         proto.SetModel(model)
#         proto.SetInput('num_iters', N(10))
#         proto.Run()
        
        #courtemanche_ramirez_nattel_1998
#     def TestPyCMLRealProto(self):
#         import tempfile, subprocess, sys, imp
#         import subprocess
#         dir = tempfile.mkdtemp()
#         test_while = 'projects/FunctionalCuration/test/protocols/compact/test_sim_environments.txt'
#         xml_file = subprocess.check_output(['python', 'projects/FunctionalCuration/src/proto/parsing/CompactSyntaxParser.py', test_while, dir])
#         xml_file = xml_file.strip()
#         class_name = 'GeneratedModel'
#         code = subprocess.check_output(['./python/pycml/translate.py', '-t', 'Python', '-p', '--Wu', '--protocol=' +xml_file, 'projects/FunctionalCuration/cellml/courtemanche_ramirez_nattel_1998', '-c', class_name, '-o', '-'])
#         module = imp.new_module(class_name)
#         exec code in module.__dict__
#         for name in module.__dict__.keys():
#             if name.startswith(class_name):
#                 model = getattr(module, name)()
#         proto = Protocol.Protocol(test_while)
#         proto.SetModel(model)
#         proto.Run()
         
             