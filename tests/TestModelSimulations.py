
import numpy as np
import unittest

import fc.language.expressions as E
import fc.language.values as V
import fc.simulations.model as Model
import fc.simulations.modifiers as Modifiers
import fc.simulations.ranges as Ranges
import fc.simulations.simulations as Simulations

N = E.N


class TestModelSimulation(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""

    def testSimpleODE(self):
        # using range made in python
        a = 5
        model = Model.TestOdeModel(a)
        for t in range(10):
            if t > 0:
                model.Simulate(t)
            self.assertEqual(model.GetOutputs()[model.outputNames.index('a')], a)
            self.assertAlmostEqual(model.GetOutputs()[model.outputNames.index('y')], t * a)

    def testUniformRange(self):
        a = 5
        model = Model.TestOdeModel(a)
        range_ = Ranges.UniformRange('count', N(0), N(10), N(1))
        time_sim = Simulations.Timecourse(range_)
        time_sim.Initialise()
        time_sim.SetModel(model)
        results = time_sim.Run()
        np.testing.assert_array_almost_equal(results.LookUp('a').array, np.array([5] * 11))
        np.testing.assert_array_almost_equal(results.LookUp('y').array, np.array([t * 5 for t in range(11)]))

    def testVectorRange(self):
        a = 5
        model = Model.TestOdeModel(a)
        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)
        time_sim.Initialise()
        time_sim.SetModel(model)
        results = time_sim.Run()
        np.testing.assert_array_almost_equal(results.LookUp('a').array, np.array([5] * 11))
        np.testing.assert_array_almost_equal(results.LookUp('y').array, np.array([t * 5 for t in range(11)]))

    def testNestedSimulations(self):
        a = 5
        model = Model.TestOdeModel(a)
        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)
        time_sim.SetModel(model)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.Initialise()
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(50, 101, 5),
                              np.arange(100, 151, 5), np.arange(150, 201, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)

    def testReset(self):
        a = 5
        model = Model.TestOdeModel(a)
        when = Modifiers.AbstractModifier.START_ONLY
        modifier = Modifiers.ResetState(when)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_, modifiers=[modifier])

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.Initialise()
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)

        # reset at each loop with modifier on nested simul, should be same result as above
        a = 5
        model = Model.TestOdeModel(a)
        when = Modifiers.AbstractModifier.EACH_LOOP
        modifier = Modifiers.ResetState(when)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_, modifiers=[modifier])
        nested_sim.Initialise()
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)

    def testSaveAndReset(self):
        # save state and reset using save state
        a = 5
        model = Model.TestOdeModel(a)
        save_modifier = Modifiers.SaveState(Modifiers.AbstractModifier.START_ONLY, 'start')
        reset_modifier = Modifiers.ResetState(Modifiers.AbstractModifier.EACH_LOOP, 'start')
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)

        range_ = Ranges.VectorRange('count', V.Array(np.array([1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_, modifiers=[save_modifier, reset_modifier])
        nested_sim.Initialise()
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)

        # save state and reset using save state
        a = 5
        model = Model.TestOdeModel(a)
        save_modifier = Modifiers.SaveState(Modifiers.AbstractModifier.END_ONLY, 'start')
        reset_modifier = Modifiers.ResetState(Modifiers.AbstractModifier.EACH_LOOP, 'start')
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        initial_time_sim = Simulations.Timecourse(range_, modifiers=[save_modifier])
        initial_time_sim.Initialise()
        initial_time_sim.SetModel(model)
        initial_time_sim.Run()

        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        inner_time_sim = Simulations.Timecourse(range_)
        range_ = Ranges.VectorRange('range', V.Array(np.array([1, 2, 3])))
        nested_sim = Simulations.Nested(inner_time_sim, range_, modifiers=[reset_modifier])
        nested_sim.Initialise()
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(50, 101, 5), np.arange(50, 101, 5), np.arange(50, 101, 5)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)

    def testSetVariable(self):
        # set variable
        a = 5
        model = Model.TestOdeModel(a)
        modifier = Modifiers.SetVariable(Modifiers.AbstractModifier.START_ONLY, 'oxmeta:leakage_current', N(1))
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_, modifiers=[modifier])
        nested_sim.Initialise()
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([np.arange(0, 11), np.arange(10, 21), np.arange(20, 31), np.arange(30, 41)])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)

    def testSetWithRange(self):
        a = 5
        model = Model.TestOdeModel(a)
        set_modifier = Modifiers.SetVariable(Modifiers.AbstractModifier.START_ONLY,
                                             'oxmeta:leakage_current', E.NameLookUp('count'))
        reset_modifier = Modifiers.ResetState(Modifiers.AbstractModifier.START_ONLY)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3])))
        time_sim = Simulations.Timecourse(range_, modifiers=[set_modifier, reset_modifier])

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.Initialise()
        nested_sim.SetModel(model)
        results = nested_sim.Run()
        predicted = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 6, 9]])
        np.testing.assert_array_almost_equal(predicted, results.LookUp('y').array)

    def testWhiles(self):
        a = 10
        model = Model.TestOdeModel(a)
        while_range = Ranges.While('while', E.Lt(E.NameLookUp('while'), N(10)))
        time_sim = Simulations.Timecourse(while_range)
        time_sim.SetModel(model)
        time_sim.Initialise()
        results = time_sim.Run()
        predicted = np.arange(0, 100, 10)
        actual = results.LookUp('y').array
        np.testing.assert_array_almost_equal(predicted, actual)
