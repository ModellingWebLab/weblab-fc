
import numpy as np
import unittest

import fc.language.expressions as E
import fc.language.values as V
import fc.simulations.model as Model
import fc.simulations.modifiers as Modifiers
import fc.simulations.ranges as Ranges
import fc.simulations.simulations as Simulations


class TestModelSimulation(unittest.TestCase):
    """Test models, simulations, ranges, and modifiers."""

    def test_simple_ODE(self):
        # using range made in python
        a = 5
        model = Model.TestOdeModel(a)
        for t in range(10):
            if t > 0:
                model.simulate(t)
            self.assertEqual(model.get_outputs()[model.output_names.index('a')], a)
            self.assertAlmostEqual(model.get_outputs()[model.output_names.index('y')], t * a)

    def test_uniform_range(self):
        a = 5
        model = Model.TestOdeModel(a)
        range_ = Ranges.UniformRange('count', E.N(0), E.N(10), E.N(1))
        time_sim = Simulations.Timecourse(range_)
        time_sim.initialise()
        time_sim.set_model(model)
        results = time_sim.run()
        np.testing.assert_array_almost_equal(results.look_up('a').array, np.array([5] * 11))
        np.testing.assert_array_almost_equal(results.look_up('y').array, np.array([t * 5 for t in range(11)]))

    def test_vector_range(self):
        a = 5
        model = Model.TestOdeModel(a)
        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)
        time_sim.initialise()
        time_sim.set_model(model)
        results = time_sim.run()
        np.testing.assert_array_almost_equal(results.look_up('a').array, np.array([5] * 11))
        np.testing.assert_array_almost_equal(results.look_up('y').array, np.array([t * 5 for t in range(11)]))

    def test_nested_simulations(self):
        a = 5
        model = Model.TestOdeModel(a)
        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)
        time_sim.set_model(model)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.initialise()
        nested_sim.set_model(model)
        results = nested_sim.run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(50, 101, 5),
                              np.arange(100, 151, 5), np.arange(150, 201, 5)])
        np.testing.assert_array_almost_equal(predicted, results.look_up('y').array)

    def test_reset(self):
        a = 5
        model = Model.TestOdeModel(a)
        when = Modifiers.AbstractModifier.START_ONLY
        modifier = Modifiers.ResetState(when)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_, modifiers=[modifier])

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.initialise()
        nested_sim.set_model(model)
        results = nested_sim.run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5)])
        np.testing.assert_array_almost_equal(predicted, results.look_up('y').array)

        # reset at each loop with modifier on nested simul, should be same result as above
        a = 5
        model = Model.TestOdeModel(a)
        when = Modifiers.AbstractModifier.EACH_LOOP
        modifier = Modifiers.ResetState(when)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_, modifiers=[modifier])
        nested_sim.initialise()
        nested_sim.set_model(model)
        results = nested_sim.run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5)])
        np.testing.assert_array_almost_equal(predicted, results.look_up('y').array)

    def test_save_and_reset(self):
        # save state and reset using save state
        a = 5
        model = Model.TestOdeModel(a)
        save_modifier = Modifiers.SaveState(Modifiers.AbstractModifier.START_ONLY, 'start')
        reset_modifier = Modifiers.ResetState(Modifiers.AbstractModifier.EACH_LOOP, 'start')
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)

        range_ = Ranges.VectorRange('count', V.Array(np.array([1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_, modifiers=[save_modifier, reset_modifier])
        nested_sim.initialise()
        nested_sim.set_model(model)
        results = nested_sim.run()
        predicted = np.array([np.arange(0, 51, 5), np.arange(0, 51, 5), np.arange(0, 51, 5)])
        np.testing.assert_array_almost_equal(predicted, results.look_up('y').array)

        # save state and reset using save state
        a = 5
        model = Model.TestOdeModel(a)
        save_modifier = Modifiers.SaveState(Modifiers.AbstractModifier.END_ONLY, 'start')
        reset_modifier = Modifiers.ResetState(Modifiers.AbstractModifier.EACH_LOOP, 'start')
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        initial_time_sim = Simulations.Timecourse(range_, modifiers=[save_modifier])
        initial_time_sim.initialise()
        initial_time_sim.set_model(model)
        initial_time_sim.run()

        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        inner_time_sim = Simulations.Timecourse(range_)
        range_ = Ranges.VectorRange('range', V.Array(np.array([1, 2, 3])))
        nested_sim = Simulations.Nested(inner_time_sim, range_, modifiers=[reset_modifier])
        nested_sim.initialise()
        nested_sim.set_model(model)
        results = nested_sim.run()
        predicted = np.array([np.arange(50, 101, 5), np.arange(50, 101, 5), np.arange(50, 101, 5)])
        np.testing.assert_array_almost_equal(predicted, results.look_up('y').array)

    def test_set_variable(self):
        a = 5
        model = Model.TestOdeModel(a)
        modifier = Modifiers.SetVariable(Modifiers.AbstractModifier.START_ONLY, 'oxmeta:leakage_current', E.N(1))
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        time_sim = Simulations.Timecourse(range_)

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_, modifiers=[modifier])
        nested_sim.initialise()
        nested_sim.set_model(model)
        results = nested_sim.run()
        predicted = np.array([np.arange(0, 11), np.arange(10, 21), np.arange(20, 31), np.arange(30, 41)])
        np.testing.assert_array_almost_equal(predicted, results.look_up('y').array)

    def test_set_variable_with_range(self):
        a = 5
        model = Model.TestOdeModel(a)
        set_modifier = Modifiers.SetVariable(
            Modifiers.AbstractModifier.START_ONLY,
            'oxmeta:leakage_current',
            E.NameLookUp('count'))
        reset_modifier = Modifiers.ResetState(Modifiers.AbstractModifier.START_ONLY)
        range_ = Ranges.VectorRange('range', V.Array(np.array([0, 1, 2, 3])))
        time_sim = Simulations.Timecourse(range_, modifiers=[set_modifier, reset_modifier])

        range_ = Ranges.VectorRange('count', V.Array(np.array([0, 1, 2, 3])))
        nested_sim = Simulations.Nested(time_sim, range_)
        nested_sim.initialise()
        nested_sim.set_model(model)
        results = nested_sim.run()
        predicted = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 6, 9]])
        np.testing.assert_array_almost_equal(predicted, results.look_up('y').array)

    def test_whiles(self):
        a = 10
        model = Model.TestOdeModel(a)
        while_range = Ranges.While('while', E.Lt(E.NameLookUp('while'), E.N(10)))
        time_sim = Simulations.Timecourse(while_range)
        time_sim.set_model(model)
        time_sim.initialise()
        results = time_sim.run()
        predicted = np.arange(0, 100, 10)
        actual = results.look_up('y').array
        np.testing.assert_array_almost_equal(predicted, actual)
