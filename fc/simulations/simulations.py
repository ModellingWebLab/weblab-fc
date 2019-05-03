
import sys
import numpy as np


from operator import attrgetter

from . import ranges as R
from .model import NestedProtocol
from .modifiers import AbstractModifier
from ..language import values as V
from .. import environment as Env
from .. import locatable


class AbstractSimulation(locatable.Locatable):
    """Base class for simulations in the protocol language."""

    def __init__(self, prefix=None):
        super(AbstractSimulation, self).__init__()
        self.prefix = prefix
        self.ranges = [self.range_]
        self.model = None
        self.results = Env.Environment()
        self.resultsList = []  # An ordered view on the unwrapped versions of simulation results
        self.env = Env.Environment()
        self.viewEnv = None
        self.indent_level = 0

        try:
            line_profile.add_function(self.add_iteration_outputs)
            line_profile.add_function(self.loop_body_start_hook)
        except NameError:
            pass

    def __getstate__(self):
        # Must remove Model class and regenerate during unpickling
        # (Pickling errors from nested class structure of ModelWrapperEnvironment)

        # Undo Simulation.set_model
        if self.model is not None:
            modelenv = self.model.get_environment_map()
            for prefix in modelenv:
                if isinstance(self, Nested):
                    self.nestedSim.env.clear_delegatee_env(prefix)
                self.results.clear_delegatee_env(prefix)
                self.env.clear_delegatee_env(prefix)

        odict = self.__dict__.copy()
        odict['model'] = None
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)

    def initialise(self, initialiseRange=True):
        if initialiseRange:
            self.range_.initialise(self.env)
        if self.viewEnv is None and isinstance(self.range_, R.While) and self.prefix:
            # NB: We can't do this in the constructor as self.prefix may not be set until later
            self.viewEnv = Env.Environment(allow_overwrite=True)
            self.env.set_delegatee_env(self.viewEnv, self.prefix)

    def clear(self):
        self.env.clear()
        self.results.clear()
        self.resultsList[:] = []
        if self.viewEnv:
            self.viewEnv.clear()

    def set_indent_level(self, indent_level):
        """Set the level of indentation to use for progress output."""
        self.indent_level = indent_level
        if self.model:
            self.model.set_indent_level(indent_level)

    def log_progress(self, *args):
        """Print a progress line showing how far through the simulation we are.

        Arguments are converted to strings and space separated, as for the print builtin.
        """
        print('  ' * self.indent_level + ' '.join(map(str, args)))
        sys.stdout.flush()

    def internal_run(self, verbose=True):
        raise NotImplementedError

    def set_output_folder(self, folder):
        self.output_folder = folder
        if self.trace:
            self.model.set_output_folder(folder)

    def loop_body_start_hook(self):
        if (isinstance(self.range_, R.While) and
                self.range_.count > 0 and
                self.resultsList and
                self.range_.get_number_of_output_points() > self.resultsList[0].shape[0]):
            for name in self.results:
                result = self.results.look_up(name).array
                shape = list(result.shape)
                shape[0] = self.range_.get_number_of_output_points()
                result.resize(tuple(shape), refcheck=False)
                # TODO: Check if the next line is needed?
                self.viewEnv.overwrite_definition(name, V.Array(result[0:1 + self.range_.count]))
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.START_ONLY and self.range_.count == 0:
                modifier.apply(self)
            elif modifier.when == AbstractModifier.EACH_LOOP:
                modifier.apply(self)

    def loop_end_hook(self):
        if isinstance(self.range_, R.While):
            for name in self.results:
                result = self.results.look_up(name)
                # resize function doesn't work with references
                result.array = result.array[0:self.range_.get_number_of_output_points()]
        for modifier in self.modifiers:
            if modifier.when == AbstractModifier.END_ONLY:
                modifier.apply(self)

    def loop_body_end_hook(self):
        if self.viewEnv is not None:
            for name in self.results:
                if name not in self.viewEnv:
                    self.viewEnv.define_name(
                        name, V.Array(self.results.look_up(name).array[0:1 + self.range_.count]))
                else:
                    self.viewEnv.overwrite_definition(
                        name, V.Array(self.results.look_up(name).array[0:1 + self.range_.count]))

    def set_model(self, model):
        if isinstance(self.model, NestedProtocol):
            self.model.proto.set_model(model)
        else:
            self.model = model
        self.model.set_indent_level(self.indent_level)
        model_env = model.get_environment_map()
        # TODO: this breaks if a model is used in multiple simulations!  Only needed for NestedProtocol?
        model.sim_env = self.env
        for prefix, env in model_env.items():
            self.env.set_delegatee_env(env, prefix)
            self.results.set_delegatee_env(env, prefix)

    def run(self, verbose=True):
        try:
            self.internal_run(verbose)
        except Exception:
            # Shrink result arrays to reflect the number of iterations actually managed!
            if self.results is not None:
                for name in self.results:
                    result = self.results.look_up(name)
                    result.array = result.array[0:self.range_.count]
            raise
        return self.results

    def add_iteration_outputs(self, outputs_list):
        """Collect model outputs for this simulation step.

        Copy model outputs from one simulation step into the overall output arrays for the
        (possibly nested) simulation.
        """
        self_results, results_list = self.results, self.resultsList
        if self_results is not None:
            if isinstance(outputs_list, tuple):
                # Some simulation outputs were missing
                outputs_list, missing_outputs = outputs_list
            else:
                missing_outputs = []
            if not self_results:
                # First iteration - create empty output arrays of the correct shape
                range_dims = tuple(r.get_number_of_output_points() for r in self.ranges)
                for name, output in zip(self.model.output_names, outputs_list):
                    result = V.Array(np.empty(range_dims + output.shape))
                    self_results.define_name(name, result)
                    results_list.append(result.unwrapped)
            elif missing_outputs:
                for i, name in missing_outputs:
                    del results_list[i]
                    self_results.allow_overwrite = True
                    self_results.remove(name)
                    self_results.allow_overwrite = False
        if results_list:
            # Note that the tuple conversion in the next line is very quick
            range_indices = tuple(map(attrgetter('count'), self.ranges))  # tuple(r.count for r in self.ranges)
            for output, result in zip(outputs_list, results_list):
                result[range_indices] = output


class Timecourse(AbstractSimulation):
    """simulate a simple loop over time."""

    def __init__(self, range_, modifiers=[]):
        self.range_ = range_
        super(Timecourse, self).__init__()
        self.ranges = [self.range_]
        self.modifiers = modifiers

        try:
            line_profile.add_function(self.internal_run)
        except NameError:
            pass

    def internal_run(self, verbose=True):
        r = self.range_
        m = self.model
        start_hook, end_hook = self.loop_body_start_hook, self.loop_body_end_hook
        add_outputs, get_outputs = self.add_iteration_outputs, m.get_outputs
        set_time, simulate = m.set_free_variable, m.simulate
        for t in r:
            if r.count == 0:
                # Record initial conditions
                start_hook()
                set_time(t)
            else:
                # Loop through remaining time points.
                # Note that the start_hook is called *after* simulate in order to match the C++ implementation:
                # in effect it is the hook for the *next* iteration of the loop.
                simulate(t)
                start_hook()
            add_outputs(get_outputs())
            end_hook()
        self.loop_end_hook()


class OneStep(AbstractSimulation):
    """simulate one logical execution of a model."""

    class null_range(R.AbstractRange):
        pass

    def __init__(self, step, modifiers=[]):
        self.step = step
        self.modifiers = modifiers
        self.range_ = self.null_range('_')
        super(OneStep, self).__init__()
        self.ranges = []

    def internal_run(self, verbose=True):
        self.loop_body_start_hook()
        self.model.simulate(self.step)
        self.add_iteration_outputs(self.model.get_outputs())
        self.loop_end_hook()


class Nested(AbstractSimulation):
    """The main nested loop simulation construct."""

    def __init__(self, nestedSim, range_, modifiers=[]):
        self.range_ = range_
        super(Nested, self).__init__()
        self.nestedSim = nestedSim
        self.modifiers = modifiers
        self.ranges = self.nestedSim.ranges
        self.ranges.insert(0, self.range_)
        self.results = self.nestedSim.results
        self.resultsList = self.nestedSim.resultsList
        nestedSim.env.set_delegatee_env(self.env)

    def initialise(self):
        self.range_.initialise(self.env)
        self.nestedSim.initialise()
        if self.trace:
            self.nestedSim.trace = True
        super(Nested, self).initialise(initialiseRange=False)

    def clear(self):
        self.nestedSim.clear()
        super(Nested, self).clear()

    def set_indent_level(self, indent_level):
        super(Nested, self).set_indent_level(indent_level)
        self.nestedSim.set_indent_level(indent_level + 1)

    def internal_run(self, verbose=True):
        for t in self.range_:
            if verbose:
                self.log_progress(
                    'nested simulation',
                    self.range_.name,
                    'step',
                    self.range_.get_current_output_number(),
                    '(value',
                    self.range_.get_current_output_point(),
                    ')')
            self.loop_body_start_hook()
            if self.output_folder:
                self.nestedSim.set_output_folder(self.output_folder.create_subfolder(
                    'run_%d' % self.range_.get_current_output_number()))
            self.nestedSim.run()
            self.loop_body_end_hook()
        self.loop_end_hook()

    def set_model(self, model):
        super(Nested, self).set_model(model)
        self.nestedSim.set_model(model)
