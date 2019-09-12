
from ..language import values as V


class AbstractRange(V.Simple):
    """Base class for ranges in the protocol language.

    Handles enforcing the update of the range's value in the simulation environment's Python bindings dictionary.
    """

    def __init__(self, name):
        """Initialise the common range properties."""
        self.name = name
        self.count = -1
        self._value = float('nan')
        self.number_of_outputs = 0
        # Set an initial empty environment so calls to set our value in constructors don't fail
        # (since initialise hasn't been called by our simulation yet)
        from ..environment import Environment
        AbstractRange.initialise(self, Environment())

    def initialise(self, env):
        """Called by the associated simulation when its environment is initialised.

        Here we define the range variable within the environment.
        Subclasses should also evaluate any expressions used to define the range.
        """
        self.env = env
        env.define_name(self.name, self)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.env.unwrapped_bindings[self.name] = value

    @property
    def unwrapped(self):
        return self._value

    @property
    def current(self):
        return self._value

    def get_current_output_point(self):
        return self._value

    def get_current_output_number(self):
        return self.count

    def get_number_of_output_points(self):
        return self.number_of_outputs


class UniformRange(AbstractRange):
    def __init__(self, name, start_expr, end_expr, step_expr):
        super(UniformRange, self).__init__(name)
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.step_expr = step_expr

        try:
            line_profile.add_function(self.__next__)
        except NameError:
            pass

    def __iter__(self):
        self.count = -1
        self.value = self.start
        return self

    def __next__(self):
        self.count += 1
        if self.count >= self.number_of_outputs:
            self.count = -1
            raise StopIteration
        else:
            self.value = self.start + self.step * self.count
            return self.value

    def initialise(self, env):
        super(UniformRange, self).initialise(env)
        self.start = self.start_expr.evaluate(self.env).value
        self.step = self.step_expr.evaluate(self.env).value
        self.end = self.end_expr.evaluate(self.env).value
        self.value = self.start
        self.number_of_outputs = int((round(self.end - self.start) / self.step)) + 1


class VectorRange(AbstractRange):
    def __init__(self, name, arr_or_expr):
        super(VectorRange, self).__init__(name)
        if isinstance(arr_or_expr, V.Array):
            self.expr = None
            self.arr_range = arr_or_expr.array
            self.value = self.arr_range[0]
            self.number_of_outputs = len(self.arr_range)
        else:
            self.expr = arr_or_expr
            self.value = float('nan')
        self.count = -1

    def initialise(self, env):
        super(VectorRange, self).initialise(env)
        if self.expr:
            self.arr_range = self.expr.evaluate(env).array
            self.value = self.arr_range[0]
            self.number_of_outputs = len(self.arr_range)

    def __iter__(self):
        self.count = -1
        self.value = 0
        return self

    def __next__(self):
        self.count += 1
        if self.count >= self.number_of_outputs:
            self.count = -1
            raise StopIteration
        else:
            self.value = self.arr_range[self.count]
            return self.current


class While(AbstractRange):
    def __init__(self, name, condition):
        super(While, self).__init__(name)
        self.condition = condition
        self._init()

    def _init(self):
        """(Re-)Initialise the range loop."""
        self.count = -1
        self.value = -1
        self.number_of_outputs = 1000

    def initialise(self, env):
        super(While, self).initialise(env)
        self._init()

    def __iter__(self):
        self.count = -1
        self.value = -1
        return self

    def __next__(self):
        self.count += 1
        self.value += 1
        if self.count >= self.number_of_outputs:
            self.number_of_outputs += 1000
        if self.count > 0 and not self.condition.evaluate(self.env).value:
            self.number_of_outputs = self.count
            raise StopIteration
        else:
            return self.value
