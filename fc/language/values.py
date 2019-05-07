
import numpy as np

from ..error_handling import ProtocolError


class AbstractValue(object):
    """Base class for values in the protocol language."""

    def __init__(self, units=None):
        self.units = units

    @property
    def unwrapped(self):
        """Return the underlying Python value."""
        return None

    def __str__(self):
        """Return the string representation of the underlying Python value."""
        return repr(self.unwrapped)


class Simple(AbstractValue):
    """Simple value class in the protocol language for numbers."""

    def __init__(self, value):
        self.value = float(value)

    @property
    def array(self):
        """View this number as a 0-d array."""
        return np.array(self.value)

    @property
    def unwrapped(self):
        return self.value


class Array(AbstractValue):
    """Class in the protocol language for arrays."""

    def __init__(self, array):
        assert isinstance(array, (np.ndarray, np.float))
        self.array = np.array(array, dtype=float, copy=False)

    @property
    def value(self):
        if self.array.ndim == 0:
            return self.array[()]
        else:
            raise AttributeError("An array with", self.array.ndim, "dimensions cannot be treated as a single value.")

    @property
    def unwrapped(self):
        return self.array


class Tuple(AbstractValue):
    """Tuple class in the protocol language."""

    def __init__(self, *values):
        self.values = tuple(values)

    @property
    def unwrapped(self):
        return self.values


class Null(AbstractValue):
    """Null class in the protocol language."""

    def __str__(self):
        return "null"


class String(AbstractValue):
    """String class in the protocol language."""

    def __init__(self, value):
        self.value = value

    @property
    def unwrapped(self):
        return self.value


class DefaultParameter(AbstractValue):
    """Class in protocol language used for default values."""

    def __str__(self):
        return "default"


class LambdaClosure(AbstractValue):
    """Class for functions in the protocol language."""

    def __init__(self, defining_env, formal_parameters, body, default_parameters):
        self.formal_parameters = formal_parameters
        self.body = body
        self.default_parameters = default_parameters
        self.defining_env = defining_env

    def __str__(self):
        """Return a string representation of this function."""
        return "function" + str(tuple(self.formal_parameters))

    def compile(self, env, actual_parameters):
        from ..environment import Environment
        local_env = Environment(delegatee=self.defining_env)
        params = actual_parameters[:]
        if len(params) < len(self.formal_parameters):
            params.extend([DefaultParameter()] * (len(self.formal_parameters) - len(params)))
        for i, param in enumerate(params):
            if not isinstance(param, DefaultParameter):
                local_env.define_name(self.formal_parameters[i], param)
            elif (self.default_parameters[i] is not None
                  and not isinstance(self.default_parameters[i], DefaultParameter)):
                if not hasattr(self.default_parameters[i], 'value'):
                    raise NotImplementedError
                local_env.define_name(self.formal_parameters[i], self.default_parameters[i])
            else:
                raise ProtocolError("One of the parameters is not defined and has no default value")
        if len(self.body) == 1:
            expression = self.body[0].compile(env)
        return expression, local_env

    def evaluate(self, env, actual_parameters):
        from ..environment import Environment
        local_env = Environment(delegatee=self.defining_env)
        if len(actual_parameters) < len(self.formal_parameters):
            actual_parameters.extend([DefaultParameter()] * (len(self.formal_parameters) - len(actual_parameters)))
        for i, param in enumerate(actual_parameters):
            if not isinstance(param, DefaultParameter):
                local_env.define_name(self.formal_parameters[i], param)
            elif self.default_parameters[i] is not None:
                local_env.define_name(self.formal_parameters[i], self.default_parameters[i])
            else:
                raise ProtocolError("One of the parameters is not defined and has no default value")
        result = local_env.execute_statements(self.body, return_allowed=True)
        return result


class LoadFunction(LambdaClosure):
    """A built-in function for loading data files from disk.

    This gets inserted into the inputs environment under the name 'load'.
    """

    def __init__(self, base_path):
        """Initialise an instance of the load() built-in.

        :param base_path: path with respect to which to resolve relative data file paths.
        """
        self.base_path = base_path

    def __str__(self):
        """Return a string representation of this function."""
        return "load()"

    def compile(self, env, actual_parameters):
        raise NotImplementedError

    def evaluate(self, env, actual_parameters):
        """evaluate a load() function call.

        :param env: the environment within which to evaluate this call
        :param actual_parameters: the values of the parameters to the call; should be a single
            string value containing the path of the file to load
        :returns: an Array containing the file's data, if successful
        """
        if len(actual_parameters) != 1:
            raise ProtocolError("A load() call takes a single parameter, not %d." % len(actual_parameters))
        if not isinstance(actual_parameters[0], String):
            raise ProtocolError("A load() call takes a string parameter with the file path to load.")
        import os
        file_path = os.path.join(self.base_path, actual_parameters[0].value)
        from ..test_support import load2d
        return load2d(file_path)

