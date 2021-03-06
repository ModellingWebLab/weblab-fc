
from ...locatable import Locatable


class AbstractExpression(Locatable):
    """Base class for expressions in the protocol language."""

    def __init__(self, *children):
        """Create a new expression node, with a list of child expressions, possibly empty."""
        super(AbstractExpression, self).__init__()
        self.children = children

        try:
            line_profile.add_function(self.real_evaluate_compiled)
        except NameError:
            pass

    # Override Object serialization methods to allow pickling with the dill module
    def __getstate__(self):
        odict = self.__dict__.copy()
        # These properties cause namespace errors during pickling, and will be
        # automatically regenerated on first reference after unpickling.
        if '_compiled_function' in odict:
            del odict['_compiled_function']
        if '_eval_globals' in odict:
            del odict['_eval_globals']
        if '_defining_envs' in odict:
            del odict['_defining_envs']
        return odict

    def get_used_variables(self):
        """Get the set of (non-local) identifiers referenced within this expression."""
        result = set()
        for child in self.children:
            result |= child.get_used_variables()
        return result

    def evaluate_children(self, env):
        """Evaluate our child expressions and return a list of their values."""
        return [child.evaluate(env) for child in self.children]

    def interpret(self, env):
        """Evaluate this expression by interpreting the expression tree.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def compile(self, array_context=True):
        """Create a string representation of this expression for evaluation by numexpr or numpy.

        By default, evaluation of the generated string is only guaranteed to give correct results (as defined
        by the interpret method) in certain array contexts, for instance maps and array comprehensions.
        If array_context is set to False, the generated string will instead only evaluate correctly when
        considered as a standalone expression operating on simple values (useful in the SetVariable modifier,
        for instance).

        TODO: Always stop compilation succeeding if an expression or subexpression is traced.
        (Make trace on the base class a property, which when set replaces the compile method with one
        that always fails.)
        """
        raise NotImplementedError

    def evaluate(self, env):
        """We always default to using interpret for evaluating standalone expressions."""
        result = self.interpret(env)
        if self.trace:
            self.trace_value(result)
        return result

    def evaluate_compiled(self, env):
        """Try to evaluate the compiled version of this expression, returning an unwrapped Python value.

        This method only gets used on the first call.  If compilation fails, this method's name will be rebound
        to the normal evaluate method (or rather, a lambda which unwraps the result).  If compilation succeeds,
        the real_evaluate_compiled method will be used.
        """
        try:
            value = self.real_evaluate_compiled(env)
        except NotImplementedError:
            self.evaluate_compiled = lambda env: self.evaluate(env).value
            value = self.evaluate_compiled(env)
        else:
            self.evaluate_compiled = self.real_evaluate_compiled
        return value

    @property
    def compiled(self):
        """Compile this expression and cache the result."""
        try:
            return self._compiled
        except AttributeError:
            # We haven't called compile yet; cache the result
            c = self._compiled = self.compile(array_context=False)
            return c

    @property
    def eval_globals(self):
        try:
            return self._eval_globals
        except AttributeError:
            d = self._eval_globals = {}
            d['abs'] = abs
            import math
            for name in ['log', 'log10', 'exp']:
                d[name] = getattr(math, name)
            import numpy
            d['___np'] = numpy
            return d

    def real_evaluate_compiled(self, env):
        """Evaluate the compiled form of this expression using eval().

        This is suitable for use only in non-array contexts, such as by the SetVariable modifier.
        """
        func = self.compiled_function
        arg_envs = self.get_defining_environments(env)
        assert env is self._root_defining_env, "Internal implementation assumption violated"
        args = [arg_envs[name].unwrapped_bindings[name] for name in self._used_var_local_names]
        return func(*args)

    def get_defining_environments(self, env):
        """Cache which environment each variable used in this expression is actually defined in.

        Stores each environment indexed by the local name of the variable within that environment.

        TODO: Handle local name conflicts!
        """
        try:
            return self._defining_envs
        except AttributeError:
            self._root_defining_env = env  # For paranoia checking that the cache is valid
            d = self._defining_envs = {}
            l = self._used_var_local_names = []  # noqa: E741
            for name in self.used_variable_list:
                local_name = name[name.rfind(':') + 1:]
                l.append(local_name)
                d[local_name] = env.find_defining_environment(name)
            return d

    @property
    def compiled_function(self):
        """A version of self.compiled that has been converted to a Python function by eval()."""
        try:
            return self._compiled_function
        except AttributeError:
            from .general import NameLookUp
            arg_defs = ', '.join(NameLookUp.pythonize_name(name) for name in self.used_variable_list)
            f = self._compiled_function = eval('lambda ' + arg_defs + ': ' + self.compiled, self.eval_globals)
            return f

    @property
    def used_variable_list(self):
        """Cached property version of self.used_variables that's in a predictable order."""
        try:
            return self._used_var_list
        except AttributeError:
            l = self._used_var_list = list(self.used_variables)  # noqa: E741
            l.sort()
            return l

    @property
    def used_variables(self):
        """Cached property version of self.get_used_variables()."""
        try:
            return self._used_vars
        except AttributeError:
            # Create the cache
            u = self._used_vars = self.get_used_variables()
            return u

