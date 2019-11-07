import numpy as np

from .error_handling import ProtocolError
from .language import values as V


class Environment(object):
    """
    Base class for environments in the protocol language.

    In the run-time implementation of a protocol many environments exist,
    mapping names to the corresponding values. Any environment may associate
    another with a particular prefix, and names starting with that prefix are
    looked up in that environment. As well as for simulation results, this
    mechanism is used to refer to names from imported protocols, and also to
    refer to model variables. In the latter case we utilise ontology terms,
    i.e. URIs, and the prefix to use is given by the namespace bindings in the
    protocol definition.

    An environment may also have a default delegatee, which is used to look up
    any name not found within the environment. Understanding the delegations
    occurring between environments is thus important for understanding what
    variable names to use if you want to look up a particular entity. The
    delegation graph is shown in the schematic on the right.

    The following environments exist.

    - An environment containing the protocol inputs.
    - An environment containing the definitions in the protocol library.
    - An environment for names defined in the post-processing section.
    - Each simulation has an associated environment as it runs, which is used
      for evaluating expressions in ranges and setVariable modifiers. The
      environment itself contains only the current value of the associated
      range(s). However it delegates to many useful environments. In a nested
      simulation, note that the environment for each nested level delegates by
      default to the next level out.
    - Environments containing the results of each simulation. These always
      exist, but only contain values once the simulation has run.
      - The environments delegate by (ontology) prefix to the environments
        containing the current values of variables in the model that will be
        simulated.
      - The names of results within these environments use the local names of
        variable annotations, i.e. with the base URI for whatever ontology
        stripped out. This may change; see #2529.
    - An environment containing the declared outputs of the protocol. Its
      contents are copied from the post-processing environment once the
      protocol has run - it does not delegate to any environment. Any variables
      to be plotted must appear in this environment.
    - Function definitions define a local environment for the function body,
      which delegates to the statically enclosing scope (i.e. the environment
      where the function was defined). Note that function parameters get
      evaluated in the environment of the caller of the function, however.

    """
    next_ident = [0]

    def __init__(self, allow_overwrite=False, delegatee=None):
        self.allow_overwrite = allow_overwrite
        self.bindings = DelegatingDict()
#         self.bindings._env = self
        self.unwrapped_bindings = DelegatingDict()
        self.unwrapped_bindings['___np'] = np
        self.delegatees = {}
        if delegatee is not None:
            self.set_delegatee_env(delegatee, "")

        try:
            line_profile.add_function(self.define_name)
            line_profile.add_function(self.find_defining_environment)
        except NameError:
            pass

    # Python requires __setstate__ (which restores the numpy module untracked by
    # DelegatingDict.__getstate()) to be paired with a (dummy) __getstate__() method
    def __getstate__(self):
        odict = self.__dict__
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self.unwrapped_bindings['___np'] = np

    def define_name(self, name, value):
        if ':' in name:
            raise ProtocolError('Names such as', name, 'with a colon are not allowed.')
        if name in self.bindings:
            raise ProtocolError(name, 'is already defined as', self.bindings[name],
                                'and may not be re-bound')
        else:
            self.bindings[name] = value
            self.unwrapped_bindings[name] = value.unwrapped

    def define_names(self, names, values):
        for name, value in zip(names, values):
            self.define_name(name, value)

    def evaluate_expr(self, expr_str, env):
        from fc.parsing.CompactSyntaxParser import CompactSyntaxParser as csp

        parse_action = csp.expr.parseString(expr_str, parseAll=True)
        expr = parse_action[0].expr()
        return expr.evaluate(env)

    def evaluate_statement(self, stmt_str, env):
        from fc.parsing.CompactSyntaxParser import CompactSyntaxParser as csp

        parse_action = csp.stmt_list.parseString(stmt_str, parseAll=True)
        stmt_list = parse_action[0].expr()
        return env.execute_statements(stmt_list)

    @staticmethod
    def fresh_ident():
        Environment.next_ident[0] += 1
        return "___%d" % Environment.next_ident[0]

    def look_up(self, name):
        return self.bindings[name]
#         try:
#             return self.bindings[name]
#         except KeyError:
#             print 'Key error looking up', name, 'in', self
#             import sys
#             tb = sys.exc_info()[2]
#             while tb:
#                 local_vars = tb.tb_frame.f_locals
#                 obj = local_vars.get('self', None)
#                 if obj and isinstance(obj, DelegatingDict):
#                     print 'Looked for', local_vars['key'], 'in', obj._env
#                 tb = tb.tb_next
#             self.debug_delegatees('root')
#             raise

    def debug_delegatees(self, prefix):
        print('Delegatees in', prefix, '(', len(self), '):', self.delegatees)
        for p, env in self.delegatees.items():
            env.debug_delegatees(p)

    def set_delegatee_env(self, delegatee, prefix=""):
        if prefix in self.delegatees and self.delegatees[prefix] is not delegatee:
            raise ProtocolError(
                "The name prefix '" + prefix +
                "' has already been used in this context. Check your simulations, imports, etc.")
        self.delegatees[prefix] = delegatee
#         print 'Delegating to', delegatee, 'for', prefix, 'in', self
        self.bindings.set_delegatee(delegatee.bindings, prefix)
        self.unwrapped_bindings.set_delegatee(delegatee.unwrapped_bindings, prefix)

    def clear_delegatee_env(self, prefix):
        if prefix in self.delegatees:
            del self.delegatees[prefix]
            del self.bindings.delegatees[prefix]
            del self.unwrapped_bindings.delegatees[prefix]

    def merge(self, env):
        self.define_names(env.bindings.keys(), env.bindings.values())

    def remove(self, name):
        if not self.allow_overwrite:
            raise ProtocolError("This environment does not support overwriting mappings")
        if name not in self.bindings:
            raise ProtocolError(
                name, "is not defined in this environment and thus cannot be removed")
        del self.bindings[name]
        del self.unwrapped_bindings[name]

    def find_defining_environment(self, name):
        """Find which environment the given name is defined in, whether this or one of its delegatees.

        Returns None if the name isn't defined anywhere.
        """
        if name in self.bindings:
            return self
        parts = name.split(':', 1)
        if len(parts) == 2:
            prefix, local_name = parts
        else:
            prefix, local_name = '', name
        try:
            return self.delegatees[prefix].find_defining_environment(local_name)
        except KeyError:
            try:
                return self.delegatees[''].find_defining_environment(name)
            except KeyError:
                return None

    def overwrite_definition(self, name, value):
        """Change the binding of name to value, if this is permitted by the defining environment."""
        defining_env = self.find_defining_environment(name)
        if defining_env is None:
            raise ProtocolError(
                name,
                'is not defined in this env or a delegating env and thus cannot be overwritten')
        if not defining_env.allow_overwrite:
            raise ProtocolError('This environment does not support overwriting mappings')
        defining_env.bindings[name] = value
        defining_env.unwrapped_bindings[name] = value.unwrapped

    def clear(self):
        self.bindings.clear()
        self.unwrapped_bindings.clear()

    def defined_names(self):
        return list(self.bindings.keys())

    def execute_statements(self, statements, return_allowed=False):
        result = V.Null()
        for statement in statements:
            result = statement.evaluate(self)
            if not isinstance(result, V.Null) and result is not None:
                if return_allowed is True:
                    break
                else:
                    raise ProtocolError('Return statement not allowed outside of function')
        return result

    # Methods to make an Environment behave a little like a (read-only) dictionary

    def __str__(self):
        return str(self.bindings)

    def __len__(self):
        return len(self.bindings)

    def __getitem__(self, key):
        return self.look_up(key)

    def __contains__(self, key):
        return key in self.bindings

    def __iter__(self):
        """Return an iterator over the names defined in the environment."""
        return iter(self.bindings)


class DelegatingDict(dict):
    def __init__(self, *args, **kwargs):
        super(DelegatingDict, self).__init__(*args, **kwargs)
        self.delegatees = {}
        from .language.expressions import NameLookUp
        self._marker = NameLookUp.PREFIXED_NAME
        self._marker_len = len(self._marker)

    # Have to delete references to modules (numpy) from the dictionary while maintaining
    # subclassing elements.
    # Pickling with extended dicts is hard...
    def __getstate__(self):
        odict = self.copy()
        if '___np' in odict:
            del odict['___np']
        return (self.__dict__, odict)

    # Separately update the state of dict (superclass) and DelegatingDict (subclass) components
    def __setstate__(self, state):
        self.__dict__.update(state[0])
        self.update(state[1])

    # IMPORTANT: Informs compiler of changes to pickling state enforced above
    def __reduce__(self):
        return (DelegatingDict, (), self.__getstate__())

    def __missing__(self, key):
        if key.startswith(self._marker):
            parts = key[self._marker_len:].split(self._marker, 1)
        else:
            parts = key.split(':', 1)
        if len(parts) == 2:
            prefix, name = parts
            if prefix in self.delegatees:
                return self.delegatees[prefix][name]
        if '' in self.delegatees:
            return self.delegatees[''][key]
        raise KeyError("Name " + key + " is not defined in env or any delegatee env")

    def set_delegatee(self, delegatee, prefix):
        self.delegatees[prefix] = delegatee

    def flatten(self):
        """Return a normal dictionary containing all the definitions in this and its delegatees."""
        result = dict(self)
        for prefix, delegatee in self.delegatees.items():
            for key, value in delegatee.flatten().items():
                full_key = prefix + ':' + key if prefix else key
                if full_key not in result:
                    result[full_key] = value
        return result


class ModelWrapperEnvironment(Environment):
    """This environment subclass provides access to a model's variables using the Environment interface.

    It supports variable lookups and the information methods, but doesn't allow defining new names.
    :meth:`overwrite_definition` must be used to change an existing variable's value.
    """

    class _BindingsDict(dict):
        """A dictionary subclass wrapping the protocol language versions of a model's variables."""

        def __init__(self, unwrapped):
            self._unwrapped = unwrapped
            try:
                line_profile.add_function(self.__getitem__)
            except NameError:
                pass

        def __getitem__(self, key):
            val = self._unwrapped[key]  # ~61%
            if isinstance(val, np.ndarray):
                return V.Array(val)
            else:
                return V.Simple(val)  # ~21%

        def __setitem__(self, key, value):
            pass

        def __contains__(self, key):
            return key in self._unwrapped

    class _UnwrappedBindingsDict(dict):
        """
        A dictionary subclass wrapping the Python versions of a model's variables.
        """
        # TODO: Look at the efficiency of get/set methods, and whether these matter for
        # overall performance (c.f. #2459).

        class _FreeVarList(list):
            """A single element list for wrapping the model's free variable."""

            def __init__(self, model):
                self._model = model

            def __getitem__(self, index):
                return self._model.free_variable

            def __setitem__(self, index, value):
                raise ProtocolError("Cannot set model free variable directly, only via simulation")

        class _OutputsList(list):
            """A pseudo-list for wrapping the model's output values."""

            def __init__(self, model):
                self._model = model

            def __getitem__(self, index):
                return self._model.get_outputs()[index]

            def __setitem__(self, index, value):
                raise ProtocolError(
                    "Cannot set model variable '" + self._model.output_names[index] +
                    "': it is a model output")

        def __init__(self, model):
            self._model = model
            # Make the underlying dict store a map from name to (vector, index) for fast lookup
            self._free_vars = self._FreeVarList(model)
            self._output_vars = self._OutputsList(model)
            # Note: we process outputs first so that if a variable is both an output and
            # something else, we prefer direct access
            for key in model.output_names:
                dict.__setitem__(self, key, (self._output_vars, model.output_names.index(key)))
            for key in model.parameter_map:
                dict.__setitem__(self, key, (model.parameters, model.parameter_map[key]))
            for key in model.state_var_map:
                dict.__setitem__(self, key, (model.state, model.state_var_map[key]))
            dict.__setitem__(self, model.free_variable_name, (self._free_vars, 0))

            try:
                line_profile.add_function(self.__getitem__)
                line_profile.add_function(self.__setitem__)
            except NameError:
                pass

        def __getitem__(self, key):
            # TODO: Catch KeyError?
            vector, index = dict.__getitem__(self, key)
            return vector[index]

        def __setitem__(self, key, value):
            # TODO: Catch KeyError?
            vector, index = dict.__getitem__(self, key)
            if vector[index] != value:
                self._model.dirty = True
                vector[index] = value

        def flatten(self):
            """Return a normal dictionary containing all the definitions in this and its delegatees."""
            return dict((key, self[key]) for key in self)

    def __init__(self, model):
        super(ModelWrapperEnvironment, self).__init__(allow_overwrite=True)
        self.model = model
        self.unwrapped_bindings = self._UnwrappedBindingsDict(model)
        self.bindings = self._BindingsDict(self.unwrapped_bindings)
        self.names = list(self.unwrapped_bindings.keys())

    def define_name(self, name, value):
        raise ProtocolError("Defining names in a model is not allowed.")

    def remove(self, name):
        raise ProtocolError("Removing names from a model is not allowed.")

    def defined_names(self):
        return self.names

