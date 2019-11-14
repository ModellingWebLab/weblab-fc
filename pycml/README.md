# Temporary PyCML files

These are not used in `fc`, but are here for reference while we switch from pycml
to the new `cellmlmanip`/`weblab_cg` solution.


## Some notes on how pycml processes the 'model interface' section

Unless otherwise noted, bare method names are defined in `protocol.py`.
Names with a `p:` prefix are defined in one of the classes in `processors.py`.

See comments in the methods for details not mentioned here; this document focuses on the main flow and rationale.

Many of the complexities in pycml arise from:
- the XML object mapping representation for mathematics, which makes creating/changing maths complicated;
- the need to put variables in components, and maintain sensible connections, including potentially through intermediate encapsulating components, ensuring unique naming;
- arising from this, managing name references in mathematics (which can be local names, fully qualified names, or ontology terms) particularly when equations are moved or changed;
- the need for units to be defined in a component and referenced by name from variables;
- the internal state needed by code generation, for instance to classify variables (computed, state, etc) and topologically sort them. This state can shift as model modifications are made, and might not be fully known until all modifications have been done, so the state can sometimes be 'nonsense' at intermediate stages.

Protocol parsing starts with `parse_protocol`, called by `apply_protocol_file`. It ensures that the different parts of the interface are processed in an order that respects the principle of least surprise for protocol authors: it tries to ensure their modifications do what they probably expect!

Imports are treated first, included those arising implicitly from nested protocols. Where relevant, units definitions, interface definitions, etc. from imports are merged with the parent protocol's definitions.

Parts of the interface are then processed in the following order:
1. Optional variable specifications (`optional ...`) using `specify_optional_variable()`. If the variable doesn't exist and a default definition is given, will add the defining equation to `self.inputs`.
2. New variable specifications (`var ...`) just adds the variable (and its units). See `declare_new_variable()`.
3. Special units conversion rules (`convert ...`) are added to the `UnitsConverter` instance by `add_units_conversion_rule()`. It will check that variables referenced in the rule exist, and skip the rule if not.
4. The units of time are set, if requested explicitly (`independent var units ...`). (This may also happen implicitly if time is an input or output, but doing it explicitly can resolve conflicts that arise if time gets units converted *after* some state variables have been converted.) `set_independent_variable_units()` will create a new time var now if required, and note that it has done so (since later all ODEs will need to be altered).
5. Input variable declarations are processed (`input ...`) with `specify_input_variable()`. This just creates the variable if it's not optional and doesn't exist, and adds it to `self.inputs` for later processing by `process_input_declarations()`.
6. Output variable declarations are procesed (`output ...`); at this point `specify_output_variable()` just records the specifications in a list for `process_output_declarations()` to use.
7. New or changed equations are noted (`define ...`; `clamp ...` is a shorthand for this). At this stage it just stores the new definition in the `self.inputs` list.

Then `modify_model` uses the information recorded above to make the bulk of the changes to the model's mathematics:
1. New units definitions are added to the model.
2. New variables in `self.inputs` are created by `_add_variable_to_model`. This can potentially replace an original version in the model, but I'm not sure this happens with the flow as it currently stands. Earlier versions of pycml did more work in `specify_input_variable` and didn't have `process_input_declarations`, but this turned out not to work well in some corner cases!
3. New/changed equations are checked, and the assigned-to variable created by `_check_equation_lhs` if it doesn't exist (and we can do so, i.e. it is also defined as an output or optional).
4. New/changed equations are added to the model by `_add_maths_to_model`. At essence this removes any existing definition of the LHS, and adds the new equation. There are special cases for things like linear interpolation on a data file, clamping to initial value, or the `original_definition` keyword. Checks are made to see if all referenced variables exist or are optional; the latter is OK provided the variable defined by the equation is also optional!
5. Inputs are finalised by `process_input_declarations()`, which checks they all exist and are of allowed type (constant, state or free), adds a new variable if units conversion is required, updates the initial value if that was specified in the protocol, and ensures the name annotation is correct.
6. Outputs are finalised by `process_output_declarations()`, which checks they all exist (unless optional) and adds a new version if units conversions are required. It also annotates all state variables with the magic `oxmeta:state_variable` name, and finally uses `process_output_variable_vectors()` to handle outputs that are vectors of variables (i.e. where an ontology term matches many variables, like `oxmeta:state_variable`).
7. If the time variable was units-converted, ODEs are transformed with `_split_all_odes()`, as discussed at https://github.com/ModellingWebLab/cellmlmanip/issues/77.
8. `_fix_model_connections()` adds new variables and connections where mathematics refers to a variable outside its own component. This won't be needed in cellmlmanip!
9. `p:finalize()` actually applies units conversions and re-does the model static analysis needed for code generation, e.g. building the dependency graph between equations. Note that we only apply conversions to connections and equations we have explicitly added/changed; we assume the original model is consistent.
10. `_filter_assignments()` is used to throw away any maths not required to compute the outputs requested by the protocol. It also adds pycml-specific annotations to ensure code generation treats inputs & outputs correctly.
11. Finally, some statistics on the modifications are printed to stderr by `report_stats()`: the numbers of inputs, outputs, state vars, equations, and variables in the final model, and the ontology terms for any missing optional variables.
