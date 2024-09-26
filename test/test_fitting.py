"""
Test running a simple fit on an FC protocol.
"""
import os

import fc
import fc.test_support


'''
def test_sine_wave():
    """Test running a sine wave protocol, comparing the output to a reference trace."""

    proto = fc.Protocol('test/protocols/test_sine_wave.txt')
    proto.set_output_folder('test_sine_wave')
    proto.set_model('test/models/beattie-2017-ikr-hh.cellml')
    proto.run()

    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))
    assert fc.test_support.check_results(
        proto,
        {'t': 1, 'V': 1, 'IKr': 1},
        os.path.join('test', 'output', 'sine_wave'),
        rel_tol=0.005,
        abs_tol=2.5e-4
    )
'''


class FittingSpec(object):
    """
    Interface for fitting experiment specifications.





    From discussion doc:

    - Each FittingSpec is linked to a single protocol, and bases its error criterion on one of the protocol's outputs.
    - Each data set is linked to a single protocol. Something inside the WL's database knows which data set columns map
      onto that protocol's output. This is part of the DataSet entity.  The FittingSpec is then compatible with any data
      set that links to the same protocol, and provides a reference value for that protocol output.

    """




class Exp2RateIKrFit(FittingSpec):
    """



    Objective
    =========

    This routine fits to the protocol output ``current``, which is linked to a model variable annotated as
    ``oxmeta:membrane_rapid_delayed_rectifier_potassium_current``.


    Parameters
    ==========

    Parameters are identified using the following steps:

    1. The variable representing the current being fitted is identified in the model.
    2. The network of variables that this current variable depends on is scanned for any variables annotated with either
       ``oxmeta:exp2_V_positive_rate`` or ``oxmeta:exp2_V_negative_rate``.
    3. The dependencies of each rate variable are scanned for variables annotated with either
       ``oxmeta:exp2_a_parameter`` or ``oxmeta:exp2_b_parameter``. It is expected that each rate depends on exactly one
       "a parameter" and one "b parameter". If any other combination is found an error is raised. All "a parameters" and
       "b parameters" found this way are treated as parameters.
    4. TODO: Also scan for `oxmeta:exp2_V_independent` or similar
    5. If a parameter is annotated as ``oxmeta:membrane_rapid_delayed_rectifier_potassium_current_conductance`` this is
       optimised as the conductance parameter.

    If no parameters can be found an error is returned.

    """
    _annot_output = 'oxmeta:membrane_rapid_delayed_rectifier_potassium_current'
    _annot_pos_rate = 'oxmeta:exp2_V_positive_rate'
    _annot_neg_rate = 'oxmeta:exp2_V_negative_rate'
    _annot_a_param = 'oxmeta:exp2_a_parameter'
    _annot_b_param = 'oxmeta:exp2_b_parameter'
    _annot_g_param = 'oxmeta:membrane_rapid_delayed_rectifier_potassium_current_conductance'
    #TODO V-independent rates

    def find_all_parameters(self, model):
        """
        Find all parameters in the given ``cellmlmanip.model`` that are compatible with this fitting spec, and return
        a mapping from variable objects to their desired units.



        Note that this method is called _before_ the model is manipulated by the protocol, and so

        OR IS IT!
        """



def test_fitting():
    """Test a fitting procedure using a sine wave protocol."""

    #
    # Get static test data (irl this would come from the front-end)
    #

    # Get protocol
    proto = 'test/protocols/test_sine_wave.txt'

    # Get model filename
    model = 'test/models/beattie-2017-ikr-hh.cellml'

    # TODO Get data set from front-end
    #data_set = front_end.get_data_set()

    # TODO Get fitting spec
    #spec = front_end.get_parsed_fitting_spec()
    spec = Exp2RateIKrFit()

    # Ensure compatibility?
    # front_end.ensure_compatibility(model, protocol, data_set, spec)
    # This was probably already done before we got here. If not, there's some room for optimisation (later).
    # Jonathan: I'm adding this kind of functionality to weblab-fc. Generally speaking I don't think we want to depend
    #           on the Django codebase (WebLab repo) here, only back-end stuff (weblab-fc, cellmlmanip, fc-runner, etc).

    #
    # Setting up a fitting problem
    #

    # Read the CellML, create a weblab model (
    # At this stage, we _must_ tell the protocol what units we want the parameters
    # in. But we don't know if all parameters are used to calculate the model
    # outputs yet (see below).
    proto.set_model(model, parameter_units)



    # Find _all_ parameters that _may_ be used in fitting later. The result maps cellmlmanip variables to desired units.
    parameter_units = spec.find_all_parameters(model)


    # Get output being fit to
    output_name = spec.find_output(protocol)
    # Again: This can be either part of spec, or done locally using info from spec. Note that this is a protocol output,
    # not a model variable.
    # Another question: I've assumed here that the units of the output are set by the protocol. So then we have
    # input-parameters=set-in-spec, but output=set-in-protocol. So slight inconsistency? Is that OK?



    # Create protocol object
    proto = fc.Protocol(proto)
    proto.set_output_folder('test_fitting')


    # Get parameters to alter during fitting.
    # The result is an ordered mapping from names to initial values
    parameters = spec.find_parameters(protocol.model)
    # Depending on whether we combine this with the next step or not, this can be `spec.annotations` or something of the
    # sort).

    # Get hardcoded prior or boundaries
    prior = spec.get_hardcoded_boundaries(protocol.model, parameters)
    # This needs to:
    #  1. Search the protocol-modified model for any parameters that declare the desired annotations.
    #  2. Check that those parameters can be combined into a multivariate prior, if desired. I.e. any "a" parameter
    #     needs a "b" parameter if we're using "exp2" rates. This will require some inspection of the model equations.
    # Considerations:
    #  - If we do this _after_ applying the protocol, all variables that are not required to calculate the protocol
    #    outputs will have been removed. This is useful if we have an AP model with 10 currents in exp2 form, but only
    #    one that we want to fit to.
    #
    # PINTS implementation
    #  Once the param pairs are known, and a single conductance has been identified (in case of the initial/default
    #  exp2-rate ion current spec), a prior can be created. This shouldn't be hard, e.g., these would exist (hardcoded):
    #
    #  class RateBoundary(pints.Boundary):
    #    def n_parameters(self):
    #      return 2
    #    ...
    #  class ConductanceBoundary(pints.Boundary):
    #   def n_parameters(self):
    #     return 1
    #   ...
    #
    #  so that the spec code would just do
    #   priors = [RateBoundary() for pair in rate_parameter_pairs]
    #   priors.append(ConductanceBoundary(min_conductance))
    #   return CompositeBoundary(priors)
    #
    #  (I've noticed we don't actually have a CompositeBoundary in PINTS, although we have the equivalent code for
    #   LogPrior, which is easily ported, https://github.com/pints-team/pints/issues/1209 .)
    #
    # Finally, if we anticipate inference rather than just fitting, we can actually create a pints.LogPrior here, and
    # convert it to a pints.Boundaries using `boundaries = pints.LogPDFBoundaries(log_prior)` if needed.
    # Resolution: Let WL do both.

    # Create stats model for PINTS
    class StatsModel(pints.ForwardModel):
        # Might want an __init__ here so that we don't reference global properties
        # later.
        # Jonathan says: The code above could perhaps all go in _init_?
        def n_parameters():
            return len(parameters)
        def n_outputs(self):
            return 1    # Can extend to multi-output later
        def simulate(self, parameter_values, times):

            # Pass parameters to Protocol
            for parameter, value in zip(parameters, parameter_values):
                protocol.set_model_variable(parameter, value)
                # Method doesn't exist yet. See
                # https://github.com/ModellingWebLab/weblab-fc/issues/213

            # Ignore times, these are fixed in protocol and known to match the data
            # Jonathan says: This generalises as part of setting protocol inputs
            #                from data; to do later!

            # Run protocol
            protocol.run(verbose=False, write_out=False)
            # I'm disabling verbose output here because I don't see us using a centralised tool like WL for day to day
            # fitting. Debugging would be hard? So I'm assuming for the moment that we've tested offline and don't need
            # elaborate output here.
            # Jonathan says: I think that's reasonable, as full output will slow things down a lot. We can still report
            #                on the parameter set that failed if there is an error, and what the error was, to enable
            #                offline debugging later.

            # Get output and return
            return protocol.output_env.look_up(output_name).array

    # Define optimisation/inference problem
    times, values = front_end.get_data_set_as_numpy_arrays(spec, data_set)
    # The above step would involve converting data to the units specified by the
    # protocol linked to the spec.
    stats_model = StatsModel()
    problem = pints.SingleOutputProblem(stats_model, times, values)
    # Notes:
    # 1. As above, times are ignored here, assumed the same as those generated by
    #    protocol. We could do a single protocol run with the default parameters to
    #    check this, if we liked.
    # 2. This would become pints.MultiOutputProblem when fitting to multi-varied
    #    data (model then has n_outputs > 1), but is otherwise the same!

    # Define error measure
    error = spec.create_error_measure(problem)
    # Where spec does e.g. `return pints.MeanSquaredError(problem)`.
    # For inference, this becomes e.g. `return pints.GaussianLogLikelihood` (which
    # should maybe be the standard, as PDFs can also be passed to a PINTS optimiser
    # (they will automatically be maximised, instead of minimised like an error).

    #
    # Run optimisation
    #

    # Starting point
    x0 = np.array(parameters.values())
    # This can be something nicer, e.g. `x0 = boundaries.sample(n=1)[0]`, which is
    # supported for pints boundaries (but we'd need to implement that for the rate
    # boundary above).
    # However, I'd advocate for repeated fits here anyway, which requires some
    # extra infra I'll discuss below.

    method = pints.CMAES    # Room to customise / get from spec
    opt = pints.OptimisationController(error, x0, boundaries, method=method)

    log_file = 'optimisation_output.txt'    # Room to customise / get from spec
    # Jonathan says: Doesn't really matter what it's called, providing it goes in
    #                the main output folder so it gets packed into the COMBINE
    #                Archive with all results!

    opt.set_log_to_screen(False)
    opt.set_log_to_file(log_file, csv=False)

    opt.set_max_unchanged_iterations(200, threshold=1e-11) # Room to customise/spec
    opt.set_max_iterations(None)                           # Room to customise/spec
    # Gary says: Yeah, certainly things you'd want to be able to tweak. Could have
    #            a call out here to a virtual method
    #            spec.setOptimisationOptions(opt) which could then work like
    #            altering a matlab optimset before you run the optimisation.

    n_cores = 4  # Room to customise / get from spec
    opt.set_parallel(n_cores)

    xbest, fbest = opt.run()

    # Instead of the above, it could be good to do repeated fits by default, and
    # bake this into the WL infrastructure.
    # The four-ways-of-fitting code had lots of nice methods for this, e.g. to
    # store parameters from a single run, load/save them etc., and I did a tidied
    # up version for the fitting tutorial (link).

    # so we can probably mine all that to set up something nicer here.
    # Not directly relevant for fitting spec design though: I think we can easily
    # tack this on later without changing the core functionality.
    # Gary says: Yeah, I think so, I wouldn't worry too much about that now, if we
    #            want to insist on it for everyone it is just a loop in here
    #            somewhere.... perhaps just a loop over spec.num_optimisation_runs
    #            or similar
    #
    # We'll also want a try-catch block, something to catch numpy warnings, etc.
    # (Fitting tutorial has some info on that too)

    # We now have xbest: an array of parameter values in the units specified by
    # the fitting spec

    # Map parameters to values
    obtained_parameter_values = dict(zip(parameters.keys(), xbest))

    # Return parameters to user
    front_end.return_to_user(xbest)

    # As discussed, we may also want to generate an updated model here! Which could
    # look something like
    output_model = non_destructive_cellml_tool.load_model(model_file)
    for parameter, value in obtained_parameter_values.items():

        current_unit = parameter_units(parameter)
        desired_unit = cellmlmanip.parse_unit(
            parameter.get_annotation('original_unit'))
        # (I'm inventing some cellmlmanip here, but this is definitely possible)
        # (Also current_unit will be a protocol unit, rather than a cellmlmanip
        # one, so would need some extra bits here. And parameter_units might have
        # variables from a different model object, but all very fixable!)
        # Jonathan says: The protocol and model share an underlying store, so units
        #                are compatible.

        # Get value in correct unit
        if current_unit != desired_unit:
            value = cellmlmanip.convert_quantity(value, current_unit, desired_unit)

        # Modify original XML serialisation in minimally invasive manner
        variable = output_model.get_variable(
            parameter.get_annotation('original_component'),
            parameter.get_annotation('original_name'))
        variable.set_value_or_initial_value(value)

    output_model.write_to_disk_with_minimal_changes(front_end.some_file_name())
    # Note that this is not something cellmlmanip can do (by design!), but again
    # not super relevant for designing the fitting spec I think.

    raise Exception('Not finished yet')
'''
