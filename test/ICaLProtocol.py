"""Copyright (C) University of Oxford, 2005-2011

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Chaste is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 2.1 of the License, or
(at your option) any later version.

Chaste is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details. The offer of Chaste under the terms of the
License is subject to the License being interpreted in accordance with
English Law and subject to any action against the University of Oxford
being under the jurisdiction of the English Courts.

You should have received a copy of the GNU Lesser General Public License
along with Chaste. If not, see <http://www.gnu.org/licenses/>.
"""

import protocol

def apply_protocol(doc):
    """
    A protocol to return the L-type calcium current over time for a given voltage.
    """
    p = protocol.Protocol(doc.model, multi_stage=True)
    LCC = doc.model.get_variable_by_oxmeta_name('membrane_L_type_calcium_current')
    t = doc.model.get_variable_by_oxmeta_name('time')
    V = doc.model.get_variable_by_oxmeta_name('membrane_voltage')
    Cao = doc.model.get_variable_by_oxmeta_name('extracellular_calcium_concentration')
    try:
        Cm = doc.model.get_variable_by_oxmeta_name('membrane_capacitance')
    except:
        Cm = None

    # Units
    new_units = protocol.cellml_units.create_new
    current_units = new_units(doc.model, 'proto_uA_per_cm2',
                              [{'units': 'ampere', 'prefix': 'micro'},
                               {'units': 'metre', 'prefix': 'centi', 'exponent': '-2'}])
    uF_per_cm2 = new_units(doc.model, 'proto_uF_per_cm2',
                           [{'units': 'farad', 'prefix': 'micro'},
                            {'units': 'metre', 'prefix': 'centi', 'exponent': '-2'}])
    microamps = new_units(doc.model, u'microamps',
                          [{'units':'ampere', 'prefix':'micro'}])
    A_per_F = new_units(doc.model, 'A_per_F',
                        [{'units': 'ampere'},
                         {'units': 'farad', 'exponent': '-1'}])
    
    # Define special units conversion for LCC
    chaste_cm = protocol.cellml_variable.create_new(doc.model, u'chaste_membrane_capacitance',
                                                    uF_per_cm2.name, initial_value=u'1')
    converter = p.get_units_converter()
    converter.add_special_conversion(A_per_F, current_units,
                                     lambda expr: converter.times_rhs_by(expr, chaste_cm))
    converter.add_special_conversion(current_units, A_per_F,
                                     lambda expr: converter.divide_rhs_by(expr, chaste_cm))
    if Cm:
        converter.add_special_conversion(microamps, current_units,
                lambda expr: converter.times_rhs_by(converter.divide_rhs_by(expr, Cm), chaste_cm))
        converter.add_special_conversion(current_units, microamps,
                lambda expr: converter.divide_rhs_by(converter.times_rhs_by(expr, Cm), chaste_cm))
    
    # LCC in desired units for comparison
    LCC = p.specify_as_output(LCC, current_units)
    
    # V and Cao should become modifiable parameters
    V = p.specify_as_input(V, V.get_units())
    Cao = p.specify_as_input(Cao, Cao.get_units())

    # Now a hack to stop translation complaining about missing currents
    i_stim = doc.model.get_variable_by_oxmeta_name('membrane_stimulus_current')
    i_stim_defn = protocol.mathml_apply.create_new(doc, u'eq', [i_stim.component.name + u',' + i_stim.name,
                                                                (u'0', LCC.units)])
    doc._cml_config.options.use_i_ionic_regexp = True
    doc._cml_config.i_ionic_definitions = [doc._cml_config._create_var_def(LCC.component.name + u',' + LCC.name, u'name')]
    
    # Quick fix to get t units-converted
    doc._cml_config.options.convert_interfaces = True
    
    p.outputs.update([V, t, Cao, LCC, i_stim])
    p.inputs.update([i_stim_defn, chaste_cm, uF_per_cm2])
    p.modify_model()
