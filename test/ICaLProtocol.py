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

    # Units
    current_units = protocol.cellml_units.create_new(
        doc.model, 'proto_uA_per_cm2',
        [{'units': 'ampere', 'prefix': 'micro'},
         {'units': 'metre', 'prefix': 'centi', 'exponent': '-2'}])
#    uF_per_cm2 = protocol.cellml_units.create_new(
#        doc.model, 'proto_uF_per_cm2',
#        [{'units': 'farad', 'prefix': 'micro'},
#         {'units': 'metre', 'prefix': 'centi', 'exponent': '-2'}])
#    chaste_cm = protocol.cellml_variable.create_new(doc.model, u'chaste_membrane_capacitance',
#                                                    uF_per_cm2.name, initial_value=u'1')
#    conc_units = protocol.cellml_units.create_new(
#        doc.model, 'proto_mM',
#        [{'units': 'mole', 'prefix': 'milli'},
#         {'units': 'litre', 'exponent': '-1'}])

    # LCC in desired units for comparison
    p.specify_as_output(LCC, current_units)
    
    # V and Cao should become modifiable parameters
    p.specify_as_input(V, V.get_units())
    p.specify_as_input(Cao, Cao.get_units())

    # Now a hack to stop translation complaining about missing currents
    i_stim = doc.model.get_variable_by_oxmeta_name('membrane_stimulus_current')
    i_stim_defn = protocol.mathml_apply.create_new(doc, u'eq', [i_stim.component.name + u',' + i_stim.name,
                                                                (u'0', LCC.units)])
    doc._cml_config.options.use_i_ionic_regexp = True
    doc._cml_config.i_ionic_definitions = [doc._cml_config._create_var_def(LCC.component.name + u',' + LCC.name, u'name')]
    
    p.outputs.update([LCC, i_stim])
    p.inputs.update([i_stim_defn])
    p.modify_model()
