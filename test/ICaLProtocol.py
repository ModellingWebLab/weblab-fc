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
    current_units = protocol.cellml_units.create_new(
        doc.model, 'proto_uA_per_cm2',
        [{'units': 'ampere', 'prefix': 'micro'},
         {'units': 'metre', 'prefix': 'centi', 'exponent': '-2'}])
    uF_per_cm2 = protocol.cellml_units.create_new(
        doc.model, 'proto_uF_per_cm2',
        [{'units': 'farad', 'prefix': 'micro'},
         {'units': 'metre', 'prefix': 'centi', 'exponent': '-2'}])
    chaste_cm = protocol.cellml_variable.create_new(doc.model, u'chaste_membrane_capacitance',
                                                    uF_per_cm2.name, initial_value=u'1')
    conc_units = protocol.cellml_units.create_new(
        doc.model, 'proto_mM',
        [{'units': 'mole', 'prefix': 'milli'},
         {'units': 'litre', 'exponent': '-1'}])
    
    # Another hack: ensure Cm appears in protocol component for conversions if needed
    if Cm:
        Cm_proto = protocol.cellml_variable.create_new(doc, Cm.name+u'_proto', Cm.units)
        Cm_proto_defn = protocol.mathml_apply.create_new(doc, u'eq', [u'protocol,' + Cm.name+u'_proto',
                                                                      Cm.component.name + u',' + Cm.name])
        doc._cml_config.Cm_variable = Cm_proto

    # LCC in desired units for comparison
    LCC_output = protocol.cellml_variable.create_new(doc, LCC.name + '_compare', current_units.name,
                                                     id=LCC.oxmeta_name + '_compare')
    LCC_output_defn = protocol.mathml_apply.create_new(doc, u'eq', [u'protocol,' + LCC.name + '_compare',
                                                                    LCC.component.name + u',' + LCC.name])
    
    # Change V to be a constant set from a new parameter
    value_name = u'membrane_voltage_value'
    V_value = protocol.cellml_variable.create_new(doc, value_name, V.units, id=value_name,
                                                  initial_value=V.initial_value)
    V_const_defn = protocol.mathml_apply.create_new(doc, u'eq', [V.component.name + u',' + V.name,
                                                                 u'protocol,' + value_name])

    # Change Ko to be a constant set from a new parameter
    Cao_value_name = u'extracellular_calcium_concentration_value'
    Cao_value = protocol.cellml_variable.create_new(doc, Cao_value_name, Cao.units, id=Cao_value_name,
                                                    initial_value=Cao.initial_value) # TODO: units-convert initial value?
    Cao_const_defn = protocol.mathml_apply.create_new(doc, u'eq', [Cao.component.name + u',' + Cao.name,
                                                                   u'protocol,' + Cao_value_name])

    # Now a hack to stop translation complaining about missing currents
    i_stim = doc.model.get_variable_by_oxmeta_name('membrane_stimulus_current')
    i_stim_defn = protocol.mathml_apply.create_new(doc, u'eq', [i_stim.component.name + u',' + i_stim.name,
                                                                (u'0', LCC.units)])
    doc._cml_config.options.use_i_ionic_regexp = True
    doc._cml_config.i_ionic_definitions = [doc._cml_config._create_var_def(LCC.component.name + u',' + LCC.name, u'name')]
    
    p.outputs = [V, LCC_output, t, Cao, LCC, i_stim]
    p.inputs = [Cao_value, Cao_const_defn, V_value, V_const_defn, i_stim_defn,
                LCC_output, LCC_output_defn, conc_units, current_units, chaste_cm, uF_per_cm2]
    if Cm:
        p.outputs.extend([Cm, Cm_proto])
        p.inputs.extend([Cm_proto_defn, Cm_proto])
    p.modify_model()
