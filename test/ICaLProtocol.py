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
    # cap = doc.model.get_variable_by_oxmeta_name('membrane_capacitance')
    Cao = doc.model.get_variable_by_oxmeta_name('extracellular_calcium_concentration')
    
    # Change V to be a constant set from a new parameter
    value_name = u'membrane_voltage_value'
    V_value = protocol.cellml_variable.create_new(doc, value_name, V.units, id=value_name,
                                                  initial_value=V.initial_value)
    V_const_defn = protocol.mathml_apply.create_new(doc, u'eq', [V.component.name + u',' + V.name,
                                                                 u'protocol,' + value_name])

    # Change Ko to be a constant set from a new parameter
    Cao_value_name = u'extracellular_calcium_concentration_value'
    Cao_value = protocol.cellml_variable.create_new(doc, Cao_value_name, Cao.units, id=Cao_value_name,
                                                  initial_value=Cao.initial_value)
    Cao_const_defn = protocol.mathml_apply.create_new(doc, u'eq', [Cao.component.name + u',' + Cao.name,
                                                                 u'protocol,' + Cao_value_name])

    # Now a hack to stop translation complaining about missing currents
    i_stim = protocol.cellml_variable.create_new(doc, u'i_stim', LCC.units, id=u'membrane_stimulus_current')
    i_stim_defn = protocol.mathml_apply.create_new(doc, u'eq', [u'protocol,i_stim',
                                                                (u'0', LCC.units)])
    doc._cml_config.options.use_i_ionic_regexp = True
    doc._cml_config.i_ionic_definitions = [doc._cml_config._create_var_def(LCC.component.name + u',' + LCC.name, u'name')]
    
    p.outputs = [V, LCC, t, Cao]
    p.inputs = [Cao_value, Cao_const_defn, V_value, V_const_defn, i_stim, i_stim_defn]
    p.modify_model()
    i_stim.set_oxmeta_name(u'membrane_stimulus_current')
