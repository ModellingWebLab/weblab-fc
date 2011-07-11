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
    Basically we just need to simulate the standard model and record V
    (and t).  However, we do need to ensure that the stimulus current
    has the expected form with the parameters needed by the protocol.
    """
    doc._cml_config.options.convert_interfaces = True
    p = protocol.Protocol(doc.model, multi_stage=True)
    V = doc.model.get_variable_by_oxmeta_name('membrane_voltage')
    t = doc.model.get_variable_by_oxmeta_name('time')
    
    def maybe_new_var(name, units, init):
        try:
            v = doc.model.get_variable_by_oxmeta_name(name)
        except:
            v = None
        if v:
            v = p.specify_as_input(v, units)
            v.initial_value = init
        else:
            v = protocol.cellml_variable.create_new(doc.model, name, units.name, initial_value=init, id=name)
            p.inputs.add(v)
        return v
    new_var = lambda name, units, init: protocol.cellml_variable.create_new(doc.model, name, units, initial_value=init, id=name)
    new_apply = lambda op, args: protocol.mathml_apply.create_new(doc, op, args)
    var = lambda v: v.component.name + u',' + v.name
    pvar = lambda v: u'protocol,' + v.name

    ms = protocol.cellml_units.create_new(doc.model, u'ms', [{'units':'second', 'prefix':'milli'}])
    i_stim = doc.model.get_variable_by_oxmeta_name('membrane_stimulus_current')
    i_stim_amp = doc.model.get_variable_by_oxmeta_name('membrane_stimulus_current_amplitude')
    i_stim_duration = doc.model.get_variable_by_oxmeta_name('membrane_stimulus_current_duration')
    stim_end = maybe_new_var(u'membrane_stimulus_current_end', ms, u'100000000000')
    stim_offset = maybe_new_var(u'membrane_stimulus_current_offset', ms, u'10')
    stim_period = maybe_new_var(u'membrane_stimulus_current_period', ms, u'1000')
    start = new_apply('geq', [var(t), pvar(stim_offset)])
    stop = new_apply('leq', [var(t), pvar(stim_end)])
    during = new_apply('leq',
                       [new_apply('minus',
                                  [new_apply('minus', [var(t), pvar(stim_offset)]),
                                   new_apply('times',
                                             [new_apply('floor',
                                                        [new_apply('divide',
                                                                   [new_apply('minus', [var(t), pvar(stim_offset)]),
                                                                    pvar(stim_period)])]),
                                              pvar(stim_period)])]),
                        var(i_stim_duration)])
    cond = new_apply(u'and', [start, stop, during])
    case = var(i_stim_amp)
    otherwise = protocol.mathml_cn.create_new(doc, u'0', i_stim_amp.units)
    i_stim_defn = protocol.mathml_piecewise.create_new(doc, [(case, cond)], otherwise)

    p.outputs = [V, t]
    p.inputs.add(i_stim_defn)
    p.modify_model()
