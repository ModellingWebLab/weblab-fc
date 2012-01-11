"""Copyright (C) University of Oxford, 2005-2012

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
    A very simple protocol, that doesn't make any changes, and is only
    really interested in membrane_voltage as an output.
    
    However, for test coverage purposes, we also include each type of
    variable in the outputs.
    """
    p = protocol.Protocol(doc.model, multi_stage=True)
    V = doc.model.get_variable_by_oxmeta_name('membrane_voltage')
    t = doc.model.get_variable_by_name('environment', 'time')
    derived = doc.model.get_variable_by_cmeta_id('FonRT')
    mapped = doc.model.get_variable_by_name('fast_sodium_current', 'V')
    param = doc.model.get_variable_by_cmeta_id('fast_sodium_current_conductance')
    p.outputs = set([V, t, mapped, param, derived])
    p.inputs = set()
    p.modify_model()
    # For the present, we don't want to avoid using Chaste's stimulus!
    doc._cml_config.finalize_config()
    doc._cml_config.find_current_vars()
