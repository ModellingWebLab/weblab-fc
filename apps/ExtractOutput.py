#!/usr/bin/env python

"""
Export a single protocol output from an HDF5 file to CSV.

Usage: ExtractOutput.py path_to_outputs_h5 output_name [path_to_output_csv]

If the output file is not specified, a file named outputs_{output_name}.csv
will be created in the same folder as the h5 file.
"""

import os
import sys

import tables
import numpy as np

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("Usage:", sys.argv[0], "path_to_output.h5 output_name [path_to_output.csv]",
          file=sys.stderr)
    sys.exit(1)

in_path = sys.argv[1]
output_name = sys.argv[2]
if len(sys.argv) == 4:
    out_path = sys.argv[3]
else:
    out_path = os.path.join(os.path.dirname(in_path), 'outputs_' + output_name + '.csv')

print("Reading output", output_name, "from", in_path, "and writing CSV to", out_path)

f = tables.open_file(in_path, 'r')
a = f.root.output._f_get_child(output_name).read()
np.savetxt(out_path, a.transpose(), delimiter=',')
