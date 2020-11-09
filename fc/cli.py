"""
Command-line interface for functional curation.
"""

import argparse
import os
import sys

import tables

from fc import Protocol
from fc.file_handling import extract_output


def run_protocol():
    """Run a single protocol on a single model."""
    parser = argparse.ArgumentParser(description='Run a functional curation protocol')
    parser.add_argument('model', help='path to the CellML model file to run on')
    parser.add_argument('protocol', help='path to the protocol file to run')
    args = parser.parse_args()

    proto = Protocol(args.protocol)
    proto.set_output_folder('{}_{}'.format(
        os.path.splitext(os.path.basename(args.protocol))[0],
        os.path.splitext(os.path.basename(args.model))[0]))
    proto.set_model(args.model)
    proto.run()


def extract_outputs():
    """Extract CSV files of outputs from a single HDF5 file."""
    parser = argparse.ArgumentParser(description='Extract CSV files from a protocol output HDF5 file')
    parser.add_argument('-o', '--output', help='a specific output name to extract')
    parser.add_argument('h5path', help='path to the HDF5 file containing all outputs')
    args = parser.parse_args()

    with tables.open_file(args.h5path, 'r') as h5file:
        output_folder = os.path.dirname(args.h5path)
        if args.output:
            extract_output(h5file, args.output, output_folder)
        else:
            for output in h5file.root.output._v_leaves.keys():
                extract_output(h5file, output, output_folder)


def check_syntax():
    """Check the syntax of a protocol file, indicating OK or not by exit code."""
    parser = argparse.ArgumentParser(
        description='Check the syntax of a protocol file, indicating OK or not by exit code')
    parser.add_argument('protocol', help='path to the protocol file to check')
    args = parser.parse_args()

    try:
        Protocol(args.protocol)
    except Exception:
        sys.exit(1)
