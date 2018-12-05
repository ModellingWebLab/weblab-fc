#!/usr/bin/env python

"""
Check the syntax of a (text syntax) protocol file.

Any errors are printed to standard output, and the return code is set to 1.
If there are no errors, we simply exit with a 0 return code.
"""

import os
import subprocess
import sys

def CheckSyntax(protocolPath):
    parser = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src/proto/parsing/CompactSyntaxParser.py')
    exit_code = subprocess.call(['python', parser, protocolPath, '--dry-run'])
    sys.exit(exit_code)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    CheckSyntax(sys.argv[1])
