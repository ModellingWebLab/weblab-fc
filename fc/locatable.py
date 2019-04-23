
import inspect
import os
import sys


class Locatable(object):
    """Base class for constructs in the protocol language.

    This class stores the location within a protocol file giving rise to the construct,
    and also whether the construct is being traced, i.e. can have its value written to a trace
    file when evaluated.  The class variable outputFolder records where trace should be written,
    if enabled, and the Trace method can be used to trace a value.
    """

    def __init__(self, location=None):
        """Create a new construct with optional location information.

        If no location is given, a string representation of the Python stack of the caller will be used.
        This is not ideal, but better than nothing!
        """
        if location is None:
            location = str(inspect.stack()[2][1:5])
        self.location = location
        self.trace = False

    def Trace(self, value, stream=sys.stdout, prefix=''):
        """Trace the given value.

        A synopsis will be written to stream, and the full value to a trace file, if self.outputFolder
        is defined.
        """
        from .language import values as V
        if isinstance(value, V.Array) and value.array.size > 10:
            stream.write(prefix + 'array shape ' + str(value.array.shape) + '\n')
        else:
            stream.write(prefix + str(value) + '\n')

        if self.outputFolder:
            trace_path = os.path.join(self.outputFolder.path, 'trace.txt')
            f = open(trace_path, 'a+')
            f.write(prefix + str(value) + '\n')
            f.close()

    # Global that sets where to write trace, if enabled
    outputFolder = None
