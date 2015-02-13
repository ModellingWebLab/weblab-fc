"""Copyright (c) 2005-2015, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import inspect
import operator
import traceback
from cStringIO import StringIO


def _ExtractProtocolInfoFromStack(frames):
    """Figure out where within a protocol an error arose, based on the Python stack trace.
    
    Returns a pair (location_list, location_message) containing a list of protocol file locations and a rendered
    explanatory message, respectively.
    """
    locations = []
    last_env = None
    trace_output = StringIO()
    
    for frame in frames:
        local_vars = frame.f_locals
        obj = local_vars.get('self', None)
        if obj and hasattr(obj, 'location'):
            if not locations or obj.location != locations[-1]:
                locations.append(obj.location)
            env = local_vars.get('env', None)
            if env and env is not last_env:
                last_env = env
                obj.Trace('Variables defined at ' + obj.location + ':', stream=trace_output)
                for name in env:
                    obj.Trace(env[name], stream=trace_output, prefix='  '+name+' = ')
    if locations:
        message = 'Protocol stack trace (most recent call last):\n  ' + '\n  '.join(locations)
        if last_env:
            message += '\n' + trace_output.getvalue()
    else:
        message = ""
    return locations, message

class ProtocolError(Exception):
    """Main class for errors raised by the functional curation framework that are intended for user viewing."""
    def __init__(self, *msgParts):
        """Create a protocol error message.
        
        The arguments are joined to create the message string as for print: converted to strings and space separated.
        In addition, when the exception is created the stack will be examined to determine what lines in the currently
        running protocol were responsible, if any, and these details added to the Python stack trace reported. (The list
        of locations will also be stored as self.locations.) Since this can make the error message rather long, we also
        store self.shortMessage as the message string created from the constructor arguments.
        """
        # Figure out where in the protocol this error arose
        self.locations, location_message = _ExtractProtocolInfoFromStack(reversed(map(operator.itemgetter(0), inspect.stack())))
        # Construct the full error message
        self.shortMessage = ' '.join(map(str, msgParts))
        msg = self.shortMessage + '\n' + location_message
        super(ProtocolError, self).__init__(msg)

class ErrorRecorder(ProtocolError):
    """A context manager for recording protocol errors that arise within a block of code.
    
    This is designed to be used in multiple successive code blocks.  Any error thrown from a managed block is recorded
    (appended to self.errors) and suppressed.
    
    In addition to the context management protocol, this class inherits from ProtocolError and may be thrown as an exception
    to report on the collected errors.  Before doing so, check if any errors have occurred by testing in a boolean
    context.
    
    TODO: Add the ability to suppress Python traceback for some errors?
    """
    def __init__(self, protoName):
        """Create a new recorder."""
        self.errors = []
        super(ErrorRecorder, self).__init__("Errors occurred during execution of %s:" % protoName)
    
    def __enter__(self):
        """Start a managed block of code."""
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Finish a managed block of code, perhaps because an exception was raised."""
        if exc_value is not None:
            self.errors.append(exc_value)
            message = "%d. %s: %s" % (len(self.errors), exc_type.__name__, str(exc_value))
            if hasattr(exc_value, 'shortMessage'):
                self.shortMessage += "\n" + exc_value.shortMessage
            else:
                message += "\nTraceback (most recent call last):\n" + ''.join(traceback.format_tb(exc_traceback))
                frames = []
                while exc_traceback:
                    frames.append(exc_traceback.tb_frame)
                    exc_traceback = exc_traceback.tb_next
                message += _ExtractProtocolInfoFromStack(frames)[1]
            args = list(self.args)
            args[0] += "\n" + message
            self.args = tuple(args)
        return True
    
    def __nonzero__(self):
        """Test if any errors have been recorded."""
        return bool(self.errors)
