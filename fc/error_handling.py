
import inspect
import operator
import traceback
from io import StringIO

from .locatable import Locatable


def _extract_protocol_info_from_stack(frames):
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
        if obj and isinstance(obj, Locatable):
            if not locations or obj.location != locations[-1]:
                locations.append(obj.location)
            env = local_vars.get('env', None)
            if env and env is not last_env:
                last_env = env
                obj.trace('Variables defined at ' + obj.location + ':', stream=trace_output)
                for name in env:
                    obj.trace(env[name], stream=trace_output, prefix='  ' + name + ' = ')
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
        store self.short_message as the message string created from the constructor arguments.
        """
        # Figure out where in the protocol this error arose
        self.locations, location_message = _extract_protocol_info_from_stack(
            map(operator.itemgetter(0), reversed(inspect.stack())))
        # Construct the full error message
        self.short_message = ' '.join(map(str, msgParts))
        msg = self.short_message + '\n' + location_message
        super(ProtocolError, self).__init__(msg)


class ErrorRecorder(ProtocolError):
    """A context manager for recording protocol errors that arise within a block of code.

    This is designed to be used in multiple successive code blocks.
    Any error thrown from a managed block is recorded (appended to self.errors) and suppressed.

    In addition to the context management protocol, this class inherits from ProtocolError
    and may be thrown as an exception to report on the collected errors.  Before doing so,
    check if any errors have occurred by testing in a boolean context.

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
            if hasattr(exc_value, 'short_message'):
                self.short_message += "\n" + exc_value.short_message
            message += "\nTraceback (most recent call last):\n" + ''.join(traceback.format_tb(exc_traceback))
            frames = []
            while exc_traceback:
                frames.append(exc_traceback.tb_frame)
                exc_traceback = exc_traceback.tb_next
            message += _extract_protocol_info_from_stack(frames)[1]
            args = list(self.args)
            args[0] += "\n" + message
            self.args = tuple(args)
        return True

    def __bool__(self):
        """Test if any errors have been recorded."""
        return bool(self.errors)
