"""
Root for the Python package implementing Functional Curation.

For most uses, including running protocols on models, you should just use:
    import fc
The main class is then available as `fc.Protocol`.

Extensions of Functional Curation can instead import the variable sub-packages and modules directly.
"""

import os, inspect  # noqa
try:
    frame = inspect.currentframe()
    MODULE_DIR = os.path.dirname(inspect.getfile(frame))
finally:
    # Always manually delete frame
    # https://docs.python.org/2/library/inspect.html#the-interpreter-stack
    del(frame)
del(os, inspect)


from .utility.protocol import Protocol  # noqa:F401

