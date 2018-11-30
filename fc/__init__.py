"""
Root for the Python package implementing Functional Curation.

For most uses, including running protocols on models, you should just use:
    import fc
The main class is then available as `fc.Protocol`.

Extensions of Functional Curation can instead import the variable sub-packages and modules directly.
"""

from .utility.protocol import Protocol  # noqa:F401
