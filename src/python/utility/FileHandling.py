"""Copyright (c) 2005-2013, University of Oxford.
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


class OutputFolder(object):
    """
    This class contains routines providing the key functionality of the C++ classes OutputFileHandler and FileFinder.
    In particular, it allows the creation of output folders within the location set for Chaste output, and safe deletion
    of such folders, without risk of wiping a user's drive due to coding error or dodgy path settings.
    """
    def __init__(self, path, cleanFolder=True):
        """Create a new output subfolder.
        
        :param path:  the subfolder to create.  Relative paths are treated as relative to GetRootOutputFolder; absolute
        paths must be under this location.  Parent folders will be created as necessary.
        :param cleanFolder:  whether to wipe the folder contents if it already exists.
        """
        raise NotImplementedError
    
    @staticmethod
    def GetRootOutputFolder(self):
        """Get the root location where Chaste output files are stored.
        
        This is read from the environment variable CHASTE_TEST_OUTPUT; if it is not set then a folder 'testoutput' in
        the current working directory is used.
        """
        raise NotImplementedError

    def GetAbsolutePath(self):
        """Get the absolute path to this output folder."""
        raise NotImplementedError
    
    def CreateSubFolder(self, path):
        """Create a new OutputFolder inside this one.
        
        :param path:  the name of the subfolder to create.  This must be a relative path.
        """
        raise NotImplementedError
    
    SIG_FILE_NAME = '.chaste_deletable_folder'
    
    @staticmethod
    def RemoveOutputFolder(self, path):
        """Remove an existing output folder.
        
        This method will only delete folders living under the root output folder.  In addition, they must have been
        created using the OutputFolder class (this is indicated by the presence of a signature file within the folder).
        
        :param path:  the folder to remove.  Relative paths are treated as relative to GetRootOutputFolder; absolute
        paths must be under this location.
        """
        raise NotImplementedError
    
    @staticmethod
    def CheckOutputPath(self, path):
        """Check whether the given path is a location within the Chaste output folder."""
        raise NotImplementedError
    