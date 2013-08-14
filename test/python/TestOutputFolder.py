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

import os
import shutil
import unittest

from fc.utility.error_handling import ProtocolError
from fc.utility.file_handling import OutputFolder


class TestOutputFolder(unittest.TestCase):
    """Test the OutputFolder class."""
    def TestCreationSingleFolder(self):
        """Test creating output folders."""
        # Single level, relative path provided
        single_folder_path = os.path.realpath(os.path.join(CHASTE_TEST_OUTPUT, 'TestOutputFolder_TestSingleFolder'))
        if os.path.exists(single_folder_path):
            shutil.rmtree(single_folder_path) # So we can test that the line below creates it
        single_folder = OutputFolder('TestOutputFolder_TestSingleFolder')
        self.assertEqual(single_folder.path, single_folder_path)
        self.assertTrue(os.path.exists(os.path.join(single_folder_path, OutputFolder.SIG_FILE_NAME)))
        # Single level, absolute path provided
        single_folder_2 = OutputFolder(single_folder_path)
        self.assertEqual(single_folder.path, single_folder_path)
        # Second level manually
        single_folder.CreateSubfolder('subfolder')
        self.assertTrue(os.path.isdir(os.path.join(single_folder_path, 'subfolder')))
        self.assertTrue(os.path.isfile(os.path.join(single_folder_path, 'subfolder', OutputFolder.SIG_FILE_NAME)))
        # Multiple levels at once
        multiple_folder_root = os.path.realpath(os.path.join(CHASTE_TEST_OUTPUT, 'TestOutputFolder_TestMultiFolders'))
        multiple_folder_path = os.path.join(multiple_folder_root, 'L1', 'L2', 'L3')
        if os.path.exists(multiple_folder_root):
            shutil.rmtree(multiple_folder_root)
        multiple_folders = OutputFolder(multiple_folder_path)
        self.assertTrue(os.path.exists(multiple_folder_path))
        self.assertTrue(os.path.exists(os.path.join(multiple_folder_root, OutputFolder.SIG_FILE_NAME)))
        self.assertTrue(os.path.exists(os.path.join(multiple_folder_root, 'L1', OutputFolder.SIG_FILE_NAME)))
        self.assertTrue(os.path.exists(os.path.join(multiple_folder_root, 'L1', 'L2', OutputFolder.SIG_FILE_NAME)))
        self.assertTrue(os.path.exists(os.path.join(multiple_folder_path, OutputFolder.SIG_FILE_NAME)))
        # Check we can remove folders we have created
        OutputFolder.RemoveOutputFolder(single_folder_path)
        OutputFolder.RemoveOutputFolder(multiple_folder_root)
        self.assertFalse(os.path.exists(single_folder_path))
        self.assertFalse(os.path.exists(multiple_folder_path))

    def TestUseOfEnvironmentVariable(self):
        # Check that setting CHASTE_TEST_OUTPUT affects where outputs appear
        original_env_var = os.environ.get('CHASTE_TEST_OUTPUT', None)
        os.environ['CHASTE_TEST_OUTPUT'] = os.path.join(original_env_var or 'testoutput', 'TestOutputFolder_NewTestOutput')
        if os.path.exists(os.environ['CHASTE_TEST_OUTPUT']):
            shutil.rmtree(os.environ['CHASTE_TEST_OUTPUT'])
        new_output_path = os.path.join(os.environ['CHASTE_TEST_OUTPUT'], 'TestUseOfEnvironmentVariable')
        OutputFolder('TestUseOfEnvironmentVariable') 
        self.assertTrue(os.path.exists(new_output_path))
        if original_env_var is not None:
            os.environ['CHASTE_TEST_OUTPUT'] = original_env_var

    def TestSafety(self):
        """Check that we're prevented from deleting content we shouldn't be able to."""
        # Content that's not under CHASTE_TEST_OUTPUT
        self.assertRaises(ProtocolError, OutputFolder.RemoveOutputFolder, '/tmp/cannot_delete')
        # Content that doesn't contain the signature file (but is under CHASTE_TEST_OUTPUT)
        testoutput = os.path.join(CHASTE_TEST_OUTPUT, 'TestOutputFolder_TestSafety')
        cannot_delete = os.path.join(testoutput, 'cannot_delete')
        if not os.path.exists(testoutput):
            os.mkdir(testoutput)
        open(cannot_delete, 'w').close()
        self.assertRaises(ProtocolError, OutputFolder.RemoveOutputFolder, cannot_delete)
        self.assertRaises(ProtocolError, OutputFolder.RemoveOutputFolder, testoutput)
        # A relative path containing .., putting it outside CHASTE_TEST_OUTPUT
        self.assertRaises(ProtocolError, OutputFolder, 'folder/../../../../../../etc')
