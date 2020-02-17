
import os
import shutil

import pytest

from fc.error_handling import ProtocolError
from fc.file_handling import OutputFolder

# Hack in variables defined by Chaste's testing framework, for now
CHASTE_TEST_OUTPUT = '/tmp/chaste_test_output'


class TestOutputFolder:
    """Test the OutputFolder class."""

    def test_creation_single_folder(self):
        """Test creating output folders."""
        # Single level, relative path provided
        single_folder_path = os.path.realpath(os.path.join(CHASTE_TEST_OUTPUT, 'TestOutputFolder_TestSingleFolder'))
        if os.path.exists(single_folder_path):
            shutil.rmtree(single_folder_path)  # So we can test that the line below creates it
        single_folder = OutputFolder('TestOutputFolder_TestSingleFolder')
        assert single_folder.path == single_folder_path
        assert os.path.exists(os.path.join(single_folder_path, OutputFolder.SIG_FILE_NAME))
        # Single level, absolute path provided
        single_folder_2 = OutputFolder(single_folder_path)
        assert single_folder_2.path == single_folder_path
        # Second level manually
        single_folder.create_subfolder('subfolder')
        assert os.path.isdir(os.path.join(single_folder_path, 'subfolder'))
        assert os.path.isfile(os.path.join(single_folder_path, 'subfolder', OutputFolder.SIG_FILE_NAME))
        # Multiple levels at once
        multiple_folder_root = os.path.realpath(os.path.join(CHASTE_TEST_OUTPUT, 'TestOutputFolder_TestMultiFolders'))
        multiple_folder_path = os.path.join(multiple_folder_root, 'L1', 'L2', 'L3')
        if os.path.exists(multiple_folder_root):
            shutil.rmtree(multiple_folder_root)
        OutputFolder(multiple_folder_path)
        assert os.path.exists(multiple_folder_path)
        assert os.path.exists(os.path.join(multiple_folder_root, OutputFolder.SIG_FILE_NAME))
        assert os.path.exists(os.path.join(multiple_folder_root, 'L1', OutputFolder.SIG_FILE_NAME))
        assert os.path.exists(os.path.join(multiple_folder_root, 'L1', 'L2', OutputFolder.SIG_FILE_NAME))
        assert os.path.exists(os.path.join(multiple_folder_path, OutputFolder.SIG_FILE_NAME))
        # Check we can remove folders we have created
        OutputFolder.remove_output_folder(single_folder_path)
        OutputFolder.remove_output_folder(multiple_folder_root)
        assert not os.path.exists(single_folder_path)
        assert not os.path.exists(multiple_folder_path)

    def test_use_of_environment_variable(self, tmp_path):
        # Check that setting CHASTE_TEST_OUTPUT affects where outputs appear
        original_env_var = os.environ.get('CHASTE_TEST_OUTPUT', None)
        os.environ['CHASTE_TEST_OUTPUT'] = os.path.join(
            original_env_var or str(tmp_path), 'TestOutputFolder_NewTestOutput')
        if os.path.exists(os.environ['CHASTE_TEST_OUTPUT']):
            shutil.rmtree(os.environ['CHASTE_TEST_OUTPUT'])
        new_output_path = os.path.join(os.environ['CHASTE_TEST_OUTPUT'], 'TestUseOfEnvironmentVariable')
        OutputFolder('TestUseOfEnvironmentVariable')
        assert os.path.exists(new_output_path)
        if original_env_var is not None:
            os.environ['CHASTE_TEST_OUTPUT'] = original_env_var

    def test_safety(self):
        """Check that we're prevented from deleting content we shouldn't be able to."""
        # Content that's not under CHASTE_TEST_OUTPUT
        with pytest.raises(ProtocolError):
            OutputFolder.remove_output_folder('/tmp/cannot_delete')
        # Content that doesn't contain the signature file (but is under CHASTE_TEST_OUTPUT)
        testoutput = os.path.join(CHASTE_TEST_OUTPUT, 'TestOutputFolder_TestSafety')
        cannot_delete = os.path.join(testoutput, 'cannot_delete')
        if not os.path.exists(testoutput):
            os.mkdir(testoutput)
        open(cannot_delete, 'w').close()
        with pytest.raises(ProtocolError):
            OutputFolder.remove_output_folder(cannot_delete)
        with pytest.raises(ProtocolError):
            OutputFolder.remove_output_folder(testoutput)
        # A relative path containing .., putting it outside CHASTE_TEST_OUTPUT
        with pytest.raises(ProtocolError):
            OutputFolder('folder/../../../../../../etc')
