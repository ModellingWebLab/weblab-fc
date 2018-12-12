
import os
import shutil

from .error_handling import ProtocolError


class OutputFolder(object):
    """
    This class contains routines providing the key functionality of the C++ classes OutputFileHandler and FileFinder.
    In particular, it allows the creation of output folders within the location set for Chaste output, and safe deletion
    of such folders, without risk of wiping a user's drive due to coding error or dodgy path settings.
    """
    SIG_FILE_NAME = '.chaste_deletable_folder'

    def __init__(self, path, cleanFolder=True):
        """Create a new output subfolder.

        :param path:  the subfolder to create.  Relative paths are treated as relative to GetRootOutputFolder; absolute
        paths must be under this location.  Parent folders will be created as necessary.
        :param cleanFolder:  whether to wipe the folder contents if it already exists.
        """
        def CreateFolder(path):
            if not os.path.exists(path):
                head, tail = os.path.split(path)
                CreateFolder(head)
                os.mkdir(path)
                f = open(os.path.join(path, OutputFolder.SIG_FILE_NAME), 'w')
                f.close()
        self.path = OutputFolder.CheckOutputPath(path)
        if os.path.exists(self.path):
            if cleanFolder:
                self.RemoveOutputFolder(self.path)
        CreateFolder(self.path)

    @staticmethod
    def GetRootOutputFolder():
        """Get the root location where Chaste output files are stored.

        This is read from the environment variable CHASTE_TEST_OUTPUT; if it is not set then a folder 'testoutput' in
        the current working directory is used.
        """
        root_folder = os.environ.get('CHASTE_TEST_OUTPUT', '/tmp/chaste_test_output')
        if not os.path.isabs(root_folder):
            root_folder = os.path.join(os.getcwd(), root_folder)
        return os.path.realpath(root_folder)

    def GetAbsolutePath(self):
        """Get the absolute path to this output folder."""
        return self.path

    def CreateSubfolder(self, path):
        """Create a new OutputFolder inside this one.

        :param path:  the name of the subfolder to create.  This must be a relative path.
        """
        if os.path.isabs(path):
            raise ProtocolError('The path for the subfolder must be a relative path.')
        return OutputFolder(os.path.join(self.path, path))

    @staticmethod
    def RemoveOutputFolder(path):
        """Remove an existing output folder.

        This method will only delete folders living under the root output folder.  In addition, they must have been
        created using the OutputFolder class (this is indicated by the presence of a signature file within the folder).

        :param path:  the folder to remove.  Relative paths are treated as relative to GetRootOutputFolder; absolute
        paths must be under this location.
        """
        abs_path = OutputFolder.CheckOutputPath(path)
        if os.path.isfile(abs_path + '/' + OutputFolder.SIG_FILE_NAME):
            shutil.rmtree(abs_path)
        else:
            raise ProtocolError("Folder cannot be removed because it was not created via the OutputFolder class.")

    @staticmethod
    def CheckOutputPath(path):
        """Check whether the given path is a location within the Chaste output folder."""
        # if path.startswith(OutputFolder.GetRootOutputFolder()):
        if os.path.isabs(path):
            abs_path = path
        else:
            abs_path = os.path.join(OutputFolder.GetRootOutputFolder(), path)
        abs_path = os.path.realpath(abs_path)
        if not abs_path.startswith(OutputFolder.GetRootOutputFolder()):
            raise ProtocolError('Cannot alter the directory or file in this path.')
        return abs_path
