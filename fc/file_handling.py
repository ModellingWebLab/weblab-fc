"""
Methods for file-based IO.
"""
import mimetypes
import os
import shutil

import numpy as np
import tables

from .error_handling import ProtocolError


class OutputFolder(object):
    """
    Manages creation and safe clean-up of folders for protocol output to be written to, in a central location defined by
    the environment variable ``CHASTE_TEST_OUTPUT``.

    This class contains routines providing the key functionality of the C++ classes OutputFileHandler and FileFinder.
    In particular, it allows the creation of output folders within the location set for Chaste output, and safe deletion
    of such folders, without risk of wiping a user's drive due to coding error or dodgy path settings.
    """
    SIG_FILE_NAME = '.chaste_deletable_folder'

    def __init__(self, path, clean_folder=True):
        """Create a new output subfolder.

        :param path:  the subfolder to create.  Relative paths are treated as relative to get_root_output_folder;
        absolute paths must be under this location.  Parent folders will be created as necessary.
        :param clean_folder:  whether to wipe the folder contents if it already exists.
        """
        def create_folder(path):
            if not os.path.exists(path):
                head, tail = os.path.split(path)
                create_folder(head)
                os.mkdir(path)
                f = open(os.path.join(path, OutputFolder.SIG_FILE_NAME), 'w')
                f.close()
        self.path = OutputFolder.check_output_path(path)
        if os.path.exists(self.path):
            if clean_folder:
                self.remove_output_folder(self.path)
        create_folder(self.path)

    @staticmethod
    def get_root_output_folder():
        """Get the root location where Chaste output files are stored.

        This is read from the environment variable CHASTE_TEST_OUTPUT; if it is not set then '/tmp/chaste_test_output'
        is used.
        """
        root_folder = os.environ.get('CHASTE_TEST_OUTPUT', '/tmp/chaste_test_output')
        if not os.path.isabs(root_folder):
            root_folder = os.path.join(os.getcwd(), root_folder)
        return os.path.realpath(root_folder)

    def get_absolute_path(self):
        """Get the absolute path to this output folder."""
        return self.path

    def create_subfolder(self, path):
        """Create a new OutputFolder inside this one.

        :param path:  the name of the subfolder to create.  This must be a relative path.
        """
        if os.path.isabs(path):
            raise ProtocolError('The path for the subfolder must be a relative path.')
        return OutputFolder(os.path.join(self.path, path))

    @staticmethod
    def remove_output_folder(path):
        """Remove an existing output folder.

        This method will only delete folders living under the root output folder.  In addition, they must have been
        created using the OutputFolder class (this is indicated by the presence of a signature file within the folder).

        :param path:  the folder to remove.  Relative paths are treated as relative to get_root_output_folder; absolute
        paths must be under this location.
        """
        abs_path = OutputFolder.check_output_path(path)
        if os.path.isfile(abs_path + '/' + OutputFolder.SIG_FILE_NAME):
            shutil.rmtree(abs_path)
        else:
            raise ProtocolError("Folder cannot be removed because it was not created via the OutputFolder class.")

    @staticmethod
    def check_output_path(path):
        """Check whether the given path is a location within the Chaste output folder."""
        # if path.startswith(OutputFolder.get_root_output_folder()):
        if os.path.isabs(path):
            abs_path = path
        else:
            abs_path = os.path.join(OutputFolder.get_root_output_folder(), path)
        abs_path = os.path.realpath(abs_path)
        if not abs_path.startswith(OutputFolder.get_root_output_folder()):
            raise ProtocolError('Cannot alter the directory or file in this path.')
        return abs_path


def sanitise_file_name(name):
    """Simply transform a name such as a graph title into a valid file name."""
    name = name.strip().replace(' ', '_')
    keep = ('.', '_')
    return ''.join(c for c in name if c.isalnum() or c in keep)


def combine_manifest(namelist):
    """Generate a COMBINE Archive manifest for a given collection of file names.

    This uses Python's mimetypes library to deduce for most extensions, with some COMBINE-specific file types
    supported specially.

    :param namelist: an iterable of file names
    :return: the contents of a manifest.xml file, as a string
    """
    mimetypes.add_type('text/csv', '.csv')  # Make csv mapping explicit (in Windows, defaults to Excel)
    manifest = [
        "<?xml version='1.0' encoding='utf-8'?>",
        "<omexManifest xmlns='http://identifiers.org/combine.specifications/omex-manifest'>",
        "    <content location='manifest.xml' format='http://identifiers.org/combine.specifications/omex-manifest'/>",
    ]
    for filename in namelist:
        try:
            ext = os.path.splitext(filename)[1]
            combine_type = {
                '.cellml': 'http://identifiers.org/combine.specifications/cellml',
            }[ext]
        except Exception:
            combine_type = mimetypes.guess_type(filename)[0]
        if combine_type is None:
            combine_type = 'application/octet-stream'
        if not combine_type.startswith('http'):
            combine_type = 'http://purl.org/NET/mediatypes/' + combine_type
        manifest.append(f"    <content location='{filename}' format='{combine_type}'/>")
    manifest.append("</omexManifest>")
    return '\n'.join(manifest)


def extract_output(h5_path_or_file, output_name, output_folder=None):
    """Extract a single protocol output from an HDF5 file to CSV.

    :param h5_path_or_file: either the path to an HDF5 file, or an already open ``tables.File``.
    :param output_name: the name of the protocol output to extract, which will also be used as the basis for the CSV
        file name.
    :param output_folder: if given, the folder to write the CSV in; otherwise it will be written to the same folder
        as the HDF5 file.
    """
    def extract(h5_file, output_folder):
        if output_folder is None:
            output_folder = os.path.dirname(h5_file.filename)
        out_path = os.path.join(output_folder, 'outputs_' + sanitise_file_name(output_name) + '.csv')

        data = h5_file.root.output._f_get_child(output_name).read()
        np.savetxt(out_path, data.transpose(), delimiter=',')

    if isinstance(h5_path_or_file, tables.File):
        extract(h5_path_or_file, output_folder)
    else:
        with tables.open_file(h5_path_or_file, 'r') as h5_file:
            extract(h5_file, output_folder)
