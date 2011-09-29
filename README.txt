This file is still to be written properly.
It will explain how to use this project along with the Chaste source.

You need:
 * A Chaste source tree somewhere you can write to, ideally on local disk and not on an NFS mount (this can cause issues)
 * This project located at <CHASTE_SOURCE>/projects/FunctionalCuration
 * All the Chaste source dependencies, including those marked as 'cardiac only'
   * Boost 1.39 or above is required for this project

Source code is in the 'src' folder, and tests in 'tests'.  Annotated CellML files are in 'cellml'.
Some interesting locations:
 * tests/protocols  contains example protocols
 * src/proto/library/  contains protocol libraries with functions available to all protocols
 * src/proto/parsing/protocol.rnc  is a schema for the protocol XML language

Run the simulations from the paper with:
  scons cl=1 b=GccOptNative ts=projects/FunctionalCuration/test/TestFunctionalCurationPaper.hpp
Output will appear in /tmp/$USER/testoutput/FunctionalCuration
(unless CHASTE_TEST_OUTPUT is set elsewhere; it defaults to /tmp/$USER/testoutput.  This location
 should probably also be on local disk)

Run all the default project tests with:
  scons cl=1 b=GccOptNative projects/FunctionalCuration

To build an executable that can run a single protocol on a single model, do:
  scons cl=1 exe=1 projects/FunctionalCuration/apps
The executable will appear at projects/FunctionalCuration/apps/src/FunctionalCuration.
You'll need LD_LIBRARY_PATH set up to run it.