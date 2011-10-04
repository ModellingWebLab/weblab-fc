= Functional Curation Project =

This is a bolt-on project to Chaste accompanying the paper:

# J. Cooper, G. Mirams, S. Niederer.
High throughput functional curation of cellular electrophysiology models.
Prog Biophys Mol Biol, 2011. doi: 10.1016/j.pbiomolbio.2011.06.003.

A pre-print is available at:
http://www.cs.ox.ac.uk/chaste/publications/2011-Cooper-Functional%20Curation.pdf

At present it is known to be compatible with the trunk revision 13911 of Chaste.
A version compatible with the next release of Chaste will also be made available.

== Installation ==

This project requires the Chaste source tree to be installed in order to be usable.
Install the Chaste source following its installation instructions.  You will need
write access to the Chaste source tree in order to use this project.  Note that it
is advisable to install the source tree on local disk, rather than an NFS mount:
we have experienced occasional problems in the latter case, depending on Linux
distribution and version.  Compiling Chaste is also fairly disk intensive!  Note
also that you will require those Chaste dependencies marked as "cardiac only", and
that this project requires at least version 1.39 of the Boost libraries.

Having installed (and ideally tested!) the Chaste source tree, unpack this project
as <Chaste>/projects/FunctionalCuration.  It is crucial to match the folder name
and location, or the project will not work.

== Usage ==

Source code for the project is contained in the `src` folder, and tests of its
functionality in `tests`.  Annotated CellML files suitable for use with the framework
are in `cellml`.  Some additional interesting locations are:
 * `tests/protocols`  contains example protocols
 * `src/proto/library`  contains protocol libraries with functions available to all protocols
 * `src/proto/parsing/protocol.rnc`  is a schema for the protocol XML language

You can run the simulations from the paper using:
  scons cl=1 b=GccOptNative ts=projects/FunctionalCuration/test/TestFunctionalCurationPaper.hpp

Output will appear in /tmp/$USER/testoutput/FunctionalCuration by default (unless the
environment variable CHASTE_TEST_OUTPUT is set to point elsewhere; it defaults to
/tmp/$USER/testoutput.  This location should probably also be on local disk).  The
test should pass (printing an 'OK' line towards the end) to show that the protocol
results generated are consistent with those in the paper.*

Note that some warnings will be printed at the end of the test output.  The following
are expected, for model/protocol combinations where we cannot run the protocol to
completion.  These were not included in the paper.
 * beeler_reuter_model_1977 under ICaL protocol: no extracellular calcium
 * luo_rudy_1991 under ICaL protocol: no extracellular calcium
 * maleckar_model_2009 under ICaL protocol: CVODE fails to solve system with CV_CONV_FAILURE
 * nygren_atrial_model_1998 under S1S2 protocol: post-processing fails (irregular index)
   due to the model self-stimulating
 * nygren_atrial_model_1998 under ICaL protocol: CVODE fails to solve system with CV_CONV_FAILURE


There are also tests covering the lower-level functionality available for use by
protocols.  Run all the default tests with:
  scons cl=1 b=GccOptNative projects/FunctionalCuration


To build an executable that can run a single protocol on a single model, do:
  scons cl=1 exe=1 projects/FunctionalCuration/apps

The executable will appear at projects/FunctionalCuration/apps/src/FunctionalCuration.
You'll need the environment variable LD_LIBRARY_PATH set up as described in the Chaste
documentation in order to run it, since it needs to find the Chaste libraries and their
dependencies.

== Contact details ==

We cannot guarantee support for this project, but will endeavour to respond to queries.
Contact information for the corresponding authors can be found via
http://www.cs.ox.ac.uk/chaste/theteam.html

== Footnotes ==

* With the exception of one case where subsequent investigation has revealed the
results in the paper to be slightly incorrect, and 3 cases where changes to numerical
parameters have resulted in differences slightly larger than 0.5%; for these
model/protocol combinations the results for comparison have been altered.
