"""Test models, simulations, ranges, and modifiers."""
import os
import pytest

import fc
from fc import test_support
from fc.simulations.solvers import CvodeSolver


@pytest.mark.skipif(os.getenv('FC_LONG_TESTS', '0') == '0', reason='FC_LONG_TESTS not set to 1')
@pytest.mark.xfail(strict=True, reason='no pycml replacement yet')
def test_s1_s2():
    proto = fc.Protocol('protocols/S1S2.txt')
    proto.set_output_folder('test_s1_s2')
    proto.set_model('test/models/courtemanche_ramirez_nattel_model_1998.cellml')
    proto.run()
    data_folder = 'test/data/TestSpeedRealProto/S1S2'
    test_support.check_results(
        proto,
        {'raw_APD90': 2, 'raw_DI': 2, 'max_S1S2_slope': 1},
        data_folder
    )


def test_s1_s2_lr91():

    # // Don't do too many runs
    # std::vector<AbstractExpressionPtr> s2_intervals
    #         = EXPR_LIST(CONST(1000))(CONST(900))(CONST(800))(CONST(700))(CONST(600))(CONST(500));
    # DEFINE(s2_intervals_expr, boost::make_shared<ArrayCreate>(s2_intervals));
    # runner.GetProtocol()->SetInput("s2_intervals", s2_intervals_expr);

    # s1_s2_intervals = [1000, 900, 800, 700, 600, 500]

    proto = fc.Protocol('test/protocols/test_S1S2.txt')
    proto.set_output_folder('test_s1_s2_lr91')
    proto.set_model('test/models/luo_rudy_1991.cellml')
    proto.run()
    data_folder = '/test/data/historic/luo_rudy_1991/S1S2'
    test_support.check_results(
        proto,
        {'APD90': 1, 'DI': 1, 'S1S2_slope': 1},   # Name and dimension of output to check
        data_folder
    )


def test_s1_s2_noble():
    """This model has time units in seconds, so we're checking that conversion works."""
    # std::string dirname = "TestS1S2ProtocolOutputs_EarmNobleModel";
    # FileFinder cellml_file("projects/FunctionalCuration/cellml/earm_noble_model_1990.cellml", RelativeTo::ChasteSourceRoot);
    # Assume we get to steady state quickly
    # ProtocolFileFinder proto_xml_file("projects/FunctionalCuration/test/protocols/xml/test_S1S2.xml", RelativeTo::ChasteSourceRoot);
    # DoTest(dirname, proto_xml_file, cellml_file, 0.0264);
    pass


'''
def do_test(rDirName,rProtocolFile, rCellmlFile, expectedSlope):
    """
        ProtocolRunner runner(rCellmlFile, rProtocolFile, rDirName, true);



        // Run
        runner.RunProtocol();
        FileFinder success_file(rDirName + "/success", RelativeTo::ChasteTestOutput);
        TS_ASSERT(success_file.Exists());

        // Check the max slope hasn't changed.
        const Environment& r_outputs = runner.GetProtocol()->rGetOutputsCollection();
        NdArray<double> max_slope = GET_ARRAY(r_outputs.Lookup("max_S1S2_slope"));
        TS_ASSERT_EQUALS(max_slope.GetNumElements(), 1u);
        TS_ASSERT_DELTA(*max_slope.Begin(), expectedSlope, 1e-3);

        // Check we did the right number of timesteps (overridden protocol input)
        NdArray<double> voltage = GET_ARRAY(r_outputs.Lookup("membrane_voltage"));
        TS_ASSERT_EQUALS(voltage.GetNumDimensions(), 2u);
        TS_ASSERT_EQUALS(voltage.GetShape()[0], s2_intervals.size());
        TS_ASSERT_EQUALS(voltage.GetShape()[1], 2001u);

        CheckManifest(rDirName, rCellmlFile.GetLeafNameNoExtension(), runner.GetProtocol());

    void CheckManifest(const std::string& rDirName, const std::string& rModelName, ProtocolPtr pProto)

        FileFinder manifest_file(rDirName + "/manifest.xml", RelativeTo::ChasteTestOutput);
        TS_ASSERT(manifest_file.Exists());
        std::map<std::string, std::string> ext_map = boost::assign::map_list_of
                ("eps", "application/postscript")
                ("csv", "text/csv")
                ("txt", "text/plain")
                ("cellml", "http://identifiers.org/combine.specifications/cellml.1.0")
                ("xml", "text/xml")
                ("hpp", "text/plain")
                ("cpp", "text/plain")
                ("gp", "text/plain")
                ("so", "application/octet-stream");
        std::set<std::string> entries {
                "machine_info_0.txt",
                "model_info.txt",
                "provenance_info_0.txt",
                "trace.txt",
                "outputs-contents.csv",
                "outputs-default-plots.csv",
                "Action_potential_traces.eps",
                "outputs_Action_potential_traces_gnuplot_data.csv",
                "outputs_Action_potential_traces_gnuplot_data.gp",
                "S1-S2_curve.eps",
                "outputs_S1-S2_curve_gnuplot_data.csv",
                "outputs_S1-S2_curve_gnuplot_data.gp",
                "outputs_APD90.csv",
                "outputs_DI.csv",
                "outputs_max_S1S2_slope.csv",
                "outputs_membrane_voltage.csv",
                "outputs_raw_APD90.csv",
                "outputs_raw_DI.csv",
                "outputs_S1S2_slope.csv",
                "outputs_s2_intervals.csv",
                "outputs_time_1d.csv",
                "outputs_PCLs.csv",
                "outputs_restitution_curve_gnuplot_data.csv",
                "restitution_curve.eps",
                "outputs_restitution_curve_gnuplot_data.gp"
                };
        entries.insert("lib" + rModelName + ".so");
        std::vector<std::string> suffixes {".cellml", "-conf.xml", ".cpp", ".hpp"};
        BOOST_FOREACH(std::string suffix, suffixes)
        {
            entries.insert(rModelName + suffix);
        }
        std::map<std::string,std::string> expected;
        expected["success"] = "text/plain";
        expected["manifest.xml"] = "http://identifiers.org/combine.specifications/omex-manifest";
        BOOST_FOREACH(const std::string& r_entry, entries)
        {
            std::size_t dot_pos = r_entry.rfind('.');
            std::string ext = r_entry.substr(dot_pos + 1);
            expected[r_entry] = ext_map[ext];
        }
        TS_ASSERT_EQUALS(expected, pProto->rGetManifest().rGetEntries());
    """
'''
