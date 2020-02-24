import os

import pytest

import fc


def test_sim_environments():
    """Run the test_sim_environments.txt test protocol."""
    proto = fc.Protocol('test/protocols/test_sim_environments.txt')
    proto.set_output_folder('test_sim_environments')
    proto.set_model('test/models/luo_rudy_1991.cellml')
    proto.run()
    # Test assertions are within the protocol itself. Here we just check it produced output.
    assert os.path.exists(os.path.join(proto.output_folder.path, 'output.h5'))
