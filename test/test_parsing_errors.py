
import pytest

import fc


def test_non_existent():
    proto_file = 'protocols/not_here.txt'
    with pytest.raises(FileNotFoundError):
        fc.Protocol(proto_file)

def test_non_existent_library():
    proto_file = 'test/protocols/test_non_existent_library.txt'
    with pytest.raises(FileNotFoundError):
        fc.Protocol(proto_file)

def test_protocol_error():
    proto_file = 'test/protocols/test_error_msg.txt'
    proto = fc.Protocol(proto_file)
    with pytest.raises(fc.error_handling.ErrorRecorder):
        proto.run()
