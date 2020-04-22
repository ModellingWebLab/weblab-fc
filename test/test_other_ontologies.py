"""
Test annotating with other ontologies than oxmeta.

Also tests annotating a category of outputs.
"""
import pytest

import fc


def test_annotating_with_other_ontologies():
    proto_file = 'test/protocols/test_other_ontologies.txt'
    proto = fc.Protocol(proto_file)
    proto.set_output_folder('test_annotating_with_other_ontologies')
    proto.set_model('test/data/test_lr91.cellml')
    proto.run()
