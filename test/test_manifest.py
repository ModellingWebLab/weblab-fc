"""
Test COMBINE Archive manifest generation.
"""

from fc.file_handling import combine_manifest


HEADER = """<?xml version='1.0' encoding='utf-8'?>
<omexManifest xmlns='http://identifiers.org/combine.specifications/omex-manifest'>
    <content location='manifest.xml' format='http://identifiers.org/combine.specifications/omex-manifest'/>
"""
FOOTER = "</omexManifest>"
MIME_NS = "http://purl.org/NET/mediatypes/"
COMBINE_NS = "http://identifiers.org/combine.specifications/"
CONTENT_TPL = "    <content location='{}' format='{}'/>\n"


def test_empty_manifest():
    assert HEADER + FOOTER == combine_manifest([])


def test_cellml_file():
    actual = combine_manifest(['file.cellml'])
    expected = HEADER + CONTENT_TPL.format('file.cellml', COMBINE_NS + 'cellml') + FOOTER
    assert expected == actual


def test_csv_file():
    actual = combine_manifest(['file.csv'])
    expected = HEADER + CONTENT_TPL.format('file.csv', MIME_NS + 'text/csv') + FOOTER
    assert expected == actual


def test_multiple_files():
    actual = combine_manifest([
        'file1.cellml',
        'file2.txt',
        'file3.csv',
        'file4',
    ])
    expected = ''.join([
        HEADER,
        CONTENT_TPL.format('file1.cellml', COMBINE_NS + 'cellml'),
        CONTENT_TPL.format('file2.txt', MIME_NS + 'text/plain'),
        CONTENT_TPL.format('file3.csv', MIME_NS + 'text/csv'),
        CONTENT_TPL.format('file4', MIME_NS + 'application/octet-stream'),
        FOOTER
    ])
    assert expected == actual
