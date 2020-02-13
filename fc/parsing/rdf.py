"""
RDF handling routines, including parsing the 'oxmeta' ontology.
"""
import pkg_resources

import rdflib

from cellmlmanip.rdf import create_rdf_node


_ONTOLOGY = None  # The 'oxmeta' ontology graph

OXMETA_NS = 'https://chaste.comlab.ox.ac.uk/cellml/ns/oxford-metadata#'
BQBIOL_NS = 'http://biomodels.net/biology-qualifiers/'

PRED_IS = create_rdf_node((BQBIOL_NS, 'is'))
PRED_IS_VERSION_OF = create_rdf_node((BQBIOL_NS, 'isVersionOf'))


def get_variables_transitively(model, term):
    """Return a list of variables annotated (directly or otherwise) with the given ontology term.

    Direct annotations are those variables annotated with the term via the bqbiol:is or
    bqbiol:isVersionOf predicates.

    However we also look transitively through the 'oxmeta' ontology for terms belonging to the RDF
    class given by ``term``, i.e. are connected to it by a path of ``rdf:type`` predicates, and
    return variables annotated with those terms.

    :param term: the ontology term to search for. Can be anything suitable as input to
        :meth:`create_rdf_node`, typically a :class:`rdflib.term.Node` or ``(ns_uri, local_name)`` pair.
    :return: a list of :class:`VariableDummy` symbols, sorted by order added to the model.
    """
    global _ONTOLOGY

    if _ONTOLOGY is None:
        # Load oxmeta ontology
        g = _ONTOLOGY = rdflib.Graph()
        oxmeta_ttl = pkg_resources.resource_stream('fc', 'ontologies/oxford-metadata.ttl')
        g.parse(oxmeta_ttl, format='turtle')

    term = create_rdf_node(term)

    cmeta_ids = set()
    for annotation in _ONTOLOGY.transitive_subjects(rdflib.RDF.type, term):
        cmeta_ids.update(model.rdf.subjects(PRED_IS, annotation))
        cmeta_ids.update(model.rdf.subjects(PRED_IS_VERSION_OF, annotation))
    symbols = []
    for cmeta_id in cmeta_ids:
        symbols.append(model.get_symbol_by_cmeta_id(cmeta_id))
    return sorted(symbols, key=lambda sym: sym.order_added)

