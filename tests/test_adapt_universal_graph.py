import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb.steps.adapt_universal_graph import _attach_document_provenance  # noqa: E402


def test_attach_document_provenance_maps_entity_sources_to_document_tokens():
    entities = [
        {
            "entity_name": "Alice",
            "source_id": "chunk_1|chunk_2",
            "source_ids": ["chunk_1", "chunk_2"],
            "properties": {"base_graph_source": "universal_graph_extraction"},
        }
    ]
    chunks = [
        {
            "chunk_id": "chunk_1",
            "hash_code": "hash_1",
            "document_id": "doc_1",
            "source_name": "one.txt",
        },
        {
            "chunk_id": "chunk_2",
            "hash_code": "hash_2",
            "document_id": "doc_2",
            "source_name": "two.txt",
        },
    ]

    [record] = _attach_document_provenance(
        entities,
        chunks=chunks,
        document_token_counts={"doc_1": 10, "doc_2": 20},
    )

    assert record["documents"] == ["doc_1", "doc_2"]
    assert record["document_names"] == {"doc_1": "one.txt", "doc_2": "two.txt"}
    assert record["document_tokens"] == {"doc_1": 10, "doc_2": 20}
    assert record["document_token_count"] == 30
    assert record["document_details"] == [
        {"document_id": "doc_1", "document_name": "one.txt", "document_token_count": 10},
        {"document_id": "doc_2", "document_name": "two.txt", "document_token_count": 20},
    ]
    assert record["properties"]["document_tokens"] == {"doc_1": 10, "doc_2": 20}
