import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import EmbeddingRequest, EmbeddingResult  # noqa: E402
from heta_framework.common.stores import (  # noqa: E402
    InMemoryGraphStore,
    InMemoryVectorStore,
    SQLStore,
    VectorQuery,
)
from heta_framework.kb.steps import (  # noqa: E402
    BuildLeanRAGGraph,
    BuildLeanRAGGraphConfig,
    LeanRAGTableNames,
    LeanRAGVectorCollections,
)


class FakeContext:
    def __init__(self, components):
        self.components = components
        self.artifacts = {}

    def get_component(self, key):
        return self.components[key]

    def get_artifact(self, key):
        return self.artifacts[key]

    def set_artifact(self, key, value):
        self.artifacts[key] = value


class FakeEmbeddingModel:
    @property
    def model_name(self):
        return "fake-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[[float(len(text)), float(text.count("Alice")), 1.0] for text in request.texts],
            model_name=self.model_name,
        )

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


def _config():
    return BuildLeanRAGGraphConfig(
        table_names=LeanRAGTableNames(
            entities="test_lean_entities",
            relations="test_lean_relations",
            communities="test_lean_communities",
            chunks="test_lean_chunks",
        ),
        vector_collections=LeanRAGVectorCollections(entities="test_lean_vectors"),
    )


def _context():
    return FakeContext(
        {
            "stores.graph": InMemoryGraphStore(),
            "stores.sql": SQLStore("sqlite:///:memory:"),
            "stores.vector": InMemoryVectorStore(),
            "models.embedding": FakeEmbeddingModel(),
        }
    )


def _put_artifacts(context):
    context.set_artifact(
        "lean_rag_chunks",
        [
            {
                "chunk_id": "chunk_1",
                "hash_code": "hash_1",
                "document_id": "doc_1",
                "text": "Alice collaborates with Bob.",
                "content": "Alice collaborates with Bob.",
                "source_key": "raw/alice.txt",
                "source_name": "alice.txt",
                "source_file_type": "txt",
                "chunk_order_index": 0,
                "token_start": 0,
                "token_end": 5,
                "token_count": 5,
                "metadata": {"page": 0},
            }
        ],
    )
    context.set_artifact(
        "lean_rag_base_relations",
        [
            {
                "src_tgt": "Alice",
                "tgt_src": "Bob",
                "description": "Alice is related to Bob.",
                "weight": 1,
                "source_id": "hash_1",
                "source_ids": ["hash_1"],
                "level": 0,
                "is_generated": False,
            }
        ],
    )
    context.set_artifact(
        "lean_rag_all_entities_layers",
        [
            [
                {
                    "entity_name": "Alice",
                    "entity_type": "person",
                    "description": "Alice appears.",
                    "source_id": "hash_1",
                    "source_ids": ["hash_1"],
                    "degree": 1,
                    "parent": "Alice Group",
                    "level": 0,
                    "is_aggregate": False,
                },
                {
                    "entity_name": "Bob",
                    "entity_type": "person",
                    "description": "Bob appears.",
                    "source_id": "hash_1",
                    "source_ids": ["hash_1"],
                    "degree": 1,
                    "parent": "Alice Group",
                    "level": 0,
                    "is_aggregate": False,
                },
            ],
            [
                {
                    "entity_name": "Alice Group",
                    "entity_type": "aggregate entity",
                    "description": "Alice and Bob group.",
                    "source_id": "hash_1",
                    "source_ids": ["hash_1"],
                    "degree": 1,
                    "parent": "root",
                    "level": 1,
                    "is_aggregate": True,
                    "children": ["Alice", "Bob"],
                    "findings": [{"summary": "Link", "explanation": "They are linked."}],
                }
            ],
        ],
    )
    context.set_artifact(
        "lean_rag_generated_relations",
        [
            {
                "src_tgt": "Alice Group",
                "tgt_src": "Other Group",
                "description": "Generated relation.",
                "weight": 1,
                "source_id": "hash_1",
                "source_ids": ["hash_1"],
                "level": 1,
                "is_generated": True,
                "evidence_relation_ids": ["base_1"],
            }
        ],
    )
    context.set_artifact(
        "lean_rag_communities",
        [
            {
                "entity_name": "Alice Group",
                "entity_description": "Alice and Bob group.",
                "findings": [{"summary": "Link", "explanation": "They are linked."}],
                "level": 0,
                "children": ["Alice", "Bob"],
                "source_id": "hash_1",
                "source_ids": ["hash_1"],
            }
        ],
    )
    context.set_artifact(
        "lean_rag_parent_edges",
        [
            {"src_tgt": "Alice", "tgt_src": "Alice Group", "level": 0},
            {"src_tgt": "Bob", "tgt_src": "Alice Group", "level": 0},
            {"src_tgt": "Alice Group", "tgt_src": "root", "level": 1},
        ],
    )


def test_build_leanrag_graph_declares_capabilities():
    step = BuildLeanRAGGraph()

    assert step.name == "build_leanrag_graph"
    assert {ref.key for ref in step.requirements.components} == {
        "stores.graph",
        "stores.sql",
        "stores.vector",
        "models.embedding",
    }
    assert "lean_rag_query" in step.capabilities.queries
    assert [asset.kind for asset in step.capabilities.search_assets] == [
        "leanrag_tables",
        "leanrag_vector_index",
    ]


def test_build_leanrag_graph_writes_sql_vectors_graph_and_exports():
    context = _context()
    _put_artifacts(context)

    async def run():
        await BuildLeanRAGGraph(_config()).run(context)
        sql_store = context.components["stores.sql"]
        vector_store = context.components["stores.vector"]
        graph_store = context.components["stores.graph"]
        entity_rows = await sql_store.fetch_all("SELECT * FROM test_lean_entities ORDER BY level, entity_name")
        relation_rows = await sql_store.fetch_all("SELECT * FROM test_lean_relations ORDER BY level")
        community_rows = await sql_store.fetch_all("SELECT * FROM test_lean_communities")
        chunk_rows = await sql_store.fetch_all("SELECT * FROM test_lean_chunks")
        vector_count = await vector_store.count("test_lean_vectors")
        aggregate_hits = await vector_store.search(
            "test_lean_vectors",
            VectorQuery(vector=[20.0, 1.0, 1.0], top_k=5, filter={"is_aggregate": True}),
        )
        base_hits = await vector_store.search(
            "test_lean_vectors",
            VectorQuery(vector=[20.0, 1.0, 1.0], top_k=5, filter={"level": 0}),
        )
        parent = next(
            (
                edge
                for edge in graph_store.edges.values()
                if edge.type == "LEANRAG_PARENT"
                and edge.source_id == "Alice"
                and edge.target_id == "Alice Group"
            ),
            None,
        )
        return (
            entity_rows,
            relation_rows,
            community_rows,
            chunk_rows,
            vector_count,
            aggregate_hits,
            base_hits,
            parent,
        )

    (
        entity_rows,
        relation_rows,
        community_rows,
        chunk_rows,
        vector_count,
        aggregate_hits,
        base_hits,
        parent,
    ) = asyncio.run(run())

    assert len(entity_rows) == 3
    assert entity_rows[0]["parent"] == "Alice Group"
    assert json.loads(entity_rows[2]["children"]) == ["Alice", "Bob"]
    assert len(relation_rows) == 2
    assert relation_rows[1]["is_generated"] == 1
    assert json.loads(relation_rows[1]["evidence_relation_ids"]) == ["base_1"]
    assert community_rows[0]["entity_name"] == "Alice Group"
    assert chunk_rows[0]["hash_code"] == "hash_1"
    assert vector_count == 3
    assert [hit.metadata["entity_name"] for hit in aggregate_hits] == ["Alice Group"]
    assert {hit.metadata["entity_name"] for hit in base_hits} == {"Alice", "Bob"}
    assert parent is not None
    assert context.artifacts["lean_rag_all_entities_json"].splitlines()[0].startswith("[")
    assert context.artifacts["lean_rag_generate_relations_json"].startswith("{")
    assert context.artifacts["lean_rag_community_json"].startswith("{")
