import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import EmbeddingRequest, EmbeddingResult  # noqa: E402
from heta_framework.common.stores import (  # noqa: E402
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
    VectorQuery,
)
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.parsing import ParsedSource  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    BuildLightRAGGraph,
    BuildLightRAGGraphConfig,
    LightRAGTableNames,
    LightRAGVectorCollections,
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
        vectors = []
        for text in request.texts:
            vectors.append([float(len(text)), float(text.count("Alice")), 1.0])
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


def _chunk() -> ParsedChunk:
    return ParsedChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        source=ParsedSource(
            key="raw/alice.txt",
            name="alice.txt",
            file_type="txt",
            content_sha256="a" * 64,
        ),
        page_index=0,
        chunk_index=0,
        text="Alice collaborates with Bob.",
        token_start=0,
        token_end=6,
        parent_chunk_ids=(),
    )


def _node_payload():
    return {
        "id": "Alice",
        "labels": ["Entity", "person"],
        "properties": {
            "name": "Alice",
            "entity_name": "Alice",
            "entity_type": "person",
            "description": "Alice is a collaborator.",
            "source_id": "chunk_1",
            "source_ids": ["chunk_1"],
            "file_path": "alice.txt",
            "file_paths": ["alice.txt"],
            "extraction_format": "json",
        },
    }


def _edge_payload():
    return {
        "id": "Alice--RELATED--Bob",
        "source_id": "Alice",
        "target_id": "Bob",
        "type": "RELATED",
        "properties": {
            "src_id": "Alice",
            "tgt_id": "Bob",
            "description": "Alice collaborates with Bob.",
            "keywords": "collaboration",
            "weight": 1.0,
            "source_id": "chunk_1",
            "source_ids": ["chunk_1"],
            "file_path": "alice.txt",
            "file_paths": ["alice.txt"],
            "extraction_format": "json",
        },
    }


async def _put_inputs(object_store, context):
    await object_store.put(
        "light_rag/graph/nodes/alice.json",
        json.dumps(_node_payload(), ensure_ascii=False).encode("utf-8"),
    )
    await object_store.put(
        "light_rag/graph/edges/alice_bob.json",
        json.dumps(_edge_payload(), ensure_ascii=False).encode("utf-8"),
    )
    await object_store.put("chunks/chunk_1.json", _chunk().to_json_bytes())
    context.set_artifact(
        "light_rag_graph_node_keys",
        ("light_rag/graph/nodes/alice.json",),
    )
    context.set_artifact(
        "light_rag_graph_edge_keys",
        ("light_rag/graph/edges/alice_bob.json",),
    )
    context.set_artifact("chunk_keys", ("chunks/chunk_1.json",))


def test_build_lightrag_graph_declares_capabilities():
    step = BuildLightRAGGraph()

    assert step.name == "build_lightrag_graph"
    assert {ref.key for ref in step.requirements.components} == {
        "models.embedding",
        "stores.objects",
        "stores.sql",
        "stores.vector",
    }
    assert step.requirements.artifacts == frozenset(
        {"light_rag_graph_node_keys", "light_rag_graph_edge_keys", "chunk_keys"}
    )
    assert step.capabilities.artifacts == frozenset({"build_light_rag_graph_result"})
    assert step.capabilities.queries == frozenset(
        {
            "light_rag_local_query",
            "light_rag_global_query",
            "light_rag_hybrid_query",
            "light_rag_mix_query",
        }
    )
    assert [asset.kind for asset in step.capabilities.search_assets] == [
        "light_rag_tables",
        "light_rag_vector_index",
    ]
    assert step.capabilities.search_assets[0].metadata["relations_table"] == (
        "light_rag_relations"
    )
    assert step.capabilities.search_assets[1].metadata["relationship_collection"] == (
        "light_rag_relationships"
    )
    assert step.capabilities.search_assets[1].metadata["chunk_collection"] == (
        "light_rag_chunks"
    )


def test_build_lightrag_graph_writes_sql_and_vectors(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    sql_store = SQLStore("sqlite:///:memory:")
    vector_store = InMemoryVectorStore()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.sql": sql_store,
            "stores.vector": vector_store,
            "models.embedding": FakeEmbeddingModel(),
        }
    )
    config = BuildLightRAGGraphConfig(
        table_names=LightRAGTableNames(
            entities="test_lr_entities",
            relations="test_lr_relations",
            chunks="test_lr_chunks",
        ),
        vector_collections=LightRAGVectorCollections(
            entities="test_lr_entity_vectors",
            relationships="test_lr_relationship_vectors",
            chunks="test_lr_chunk_vectors",
        ),
    )

    async def run():
        await _put_inputs(object_store, context)
        await BuildLightRAGGraph(config).run(context)
        entity_rows = await sql_store.fetch_all("SELECT * FROM test_lr_entities")
        relation_rows = await sql_store.fetch_all("SELECT * FROM test_lr_relations")
        chunk_rows = await sql_store.fetch_all("SELECT * FROM test_lr_chunks")
        entity_vector_count = await vector_store.count("test_lr_entity_vectors")
        relationship_vector_count = await vector_store.count(
            "test_lr_relationship_vectors"
        )
        chunk_vector_count = await vector_store.count("test_lr_chunk_vectors")
        entity_hits = await vector_store.search(
            "test_lr_entity_vectors",
            VectorQuery(vector=[10.0, 1.0, 1.0], top_k=1),
        )
        relationship_hits = await vector_store.search(
            "test_lr_relationship_vectors",
            VectorQuery(vector=[10.0, 1.0, 1.0], top_k=1),
        )
        chunk_hits = await vector_store.search(
            "test_lr_chunk_vectors",
            VectorQuery(vector=[10.0, 1.0, 1.0], top_k=1),
        )
        return (
            entity_rows,
            relation_rows,
            chunk_rows,
            entity_vector_count,
            relationship_vector_count,
            chunk_vector_count,
            entity_hits,
            relationship_hits,
            chunk_hits,
            context.artifacts["build_light_rag_graph_result"],
        )

    try:
        (
            entity_rows,
            relation_rows,
            chunk_rows,
            entity_vector_count,
            relationship_vector_count,
            chunk_vector_count,
            entity_hits,
            relationship_hits,
            chunk_hits,
            result,
        ) = asyncio.run(run())
    finally:
        asyncio.run(sql_store.aclose())

    assert result.entity_count == 1
    assert result.relation_count == 1
    assert result.chunk_count == 1
    assert result.entity_vector_count == 1
    assert result.relationship_vector_count == 1
    assert result.chunk_vector_count == 1
    assert result.vector_dimension == 3

    assert entity_rows[0]["entity_id"] == "Alice"
    assert entity_rows[0]["entity_name"] == "Alice"
    assert entity_rows[0]["entity_type"] == "person"
    assert entity_rows[0]["description"] == "Alice is a collaborator."
    assert entity_rows[0]["source_id"] == "chunk_1"
    assert json.loads(entity_rows[0]["source_ids"]) == ["chunk_1"]
    assert entity_rows[0]["file_path"] == "alice.txt"

    assert relation_rows[0]["relation_id"] == "Alice--RELATED--Bob"
    assert relation_rows[0]["source_entity_id"] == "Alice"
    assert relation_rows[0]["target_entity_id"] == "Bob"
    assert relation_rows[0]["keywords"] == "collaboration"
    assert relation_rows[0]["weight"] == 1.0
    assert json.loads(relation_rows[0]["source_ids"]) == ["chunk_1"]

    assert chunk_rows[0]["chunk_id"] == "chunk_1"
    assert chunk_rows[0]["content"] == "Alice collaborates with Bob."

    assert entity_vector_count == 1
    assert relationship_vector_count == 1
    assert chunk_vector_count == 1
    assert entity_hits[0].metadata["entity_name"] == "Alice"
    assert entity_hits[0].metadata["source_chunk_ids"] == ["chunk_1"]
    assert relationship_hits[0].metadata["src_id"] == "Alice"
    assert relationship_hits[0].metadata["tgt_id"] == "Bob"
    assert relationship_hits[0].metadata["keywords"] == "collaboration"
    assert relationship_hits[0].metadata["weight"] == 1.0
    assert chunk_hits[0].id == "chunk_1"
    assert chunk_hits[0].metadata["source_name"] == "alice.txt"
