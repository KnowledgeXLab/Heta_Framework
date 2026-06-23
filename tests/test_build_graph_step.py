import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import EmbeddingRequest, EmbeddingResult  # noqa: E402
from heta_framework.common.stores import (  # noqa: E402
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
    VectorQuery,
)
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation  # noqa: E402
from heta_framework.kb import KnowledgeBase, KnowledgeModels, KnowledgeRecipe, KnowledgeStores  # noqa: E402
from heta_framework.kb.parsing import ParsedSource  # noqa: E402
from heta_framework.kb.steps import BuildGraph, BuildGraphConfig, GraphTableNames  # noqa: E402


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
            vectors.append([float(len(text)), float(text.count("上海")), 1.0])
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


def _graph_table_names(prefix: str) -> GraphTableNames:
    return GraphTableNames(
        entities=f"{prefix}_entities",
        relations=f"{prefix}_relations",
        evidence=f"{prefix}_graph_evidence",
    )


def _chunk(chunk_id: str = "chunk_1") -> ParsedChunk:
    return ParsedChunk(
        chunk_id=chunk_id,
        document_id="doc_1",
        source=ParsedSource(
            key="raw/paper.pdf",
            name="paper.pdf",
            file_type="pdf",
            content_sha256="a" * 64,
        ),
        page_index=0,
        chunk_index=0,
        text="上海市包含徐汇区。",
        token_start=0,
        token_end=9,
    )


def _entity(entity_id: str, name: str, *, chunk_id: str = "chunk_1") -> ExtractedEntity:
    return ExtractedEntity(
        entity_id=entity_id,
        chunk_id=chunk_id,
        document_id="doc_1",
        name=name,
        type="客观实体",
        subtype="行政区划",
        description=f"{name} 是一个行政区划实体。",
        attributes={"source": "test"},
        source_chunk_ids=(chunk_id,),
    )


def _relation(
    relation_id: str,
    source_entity_id: str,
    target_entity_id: str,
    *,
    chunk_id: str = "chunk_1",
) -> ExtractedRelation:
    return ExtractedRelation(
        relation_id=relation_id,
        chunk_id=chunk_id,
        document_id="doc_1",
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        source_entity_name="上海市",
        target_entity_name="徐汇区",
        type="空间关系",
        name="包含行政区",
        description="徐汇区是上海市下辖的市辖区。",
        attributes={"confidence": "high"},
        source_chunk_ids=(chunk_id,),
    )


async def _put_graph_inputs(
    object_store,
    context,
    *,
    entity_keys_artifact="deduplicated_entity_keys",
    relation_keys_artifact="deduplicated_relation_keys",
):
    entities = (_entity("entity_shanghai", "上海市"), _entity("entity_xuhui", "徐汇区"))
    relation = _relation("relation_contains", "entity_shanghai", "entity_xuhui")
    chunk = _chunk()
    entity_keys = []
    for entity in entities:
        key = f"deduplicated_entities/{entity.entity_id}.json"
        await object_store.put(key, entity.to_json_bytes())
        entity_keys.append(key)
    relation_key = f"deduplicated_relations/{relation.relation_id}.json"
    await object_store.put(relation_key, relation.to_json_bytes())
    await object_store.put("chunks/chunk_1.json", chunk.to_json_bytes())
    context.set_artifact(entity_keys_artifact, tuple(entity_keys))
    context.set_artifact(relation_keys_artifact, (relation_key,))
    context.set_artifact("chunk_keys", ("chunks/chunk_1.json",))


def test_build_graph_declares_requirements_and_capabilities():
    step = BuildGraph()

    assert step.name == "build_graph"
    assert {ref.key for ref in step.requirements.components} == {
        "models.embedding",
        "stores.objects",
        "stores.sql",
        "stores.vector",
    }
    assert step.requirements.artifacts == frozenset(
        {"deduplicated_entity_keys", "deduplicated_relation_keys", "chunk_keys"}
    )
    assert step.capabilities.artifacts == frozenset({"build_graph_result"})
    assert step.capabilities.queries == frozenset({"heta_graph_search"})
    assert [asset.kind for asset in step.capabilities.search_assets] == [
        "graph_tables",
        "graph_vector_index",
    ]
    assert step.capabilities.search_assets[0].metadata["entities_table"] == "entities"
    assert step.capabilities.search_assets[1].metadata["entity_collection"] == "graph_entities"


def test_build_graph_writes_heta_style_sql_tables(tmp_path):
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

    async def run():
        await _put_graph_inputs(object_store, context)
        await BuildGraph(
            BuildGraphConfig(table_names=_graph_table_names("papers"))
        ).run(context)
        entity_rows = await sql_store.fetch_all(
            "SELECT * FROM papers_entities ORDER BY entity_id"
        )
        relation_rows = await sql_store.fetch_all("SELECT * FROM papers_relations")
        evidence_rows = await sql_store.fetch_all(
            "SELECT * FROM papers_graph_evidence ORDER BY fact_type, fact_id"
        )
        entity_vectors = await vector_store.count("graph_entities")
        relation_vectors = await vector_store.count("graph_relations")
        entity_hits = await vector_store.search(
            "graph_entities",
            VectorQuery(vector=[10.0, 1.0, 1.0], top_k=2),
        )
        relation_hits = await vector_store.search(
            "graph_relations",
            VectorQuery(vector=[10.0, 1.0, 1.0], top_k=2),
        )
        return (
            entity_rows,
            relation_rows,
            evidence_rows,
            entity_vectors,
            relation_vectors,
            entity_hits,
            relation_hits,
            context.artifacts["build_graph_result"],
        )

    try:
        (
            entity_rows,
            relation_rows,
            evidence_rows,
            entity_vectors,
            relation_vectors,
            entity_hits,
            relation_hits,
            result,
        ) = asyncio.run(run())
    finally:
        asyncio.run(sql_store.aclose())

    assert result.entity_count == 2
    assert result.relation_count == 1
    assert result.evidence_count == 3
    assert result.entity_vector_count == 2
    assert result.relation_vector_count == 1
    assert result.vector_dimension == 3
    assert result.skipped_evidence_count == 0
    assert entity_rows[0]["entity_id"] == "entity_shanghai"
    assert entity_rows[0]["entity_name"] == "上海市"
    assert entity_rows[0]["entity_type"] == "客观实体"
    assert entity_rows[0]["entity_subtype"] == "行政区划"
    assert json.loads(entity_rows[0]["attributes"]) == {"source": "test"}
    assert relation_rows[0]["relation_id"] == "relation_contains"
    assert relation_rows[0]["source_entity_id"] == "entity_shanghai"
    assert relation_rows[0]["target_entity_id"] == "entity_xuhui"
    assert relation_rows[0]["relation_type"] == "空间关系"
    assert relation_rows[0]["relation_name"] == "包含行政区"
    assert json.loads(relation_rows[0]["attributes"]) == {"confidence": "high"}
    assert {row["fact_type"] for row in evidence_rows} == {"entity", "relation"}
    assert {row["chunk_id"] for row in evidence_rows} == {"chunk_1"}
    assert {row["source_key"] for row in evidence_rows} == {"raw/paper.pdf"}
    assert entity_vectors == 2
    assert relation_vectors == 1
    assert {hit.metadata["fact_type"] for hit in entity_hits} == {"entity"}
    assert {hit.metadata["fact_type"] for hit in relation_hits} == {"relation"}


def test_knowledge_base_query_runs_heta_graph_search(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path)
        sql_store = SQLStore("sqlite:///:memory:")
        vector_store = InMemoryVectorStore()
        context = FakeContext({})
        await _put_graph_inputs(object_store, context)
        recipe = KnowledgeRecipe(
            models=KnowledgeModels(embedding=FakeEmbeddingModel()),
            stores=KnowledgeStores(
                objects=object_store,
                sql=sql_store,
                vector=vector_store,
            ),
            steps=(BuildGraph(),),
        )

        kb = await KnowledgeBase.create(
            recipe=recipe,
            name="graph-query-test",
            initial_artifacts=context.artifacts,
        )
        response = await kb.query("上海市", mode="heta_graph_search", top_k=4)
        await sql_store.aclose()
        return kb, response

    kb, response = asyncio.run(run())

    assert "heta_graph_search" in kb.available_queries
    assert response.mode == "heta_graph_search"
    assert {result.kind for result in response.results} >= {"entity", "relation"}
    relation = next(result for result in response.results if result.kind == "relation")
    assert relation.metadata["matched_by"] in {"entity_one_hop", "relation_vector"}
    assert relation.metadata["source_entity_name"] == "上海市"
    assert relation.metadata["target_entity_name"] == "徐汇区"


def test_build_graph_can_use_raw_extraction_artifacts(tmp_path):
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

    async def run():
        await _put_graph_inputs(
            object_store,
            context,
            entity_keys_artifact="entity_keys",
            relation_keys_artifact="relation_keys",
        )
        await BuildGraph(
            BuildGraphConfig(
                table_names=_graph_table_names("papers_raw"),
                entity_keys_artifact="entity_keys",
                relation_keys_artifact="relation_keys",
            )
        ).run(context)
        entity_count = await sql_store.fetch_one(
            "SELECT COUNT(*) AS count FROM papers_raw_entities"
        )
        relation_count = await sql_store.fetch_one(
            "SELECT COUNT(*) AS count FROM papers_raw_relations"
        )
        return entity_count, relation_count

    try:
        entity_count, relation_count = asyncio.run(run())
    finally:
        asyncio.run(sql_store.aclose())

    assert entity_count == {"count": 2}
    assert relation_count == {"count": 1}


def test_build_graph_skips_missing_evidence_chunks(tmp_path):
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

    async def run():
        entity = _entity("entity_shanghai", "上海市", chunk_id="missing_chunk")
        relation = _relation(
            "relation_missing",
            "entity_shanghai",
            "entity_xuhui",
            chunk_id="missing_chunk",
        )
        await object_store.put(
            "deduplicated_entities/entity_shanghai.json",
            entity.to_json_bytes(),
        )
        await object_store.put(
            "deduplicated_relations/relation_missing.json",
            relation.to_json_bytes(),
        )
        context.set_artifact(
            "deduplicated_entity_keys",
            ("deduplicated_entities/entity_shanghai.json",),
        )
        context.set_artifact(
            "deduplicated_relation_keys",
            ("deduplicated_relations/relation_missing.json",),
        )
        context.set_artifact("chunk_keys", ())
        await BuildGraph(
            BuildGraphConfig(table_names=_graph_table_names("papers"))
        ).run(context)
        evidence_count = await sql_store.fetch_one(
            "SELECT COUNT(*) AS count FROM papers_graph_evidence"
        )
        return context.artifacts["build_graph_result"], evidence_count

    try:
        result, evidence_count = asyncio.run(run())
    finally:
        asyncio.run(sql_store.aclose())

    assert result.entity_count == 1
    assert result.relation_count == 1
    assert result.evidence_count == 0
    assert result.entity_vector_count == 1
    assert result.relation_vector_count == 1
    assert evidence_count == {"count": 0}
    assert result.skipped_evidence_count == 2
    assert {issue.code for issue in result.issues} == {"missing_evidence_chunk"}


def test_build_graph_config_validates_values():
    with pytest.raises(ValueError, match="entity_keys_artifact"):
        BuildGraphConfig(entity_keys_artifact="")
    with pytest.raises(ValueError, match="batch_size"):
        BuildGraphConfig(batch_size=0)
    with pytest.raises(ValueError, match="SQL identifier"):
        BuildGraphConfig(table_names=GraphTableNames(entities="bad-name"))
