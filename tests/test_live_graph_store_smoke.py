import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import EmbeddingModel, EmbeddingRequest  # noqa: E402
from heta_framework.common.stores import (  # noqa: E402
    LocalObjectStore,
    MilvusVectorStore,
    SQLStore,
    VectorQuery,
)
from heta_framework.kb import (  # noqa: E402
    BuildGraph,
    BuildGraphConfig,
    GraphTableNames,
    GraphVectorCollections,
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeRecipe,
    KnowledgeStores,
)
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation  # noqa: E402
from heta_framework.kb.parsing import ParsedSource  # noqa: E402


pytestmark = pytest.mark.live


def test_live_heta_graph_build_writes_postgres_and_milvus(tmp_path: Path) -> None:
    if os.getenv("HETA_RUN_LIVE_GRAPH_SMOKE") != "1":
        pytest.skip("set HETA_RUN_LIVE_GRAPH_SMOKE=1 to run live graph store smoke")
    asyncio.run(_run_live_graph_build(tmp_path))


async def _run_live_graph_build(tmp_path: Path) -> None:
    config = _load_config()
    suffix = uuid.uuid4().hex[:12]
    table_names = GraphTableNames(
        entities=f"heta_live_entities_{suffix}",
        relations=f"heta_live_relations_{suffix}",
        evidence=f"heta_live_graph_evidence_{suffix}",
    )
    vector_collections = GraphVectorCollections(
        entities=f"heta_live_graph_entities_{suffix}",
        relations=f"heta_live_graph_relations_{suffix}",
    )

    object_store = LocalObjectStore(tmp_path / "objects")
    sql_store = SQLStore(_postgres_url(config))
    vector_store = MilvusVectorStore(
        uri=_milvus_uri(config),
        db_name=_milvus_db_name(config),
        timeout=float(os.getenv("HETA_LIVE_MILVUS_TIMEOUT", "10")),
    )
    embedding = _embedding_model(config)

    entity_keys, relation_keys, chunk_keys = await _put_graph_inputs(object_store)
    recipe = KnowledgeRecipe(
        models=KnowledgeModels(embedding=embedding),
        stores=KnowledgeStores(
            objects=object_store,
            sql=sql_store,
            vector=vector_store,
        ),
        steps=(
            BuildGraph(
                BuildGraphConfig(
                    table_names=table_names,
                    vector_collections=vector_collections,
                )
            ),
        ),
    )
    recipe.require_valid(
        initial_artifacts={
            "deduplicated_entity_keys",
            "deduplicated_relation_keys",
            "chunk_keys",
        }
    )

    try:
        kb = await KnowledgeBase.create(
            recipe=recipe,
            name="live-heta-graph-smoke",
            initial_artifacts={
                "deduplicated_entity_keys": entity_keys,
                "deduplicated_relation_keys": relation_keys,
                "chunk_keys": chunk_keys,
            },
        )

        assert kb.run_record.status == "succeeded", _format_step_errors(kb.run_record.step_records)
        assert kb.run_record.capabilities.queries == frozenset({"heta_graph_search"})

        entity_count = await sql_store.fetch_one(
            f"SELECT COUNT(*) AS count FROM {table_names.entities}"
        )
        relation_count = await sql_store.fetch_one(
            f"SELECT COUNT(*) AS count FROM {table_names.relations}"
        )
        evidence_count = await sql_store.fetch_one(
            f"SELECT COUNT(*) AS count FROM {table_names.evidence}"
        )
        assert entity_count == {"count": 2}
        assert relation_count == {"count": 1}
        assert evidence_count == {"count": 3}

        assert await vector_store.count(vector_collections.entities) == 2
        assert await vector_store.count(vector_collections.relations) == 1

        entity_rows = await sql_store.fetch_all(
            f"SELECT entity_id, entity_name, entity_type FROM {table_names.entities} "
            "ORDER BY entity_id"
        )
        assert {row["entity_name"] for row in entity_rows} == {"Heta", "KnowledgeRecipe"}

        query_vector = _first_vector_from_result(
            await embedding.embed(
                EmbeddingRequest(texts=["Heta framework recipe builds knowledge bases"])
            )
        )
        entity_hits = await vector_store.search(
            vector_collections.entities,
            VectorQuery(vector=query_vector, top_k=2),
        )
        assert entity_hits
        assert {hit.metadata["fact_type"] for hit in entity_hits} == {"entity"}
    finally:
        await _drop_table(sql_store, table_names.evidence)
        await _drop_table(sql_store, table_names.relations)
        await _drop_table(sql_store, table_names.entities)
        await vector_store.drop_collection(vector_collections.entities)
        await vector_store.drop_collection(vector_collections.relations)
        await embedding.aclose()
        await vector_store.aclose()
        await sql_store.aclose()
        await object_store.aclose()


async def _put_graph_inputs(
    object_store: LocalObjectStore,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    chunk = ParsedChunk(
        chunk_id="chunk_heta_recipe",
        document_id="doc_heta_recipe",
        source=ParsedSource(
            key="raw/heta_recipe.txt",
            name="heta_recipe.txt",
            file_type="txt",
            content_sha256="a" * 64,
        ),
        page_index=0,
        chunk_index=0,
        text=(
            "Heta uses KnowledgeRecipe to compose models, stores, parsers, and steps "
            "for building a knowledge base."
        ),
        token_start=0,
        token_end=32,
    )
    heta = ExtractedEntity(
        entity_id="entity_heta",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        name="Heta",
        type="framework",
        subtype="knowledge_base_framework",
        description="Heta is a framework for building knowledge bases.",
        attributes={"domain": "knowledge engineering"},
        source_chunk_ids=(chunk.chunk_id,),
    )
    recipe = ExtractedEntity(
        entity_id="entity_knowledge_recipe",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        name="KnowledgeRecipe",
        type="concept",
        subtype="construction_plan",
        description="KnowledgeRecipe describes components and steps for building a knowledge base.",
        attributes={"role": "build plan"},
        source_chunk_ids=(chunk.chunk_id,),
    )
    relation = ExtractedRelation(
        relation_id="relation_heta_uses_recipe",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_entity_id=heta.entity_id,
        target_entity_id=recipe.entity_id,
        source_entity_name=heta.name,
        target_entity_name=recipe.name,
        type="architecture",
        name="uses",
        description="Heta uses KnowledgeRecipe as the construction plan for a knowledge base.",
        attributes={"scope": "framework"},
        source_chunk_ids=(chunk.chunk_id,),
    )

    chunk_key = "chunks/chunk_heta_recipe.json"
    entity_keys = (
        "deduplicated_entities/entity_heta.json",
        "deduplicated_entities/entity_knowledge_recipe.json",
    )
    relation_keys = ("deduplicated_relations/relation_heta_uses_recipe.json",)
    await object_store.put(chunk_key, chunk.to_json_bytes())
    await object_store.put(entity_keys[0], heta.to_json_bytes())
    await object_store.put(entity_keys[1], recipe.to_json_bytes())
    await object_store.put(relation_keys[0], relation.to_json_bytes())
    return entity_keys, relation_keys, (chunk_key,)


def _embedding_model(config: dict[str, Any]) -> EmbeddingModel:
    if os.getenv("HETA_LIVE_GRAPH_FAKE_EMBEDDING", "1") == "1":
        return _FakeEmbeddingModel()
    raw = config["hetadb"]["embedding_api"]
    return EmbeddingModel(
        model_name=_litellm_model_name(os.getenv("HETA_LIVE_EMBEDDING_MODEL_NAME", raw["model"])),
        api_key=os.getenv("HETA_LIVE_EMBEDDING_API_KEY", raw["api_key"]),
        api_base=os.getenv("HETA_LIVE_EMBEDDING_API_BASE", raw["base_url"]),
        request_timeout=float(raw.get("timeout", 90)),
        max_retries=1,
        max_concurrent_requests=2,
    )


def _litellm_model_name(model_name: str) -> str:
    known_prefixes = (
        "openai/",
        "azure/",
        "dashscope/",
        "gemini/",
        "anthropic/",
        "cohere/",
        "voyage/",
    )
    if model_name.startswith(known_prefixes):
        return model_name
    return f"openai/{model_name}"


def _postgres_url(config: dict[str, Any]) -> str:
    if url := os.getenv("HETA_LIVE_POSTGRES_URL") or os.getenv("HETA_LIVE_SQL_URL"):
        return url
    pg = config["persistence"]["postgresql"]
    user = quote_plus(str(pg["user"]))
    password = quote_plus(str(pg["password"]))
    host = pg["host"]
    port = pg["port"]
    database = pg["database"]
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def _milvus_uri(config: dict[str, Any]) -> str:
    if uri := os.getenv("HETA_LIVE_MILVUS_URI"):
        return uri
    milvus = config["persistence"]["milvus"]
    return str(milvus.get("url") or f"http://{milvus['host']}:{milvus['port']}")


def _milvus_db_name(config: dict[str, Any]) -> str | None:
    if "HETA_LIVE_MILVUS_DB" in os.environ:
        value = os.environ["HETA_LIVE_MILVUS_DB"].strip()
        return value or None
    return config.get("hetadb", {}).get("milvus", {}).get("db_name")


def _load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if not config_path.exists():
        pytest.skip("config.yaml is required for live graph store smoke")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "persistence" not in data or "hetadb" not in data:
        pytest.skip("config.yaml must include persistence and hetadb sections")
    return data


async def _drop_table(sql_store: SQLStore, table: str) -> None:
    await sql_store.execute(f"DROP TABLE IF EXISTS {table}")


def _format_step_errors(step_records: tuple[Any, ...]) -> str:
    return "\n".join(
        f"{record.step_name}: {record.status}: {record.error}"
        for record in step_records
        if record.status != "succeeded"
    )


def _first_vector_from_result(result: Any) -> list[float]:
    return [float(value) for value in result.vectors[0]]


class _FakeEmbeddingModel:
    @property
    def model_name(self) -> str:
        return "fake-live-graph-embedding"

    async def embed(self, request: EmbeddingRequest) -> Any:
        vectors = []
        for text in request.texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "heta" in lower else 0.0,
                    1.0 if "recipe" in lower else 0.0,
                    1.0 if "knowledge" in lower else 0.0,
                ]
            )
        return type(
            "EmbeddingResult",
            (),
            {"vectors": vectors, "model_name": self.model_name},
        )()

    async def embed_many(self, requests: Any) -> list[Any]:
        return [await self.embed(request) for request in requests]

    async def aclose(self) -> None:
        return None
