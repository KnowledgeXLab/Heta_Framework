import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb import (  # noqa: E402
    KnowledgeBaseBuilder,
    KnowledgeRecipe,
    QueryContext,
    QueryEngineRegistry,
    QueryRequest,
    QueryResponse,
    QueryResult,
    SearchAsset,
    SearchAssetCollection,
    SearchAssetRef,
    StepCleanupPlan,
    StepCapabilities,
    StepRequirements,
)
from heta_framework.common.models import EmbeddingRequest, EmbeddingResult  # noqa: E402
from heta_framework.common.stores import (  # noqa: E402
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
    VectorCollectionConfig,
    VectorRecord,
)
from heta_framework.kb import KnowledgeBase, KnowledgeModels, KnowledgeStores  # noqa: E402


class FakeSearchStep:
    name = "index_fake"

    @property
    def requirements(self):
        return StepRequirements()

    @property
    def capabilities(self):
        return StepCapabilities(
            artifacts=frozenset({"index_fake_result"}),
            queries=frozenset({"fake_search"}),
            search_assets=(
                SearchAsset(
                    kind="chunk_vector_index",
                    name="chunks",
                    store="vector",
                    metadata={"collection": "chunks"},
                ),
            ),
        )

    async def run(self, context):
        context.set_artifact("index_fake_result", {"ok": True})

    def cleanup_plan(self, artifacts):
        return StepCleanupPlan()


class FakeKeywordStep:
    name = "persist_fake"

    @property
    def requirements(self):
        return StepRequirements()

    @property
    def capabilities(self):
        return StepCapabilities(
            artifacts=frozenset({"persist_fake_result"}),
            queries=frozenset({"keyword_search"}),
            search_assets=(
                SearchAsset(
                    kind="chunk_text_index",
                    name="chunks",
                    store="sql",
                    metadata={"table": "chunks", "dialect": "generic"},
                ),
            ),
        )

    async def run(self, context):
        context.set_artifact("persist_fake_result", {"ok": True})

    def cleanup_plan(self, artifacts):
        return StepCleanupPlan()


class FakeQueryEngine:
    mode = "fake_search"
    required_assets = frozenset({SearchAssetRef(kind="chunk_vector_index")})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        asset = context.assets.require(SearchAssetRef(kind="chunk_vector_index"))
        return QueryResponse(
            mode=self.mode,
            results=(
                QueryResult(
                    id="hit_1",
                    text=f"{request.text} -> {asset.metadata['collection']}",
                    score=1.0,
                ),
            ),
        )


class DelegatingQueryEngine:
    mode = "delegate_search"
    required_assets = frozenset({SearchAssetRef(kind="chunk_vector_index")})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        return await context.query("fake_search", request)


class RecursiveQueryEngine:
    mode = "recursive_search"
    required_assets = frozenset()

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        return await context.query("recursive_search", request)


def test_search_asset_collection_matches_by_kind_or_name():
    chunks = SearchAsset(kind="chunk_vector_index", name="chunks", store="vector")
    graph = SearchAsset(kind="graph_tables", name="default", store="sql")
    assets = SearchAssetCollection([chunks, graph])

    assert assets.contains(SearchAssetRef(kind="chunk_vector_index"))
    assert assets.require(SearchAssetRef(kind="chunk_vector_index")) is chunks
    assert assets.require(SearchAssetRef(kind="graph_tables", name="default")) is graph
    assert assets.missing([SearchAssetRef(kind="chunk_text_table")]) == (
        SearchAssetRef(kind="chunk_text_table"),
    )


def test_search_asset_collection_rejects_duplicate_asset_keys():
    with pytest.raises(ValueError, match="duplicate"):
        SearchAssetCollection(
            [
                SearchAsset(kind="chunk_vector_index", name="chunks"),
                SearchAsset(kind="chunk_vector_index", name="chunks"),
            ]
        )


def test_query_engine_registry_reports_available_modes():
    registry = QueryEngineRegistry([FakeQueryEngine()])

    assert registry.modes == frozenset({"fake_search"})
    assert registry.available_modes(SearchAssetCollection()) == frozenset()
    assert registry.available_modes(
        SearchAssetCollection([SearchAsset(kind="chunk_vector_index", name="chunks")])
    ) == frozenset({"fake_search"})


def test_query_context_can_delegate_to_another_engine():
    async def run():
        build_result = await KnowledgeBaseBuilder().build(KnowledgeRecipe())
        context = QueryContext(
            recipe=KnowledgeRecipe(),
            run_record=build_result.record,
            assets=SearchAssetCollection(
                [
                    SearchAsset(
                        kind="chunk_vector_index",
                        name="chunks",
                        metadata={"collection": "chunks"},
                    )
                ]
            ),
            engines=QueryEngineRegistry([FakeQueryEngine(), DelegatingQueryEngine()]),
        )

        response = await context.query(
            "delegate_search",
            QueryRequest(text="heta", mode="delegate_search"),
        )

        assert response.mode == "fake_search"
        assert response.results[0].text == "heta -> chunks"

    asyncio.run(run())


def test_query_context_rejects_recursive_engine_calls():
    async def run():
        build_result = await KnowledgeBaseBuilder().build(KnowledgeRecipe())
        context = QueryContext(
            recipe=KnowledgeRecipe(),
            run_record=build_result.record,
            assets=SearchAssetCollection(),
            engines=QueryEngineRegistry([RecursiveQueryEngine()]),
        )

        with pytest.raises(RuntimeError, match="recursive query engine call"):
            await context.query(
                "recursive_search",
                QueryRequest(text="heta", mode="recursive_search"),
            )

    asyncio.run(run())


def test_builder_collects_search_assets_from_successful_steps():
    recipe = KnowledgeRecipe(steps=(FakeSearchStep(),))

    result = asyncio.run(KnowledgeBaseBuilder().build(recipe))

    assert result.capabilities.queries == frozenset({"fake_search"})
    assert result.capabilities.search_assets == (
        SearchAsset(
            kind="chunk_vector_index",
            name="chunks",
            store="vector",
            metadata={"collection": "chunks"},
        ),
    )


class FakeEmbeddingModel:
    model_name = "test/query-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[_vector_for_text(text) for text in request.texts],
            model_name=self.model_name,
        )

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


def test_knowledge_base_query_runs_vector_search():
    async def run():
        vector_store = InMemoryVectorStore()
        await vector_store.create_collection(VectorCollectionConfig(name="chunks", dimension=8))
        await vector_store.upsert(
            "chunks",
            [
                VectorRecord(
                    id="chunk_heta",
                    vector=_vector_for_text("heta builds knowledge bases"),
                    text="Heta builds knowledge bases from recipes.",
                    metadata={"document_id": "doc_1", "source_key": "raw/heta.txt"},
                ),
                VectorRecord(
                    id="chunk_other",
                    vector=_vector_for_text("unrelated"),
                    text="Unrelated note.",
                    metadata={"document_id": "doc_2", "source_key": "raw/other.txt"},
                ),
            ],
        )
        recipe = KnowledgeRecipe(
            models=KnowledgeModels(embedding=FakeEmbeddingModel()),
            stores=KnowledgeStores(vector=vector_store),
            steps=(FakeSearchStep(),),
        )

        kb = await KnowledgeBase.create(recipe=recipe, name="query-test")
        assert kb.available_queries == frozenset({"vector_search"})

        response = await kb.query("heta knowledge base", mode="vector_search", top_k=1)

        assert response.mode == "vector_search"
        assert len(response.results) == 1
        assert response.results[0].id == "chunk_heta"
        assert response.results[0].source["object_key"] == "raw/heta.txt"
        assert response.results[0].source["chunk_ids"] == ("chunk_heta",)
        assert response.citations[0].source == response.results[0].source

    asyncio.run(run())


def test_knowledge_base_load_preserves_query_capabilities(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path / "objects")
        vector_store = InMemoryVectorStore()
        await vector_store.create_collection(VectorCollectionConfig(name="chunks", dimension=8))
        await vector_store.upsert(
            "chunks",
            [
                VectorRecord(
                    id="chunk_heta",
                    vector=_vector_for_text("heta recipe"),
                    text="Heta recipes compose models, stores, parsers, and steps.",
                    metadata={"document_id": "doc_1", "source_key": "raw/heta.txt"},
                )
            ],
        )
        recipe = KnowledgeRecipe(
            models=KnowledgeModels(embedding=FakeEmbeddingModel()),
            stores=KnowledgeStores(objects=object_store, vector=vector_store),
            steps=(FakeSearchStep(),),
        )

        created = await KnowledgeBase.create(recipe=recipe, name="loaded-query-test")
        loaded = await KnowledgeBase.load(recipe=recipe, name="loaded-query-test")

        assert loaded.run_record.run_id == created.run_record.run_id
        assert loaded.available_queries == frozenset({"vector_search"})

        response = await loaded.query("heta recipe", mode="vector_search", top_k=1)

        assert response.results[0].id == "chunk_heta"
        assert response.citations[0].source == response.results[0].source

    asyncio.run(run())


def test_knowledge_base_query_runs_keyword_search():
    async def run():
        sql_store = SQLStore("sqlite:///:memory:")
        await sql_store.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT,
                document_id TEXT,
                content_text TEXT,
                source_id TEXT,
                source_chunk TEXT,
                metadata_json TEXT
            )
            """
        )
        await sql_store.execute(
            """
            INSERT INTO chunks
            (chunk_id, document_id, content_text, source_id, source_chunk, metadata_json)
            VALUES
            (:chunk_id, :document_id, :content_text, :source_id, :source_chunk, :metadata_json)
            """,
            {
                "chunk_id": "chunk_keyword",
                "document_id": "doc_1",
                "content_text": "Heta persists chunks for keyword search.",
                "source_id": "raw/heta.txt",
                "source_chunk": '["chunk_keyword"]',
                "metadata_json": '{"page_index":0,"chunk_index":0}',
            },
        )
        await sql_store.execute(
            """
            INSERT INTO chunks
            (chunk_id, document_id, content_text, source_id, source_chunk, metadata_json)
            VALUES
            (:chunk_id, :document_id, :content_text, :source_id, :source_chunk, :metadata_json)
            """,
            {
                "chunk_id": "chunk_other",
                "document_id": "doc_2",
                "content_text": "Unrelated note.",
                "source_id": "raw/other.txt",
                "source_chunk": '["chunk_other"]',
                "metadata_json": "{}",
            },
        )
        recipe = KnowledgeRecipe(
            stores=KnowledgeStores(sql=sql_store),
            steps=(FakeKeywordStep(),),
        )

        kb = await KnowledgeBase.create(recipe=recipe, name="keyword-test")
        assert "keyword_search" in kb.available_queries

        response = await kb.query("keyword search", mode="keyword_search", top_k=3)

        assert response.mode == "keyword_search"
        assert [result.id for result in response.results] == ["chunk_keyword"]
        assert response.results[0].source["object_key"] == "raw/heta.txt"
        assert response.results[0].source["chunk_ids"] == ("chunk_keyword",)
        assert response.results[0].source["page_index"] == 0
        assert response.citations[0].source == response.results[0].source
        await sql_store.aclose()

    asyncio.run(run())


def _vector_for_text(text: str) -> list[float]:
    buckets = [0.0] * 8
    for index, char in enumerate(text):
        buckets[index % len(buckets)] += (ord(char) % 31) / 31.0
    total = sum(abs(value) for value in buckets) or 1.0
    return [value / total for value in buckets]
