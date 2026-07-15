import asyncio
import json
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (  # noqa: E402
    EmbeddingRequest,
    EmbeddingResult,
    ModelChunk,
    ModelRequest,
    ModelResult,
)
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore, SQLStore  # noqa: E402
from heta_framework.kb import KnowledgeModels, KnowledgeRecipe, KnowledgeStores  # noqa: E402
from heta_framework.kb.search import QueryContext, QueryEngineRegistry, QueryRequest, SearchAssetCollection  # noqa: E402
from heta_framework.kb.search.engines import (  # noqa: E402
    LightRAGGlobalQueryEngine,
    LightRAGHybridQueryEngine,
    LightRAGLocalQueryEngine,
    LightRAGMixQueryEngine,
)
from heta_framework.kb.steps import BuildLightRAGGraph  # noqa: E402
from test_build_lightrag_graph_step import _put_inputs  # noqa: E402


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
        vectors = [[float(len(text)), float(text.count("Alice")), 1.0] for text in request.texts]
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


class FakeLanguageModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    @property
    def model_name(self):
        return "fake-language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, dict):
            return ModelResult(
                text=json.dumps(response, ensure_ascii=False),
                parsed=response,
                model_name=self.model_name,
            )
        return ModelResult(text=str(response), model_name=self.model_name)

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


async def _built_query_context(tmp_path, language_model):
    object_store = LocalObjectStore(tmp_path)
    sql_store = SQLStore("sqlite:///:memory:")
    vector_store = InMemoryVectorStore()
    embedding_model = FakeEmbeddingModel()
    build_context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.sql": sql_store,
            "stores.vector": vector_store,
            "models.embedding": embedding_model,
        }
    )
    await _put_inputs(object_store, build_context)
    step = BuildLightRAGGraph()
    await step.run(build_context)
    recipe = KnowledgeRecipe(
        models=KnowledgeModels(language=language_model, embedding=embedding_model),
        stores=KnowledgeStores(objects=object_store, sql=sql_store, vector=vector_store),
        steps=(step,),
    )
    query_context = QueryContext(
        recipe=recipe,
        run_record=None,
        assets=SearchAssetCollection(step.capabilities.search_assets),
        engines=QueryEngineRegistry.defaults(),
    )
    return query_context, sql_store


def test_lightrag_local_query_returns_context_and_answer(tmp_path):
    model = FakeLanguageModel(
        [
            {"high_level_keywords": ["collaboration"], "low_level_keywords": ["Alice"]},
            "Alice collaborates with Bob.",
        ]
    )

    async def run():
        context, sql_store = await _built_query_context(tmp_path, model)
        try:
            return await LightRAGLocalQueryEngine().query(
                QueryRequest("What does Alice do?", top_k=3),
                context,
            )
        finally:
            await sql_store.aclose()

    response = asyncio.run(run())

    assert response.mode == "light_rag_local_query"
    assert response.answer == "Alice collaborates with Bob."
    assert response.metadata["keywords"]["low_level"] == ["Alice"]
    assert response.metadata["entity_count"] == 1
    assert response.metadata["relation_count"] == 1
    assert response.metadata["chunk_count"] == 1
    assert "Knowledge Graph Data (Entity)" in response.metadata["context"]
    assert any(result.kind == "chunk" for result in response.results)


def test_lightrag_global_query_uses_relationship_vector(tmp_path):
    model = FakeLanguageModel(
        [
            {"high_level_keywords": ["collaboration"], "low_level_keywords": []},
            "The relationship is collaboration.",
        ]
    )

    async def run():
        context, sql_store = await _built_query_context(tmp_path, model)
        try:
            return await LightRAGGlobalQueryEngine().query(
                QueryRequest("What relationship exists?", top_k=3),
                context,
            )
        finally:
            await sql_store.aclose()

    response = asyncio.run(run())

    assert response.mode == "light_rag_global_query"
    assert response.metadata["relation_count"] == 1
    raw = response.metadata["raw_data"]
    assert raw["data"]["relationships"][0]["keywords"] == "collaboration"
    assert response.metadata["relationship_collection"] == "light_rag_relationships"


def test_lightrag_hybrid_query_deduplicates_context(tmp_path):
    model = FakeLanguageModel(
        [
            {"high_level_keywords": ["collaboration"], "low_level_keywords": ["Alice"]},
            "Hybrid answer.",
        ]
    )

    async def run():
        context, sql_store = await _built_query_context(tmp_path, model)
        try:
            return await LightRAGHybridQueryEngine().query(
                QueryRequest("Summarize Alice collaboration", top_k=3),
                context,
            )
        finally:
            await sql_store.aclose()

    response = asyncio.run(run())

    assert response.mode == "light_rag_hybrid_query"
    assert response.answer == "Hybrid answer."
    assert response.metadata["entity_count"] == 1
    assert response.metadata["relation_count"] == 1
    assert response.metadata["chunk_count"] == 1
    assert len([result for result in response.results if result.kind == "chunk"]) == 1


def test_lightrag_mix_query_includes_direct_chunk_vector_search(tmp_path):
    model = FakeLanguageModel(
        [
            {"high_level_keywords": ["collaboration"], "low_level_keywords": ["Alice"]},
            "Mix answer.",
        ]
    )

    async def run():
        context, sql_store = await _built_query_context(tmp_path, model)
        try:
            return await LightRAGMixQueryEngine().query(
                QueryRequest("Summarize Alice collaboration", top_k=3),
                context,
            )
        finally:
            await sql_store.aclose()

    response = asyncio.run(run())

    assert response.mode == "light_rag_mix_query"
    assert response.answer == "Mix answer."
    assert response.metadata["entity_count"] == 1
    assert response.metadata["relation_count"] == 1
    assert response.metadata["chunk_count"] == 1
    assert response.metadata["vector_chunk_count"] == 1
    assert response.metadata["fact_chunk_count"] == 1
    assert response.metadata["chunk_collection"] == "light_rag_chunks"


def test_lightrag_query_modes_registered_by_default():
    modes = QueryEngineRegistry.defaults().modes

    assert "light_rag_local_query" in modes
    assert "light_rag_global_query" in modes
    assert "light_rag_hybrid_query" in modes
    assert "light_rag_mix_query" in modes
