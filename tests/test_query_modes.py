import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (  # noqa: E402
    ModelRequest,
    ModelResult,
    RerankItem,
    RerankRequest,
    RerankResult,
)
from heta_framework.kb import (  # noqa: E402
    KnowledgeBaseBuilder,
    KnowledgeModels,
    KnowledgeRecipe,
    QueryContext,
    QueryEngineRegistry,
    QueryRequest,
    QueryResponse,
    QueryResult,
    SearchAsset,
    SearchAssetCollection,
    SearchAssetRef,
)
from heta_framework.kb.search.engines import (  # noqa: E402
    HybridSearchEngine,
    MultiHopSearchEngine,
    RerankSearchEngine,
    RewriteSearchEngine,
)


class FakeVectorEngine:
    mode = "vector_search"
    required_assets = frozenset({SearchAssetRef(kind="chunk_vector_index")})

    async def query(self, request, context):
        return QueryResponse(
            mode=self.mode,
            results=(
                QueryResult(
                    id="chunk_a",
                    text=f"vector {request.text}",
                    score=0.9,
                    source={"object_key": "raw/vector.pdf", "chunk_ids": ("chunk_a",)},
                ),
                QueryResult(
                    id="chunk_b",
                    text="shared chunk",
                    score=0.8,
                    source={"object_key": "raw/shared.pdf", "chunk_ids": ("chunk_b",)},
                ),
            ),
        )


class FakeFullTextEngine:
    mode = "full_text_search"
    required_assets = frozenset({SearchAssetRef(kind="chunk_full_text_index")})

    async def query(self, request, context):
        return QueryResponse(
            mode=self.mode,
            results=(
                QueryResult(
                    id="chunk_b",
                    text="shared chunk",
                    score=0.7,
                    source={"object_key": "raw/shared.pdf", "chunk_ids": ("chunk_b",)},
                ),
                QueryResult(
                    id="chunk_c",
                    text=f"full text {request.text}",
                    score=0.6,
                    source={"object_key": "raw/full-text.pdf", "chunk_ids": ("chunk_c",)},
                ),
            ),
        )


class FakeHybridEngine:
    mode = "hybrid_search"
    required_assets = frozenset(
        {
            SearchAssetRef(kind="chunk_vector_index"),
            SearchAssetRef(kind="graph_tables"),
            SearchAssetRef(kind="graph_vector_index"),
        }
    )

    async def query(self, request, context):
        return QueryResponse(
            mode=self.mode,
            results=(
                QueryResult(
                    id="chunk_a",
                    text=f"hybrid {request.text}",
                    score=0.9,
                    source={"object_key": "raw/hybrid.pdf", "chunk_ids": ("chunk_a",)},
                ),
                QueryResult(
                    id="chunk_b",
                    text="shared chunk",
                    score=0.8,
                    source={"object_key": "raw/shared.pdf", "chunk_ids": ("chunk_b",)},
                ),
            ),
        )


class FakeGraphEngine:
    mode = "heta_graph_search"
    required_assets = frozenset(
        {
            SearchAssetRef(kind="graph_tables"),
            SearchAssetRef(kind="graph_vector_index"),
        }
    )

    async def query(self, request, context):
        return QueryResponse(
            mode=self.mode,
            results=(
                QueryResult(
                    id="entity_ocean",
                    text="graph ocean entity",
                    score=0.3,
                    kind="entity",
                    source={"object_key": "raw/graph.pdf", "chunk_ids": ("chunk_g1",)},
                ),
                QueryResult(
                    id="relation_bio",
                    text="graph biodiversity relation",
                    score=0.2,
                    kind="relation",
                    source={"object_key": "raw/graph.pdf", "chunk_ids": ("chunk_g2",)},
                ),
            ),
        )


class FakeBaseEngine:
    mode = "heta_rerank_search"
    required_assets = frozenset({SearchAssetRef(kind="chunk_vector_index")})

    async def query(self, request, context):
        return QueryResponse(
            mode=self.mode,
            results=(
                QueryResult(
                    id=f"{request.text}:1",
                    text=f"evidence for {request.text}",
                    score=1.0,
                    source={"object_key": "raw/base.pdf", "chunk_ids": (f"{request.text}:1",)},
                ),
                QueryResult(
                    id=f"{request.text}:2",
                    text=f"more evidence for {request.text}",
                    score=0.5,
                    source={"object_key": "raw/base.pdf", "chunk_ids": (f"{request.text}:2",)},
                ),
            ),
        )


class FakeReranker:
    model_name = "test/reranker"

    async def rerank(self, request: RerankRequest) -> RerankResult:
        rankings = [
            RerankItem(index=index, score=1.0 - index / 10)
            for index, document in enumerate(request.documents)
            if "full text" in document
        ]
        rankings.extend(
            RerankItem(index=index, score=0.1 - index / 100)
            for index, document in enumerate(request.documents)
            if "full text" not in document
        )
        return RerankResult(rankings=rankings, model_name=self.model_name)

    async def rerank_many(self, requests):
        return [await self.rerank(request) for request in requests]


class FakeLanguageModel:
    model_name = "test/language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        prompt = request.prompt or ""
        if "Generate alternative search queries" in prompt:
            return ModelResult(
                text='{"queries":["expanded alpha","expanded beta","expanded gamma"]}',
                parsed={"queries": ["expanded alpha", "expanded beta", "expanded gamma"]},
                model_name=self.model_name,
            )
        if "Decide whether the observation contains information useful" in prompt:
            return ModelResult(
                text='{"usefulness":true,"information":"useful evidence"}',
                parsed={"usefulness": True, "information": "useful evidence"},
                model_name=self.model_name,
            )
        if "Judge whether the accumulated information is sufficient" in prompt:
            return ModelResult(
                text='{"judge":true,"answer":"final multi-hop answer"}',
                parsed={"judge": True, "answer": "final multi-hop answer"},
                model_name=self.model_name,
            )
        return ModelResult(text="direct answer", model_name=self.model_name)

    async def invoke_many(self, requests):
        return [await self.invoke(request) for request in requests]

    def stream(self, request):
        async def chunks():
            if False:
                yield None

        return chunks()


def test_rerank_search_fuses_candidates_and_uses_reranker():
    async def run():
        context = await _context(
            recipe=KnowledgeRecipe(models=KnowledgeModels(reranker=FakeReranker())),
            engines=QueryEngineRegistry(
                [FakeHybridEngine(), FakeFullTextEngine(), RerankSearchEngine()]
            ),
            assets=_heta_rerank_assets(),
        )

        response = await context.query(
            "heta_rerank_search",
            QueryRequest(text="heta", mode="heta_rerank_search", top_k=2),
        )

        assert response.mode == "heta_rerank_search"
        assert [result.id for result in response.results] == ["chunk_c", "chunk_b"]
        assert response.metadata["used_reranker"] is True
        assert response.results[0].metadata["reranker_model"] == "test/reranker"
        assert response.results[0].source["object_key"] == "raw/full-text.pdf"
        assert response.citations[0].source == response.results[0].source

    asyncio.run(run())


def test_hybrid_search_fuses_vector_and_graph_with_weighted_rrf():
    async def run():
        context = await _context(
            recipe=KnowledgeRecipe(),
            engines=QueryEngineRegistry(
                [FakeVectorEngine(), FakeGraphEngine(), HybridSearchEngine()]
            ),
            assets=_hybrid_assets(),
        )

        response = await context.query(
            "hybrid_search",
            QueryRequest(
                text="marine biodiversity",
                mode="hybrid_search",
                top_k=3,
                options={"hybrid_weights": {"heta_graph_search": 2.0}},
            ),
        )

        assert response.mode == "hybrid_search"
        assert response.metadata["fusion"] == "weighted_rrf"
        assert response.metadata["weights"]["heta_graph_search"] == 2.0
        assert [result.id for result in response.results] == [
            "entity_ocean",
            "relation_bio",
            "chunk_a",
        ]
        assert response.results[0].metadata["retrieval_modes"] == ("heta_graph_search",)
        assert response.results[0].source["object_key"] == "raw/graph.pdf"
        assert response.citations[0].result_id == response.results[0].id

    asyncio.run(run())


def test_rerank_search_falls_back_to_rrf_without_reranker():
    async def run():
        context = await _context(
            recipe=KnowledgeRecipe(),
            engines=QueryEngineRegistry(
                [FakeHybridEngine(), FakeFullTextEngine(), RerankSearchEngine()]
            ),
            assets=_heta_rerank_assets(),
        )

        response = await context.query(
            "heta_rerank_search",
            QueryRequest(text="heta", mode="heta_rerank_search", top_k=3),
        )

        assert response.metadata["used_reranker"] is False
        assert [result.id for result in response.results] == ["chunk_b", "chunk_a", "chunk_c"]
        assert response.results[0].metadata["retrieval_modes"] == (
            "hybrid_search",
            "full_text_search",
        )
        assert response.results[0].source["object_key"] == "raw/shared.pdf"
        assert response.citations[0].source == response.results[0].source

    asyncio.run(run())


def test_query_engine_generates_answer_when_requested():
    async def run():
        context = await _context(
            recipe=KnowledgeRecipe(models=KnowledgeModels(language=FakeLanguageModel())),
            engines=QueryEngineRegistry(
                [FakeHybridEngine(), FakeFullTextEngine(), RerankSearchEngine()]
            ),
            assets=_heta_rerank_assets(),
        )

        response = await context.query(
            "heta_rerank_search",
            QueryRequest(
                text="heta",
                mode="heta_rerank_search",
                top_k=2,
                options={"generate_answer": True},
            ),
        )

        assert response.answer == "direct answer"
        assert response.metadata["answer_generation"] == "generated"
        assert response.citations[0].source == response.results[0].source

    asyncio.run(run())


def test_query_engine_does_not_fail_when_answer_model_is_missing():
    async def run():
        context = await _context(
            recipe=KnowledgeRecipe(),
            engines=QueryEngineRegistry(
                [FakeHybridEngine(), FakeFullTextEngine(), RerankSearchEngine()]
            ),
            assets=_heta_rerank_assets(),
        )

        response = await context.query(
            "heta_rerank_search",
            QueryRequest(
                text="heta",
                mode="heta_rerank_search",
                top_k=2,
                options={"generate_answer": True},
            ),
        )

        assert response.answer is None
        assert response.results
        assert response.metadata["answer_generation"] == "missing_language_model"

    asyncio.run(run())


def test_rewrite_search_generates_variants_and_fuses_base_results():
    async def run():
        context = await _context(
            recipe=KnowledgeRecipe(models=KnowledgeModels(language=FakeLanguageModel())),
            engines=QueryEngineRegistry(
                [
                    FakeBaseEngine(),
                    RewriteSearchEngine(
                        base_mode="heta_rerank_search",
                        required_asset_refs=frozenset({SearchAssetRef(kind="chunk_vector_index")})
                    ),
                ]
            ),
            assets=(SearchAsset(kind="chunk_vector_index", name="chunks"),),
        )

        response = await context.query(
            "heta_rewrite_search",
            QueryRequest(text="original", mode="heta_rewrite_search", top_k=3),
        )

        assert response.mode == "heta_rewrite_search"
        assert response.metadata["variations"] == (
            "expanded alpha",
            "expanded beta",
            "expanded gamma",
        )
        assert response.metadata["issues"] == ()
        assert len(response.results) == 3
        assert response.results[0].id == "expanded alpha:1"
        assert response.citations[0].result_id == response.results[0].id
        assert response.citations[0].source == response.results[0].source

    asyncio.run(run())


def test_multihop_search_retrieves_until_answer_is_available():
    async def run():
        context = await _context(
            recipe=KnowledgeRecipe(models=KnowledgeModels(language=FakeLanguageModel())),
            engines=QueryEngineRegistry(
                [
                    FakeBaseEngine(),
                    MultiHopSearchEngine(
                        base_mode="heta_rerank_search",
                        required_asset_refs=frozenset({SearchAssetRef(kind="chunk_vector_index")})
                    ),
                ]
            ),
            assets=(SearchAsset(kind="chunk_vector_index", name="chunks"),),
        )

        response = await context.query(
            "heta_multihop_search",
            QueryRequest(text="complex question", mode="heta_multihop_search", top_k=5),
        )

        assert response.mode == "heta_multihop_search"
        assert response.answer == "final multi-hop answer"
        assert response.metadata["rounds"] == 1
        assert response.metadata["round_reports"] == (
            {
                "round": 1,
                "query": "complex question",
                "result_count": 2,
                "extracted_information": True,
                "answered": True,
                "next_query": None,
            },
        )
        assert [result.id for result in response.results] == [
            "complex question:1",
            "complex question:2",
        ]
        assert response.citations[0].result_id == "complex question:1"
        assert response.citations[0].source == response.results[0].source

    asyncio.run(run())


def test_default_registry_exposes_all_search_modes_when_assets_and_components_exist():
    registry = QueryEngineRegistry.defaults()
    recipe = KnowledgeRecipe(models=KnowledgeModels(language=FakeLanguageModel()))
    assets = SearchAssetCollection(
        (
            SearchAsset(kind="chunk_vector_index", name="chunks"),
            SearchAsset(kind="chunk_text_index", name="chunks"),
            SearchAsset(kind="chunk_full_text_index", name="chunks_full_text"),
            SearchAsset(kind="graph_tables", name="graph"),
            SearchAsset(kind="graph_vector_index", name="graph_vectors"),
        )
    )

    assert registry.available_modes_for(recipe, assets) == frozenset(
        {
            "vector_search",
            "sql_text_search",
            "full_text_search",
            "heta_graph_search",
            "hybrid_search",
            "heta_rerank_search",
            "heta_rewrite_search",
            "heta_multihop_search",
        }
    )


async def _context(*, recipe, engines, assets):
    build_result = await KnowledgeBaseBuilder().build(recipe)
    return QueryContext(
        recipe=recipe,
        run_record=build_result.record,
        assets=SearchAssetCollection(assets),
        engines=engines,
    )


def _chunk_assets():
    return (
        SearchAsset(kind="chunk_vector_index", name="chunks"),
        SearchAsset(kind="chunk_text_index", name="chunks"),
        SearchAsset(kind="chunk_full_text_index", name="chunks_full_text"),
    )


def _heta_rerank_assets():
    return (
        SearchAsset(kind="chunk_vector_index", name="chunks"),
        SearchAsset(kind="chunk_full_text_index", name="chunks_full_text"),
        SearchAsset(kind="graph_tables", name="graph"),
        SearchAsset(kind="graph_vector_index", name="graph_vectors"),
    )


def _hybrid_assets():
    return (
        SearchAsset(kind="chunk_vector_index", name="chunks"),
        SearchAsset(kind="graph_tables", name="graph"),
        SearchAsset(kind="graph_vector_index", name="graph_vectors"),
    )
