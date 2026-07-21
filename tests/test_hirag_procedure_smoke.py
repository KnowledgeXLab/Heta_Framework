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
from heta_framework.common.stores import InMemoryGraphStore, InMemoryVectorStore, LocalObjectStore, SQLStore  # noqa: E402
from heta_framework.kb import (  # noqa: E402
    DocumentParserRegistry,
    HiRAGProcedure,
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    TextParser,
)


class SmokeLanguageModel:
    model_name = "fake-hirag-smoke-language"

    def __init__(self) -> None:
        self.requests = []

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        trace = request.trace_context or {}
        stage = str(trace.get("stage") or "")
        if stage == "hi_entity_extraction":
            return ModelResult(
                text="##".join(
                    [
                        '("entity"<|>"alice"<|>"person"<|>"Alice is a researcher.")',
                        '("entity"<|>"bob"<|>"person"<|>"Bob is a collaborator.")',
                        '("entity"<|>"acme"<|>"organization"<|>"Acme supports the work.")',
                        "<|COMPLETE|>",
                    ]
                ),
                model_name=self.model_name,
            )
        if stage == "hi_entity_extraction:if_loop":
            return ModelResult(text="no", model_name=self.model_name)
        if stage == "hi_relation_extraction":
            return ModelResult(
                text='("relationship"<|>"alice"<|>"bob"<|>"Alice collaborates with Bob."<|>"1.0")<|COMPLETE|>',
                model_name=self.model_name,
            )
        if stage == "hi_relation_extraction:if_loop":
            return ModelResult(text="no", model_name=self.model_name)
        if stage == "summary_clusters":
            return ModelResult(
                text='("entity"<|>"collaboration"<|>"event"<|>"Alice and Bob collaborate.")<|COMPLETE|>',
                model_name=self.model_name,
            )
        if stage == "community_report":
            return ModelResult(
                text=json.dumps(
                    {
                        "title": "Collaboration Community",
                        "summary": "Alice and Bob collaborate.",
                        "findings": [{"summary": "Collaboration", "explanation": "The graph links Alice and Bob."}],
                        "rating": 8,
                    }
                ),
                model_name=self.model_name,
            )
        if stage == "answer_generation":
            return ModelResult(text="Alice collaborates with Bob.", model_name=self.model_name)
        raise AssertionError(f"unexpected language model request: {trace}")

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


class SmokeEmbeddingModel:
    model_name = "fake-hirag-smoke-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        vectors = []
        for index, text in enumerate(request.texts):
            vectors.append([float(index + 1), float(len(text.split()) + 1), 1.0])
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]


def test_hirag_procedure_smoke_returns_real_chunk_sources(tmp_path: Path) -> None:
    asyncio.run(_run_hirag_smoke(tmp_path))


async def _run_hirag_smoke(tmp_path: Path) -> None:
    object_store = LocalObjectStore(tmp_path / "objects")
    graph_store = InMemoryGraphStore()
    vector_store = InMemoryVectorStore()
    sql_store = SQLStore(f"sqlite:///{tmp_path / 'hirag.db'}")

    await object_store.put(
        "raw/alice.txt",
        b"Alice collaborates with Bob on knowledge graph research at Acme.",
    )

    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
        models=KnowledgeModels(language=SmokeLanguageModel(), embedding=SmokeEmbeddingModel()),
        stores=KnowledgeStores(
            objects=object_store,
            graph=graph_store,
            vector=vector_store,
            sql=sql_store,
        ),
        steps=(
            HiRAGProcedure(
                chunk_token_size=128,
                chunk_overlap_token_size=16,
                entity_extract_max_gleaning=0,
                hierarchical_layers=2,
                clustering_backend="deterministic",
            ),
        ),
    )
    recipe.require_valid()
    kb = await KnowledgeBase.create(recipe=recipe, name="hirag-smoke")

    assert kb.run_record.status == "succeeded"
    assert "hi_rag_query" in kb.available_queries
    assert await graph_store.count_nodes() >= 3
    assert await vector_store.count("hi_rag_entities") >= 3

    for mode in (
        "hi_rag_query",
        "hi_rag_nobridge_query",
        "hi_rag_local_query",
        "hi_rag_global_query",
        "hi_rag_bridge_query",
    ):
        response = await kb.query(
            "How does Alice relate to Bob?",
            mode=mode,
            top_k=3,
            options={"generate_answer": False, "top_m": 3},
            trace=True,
        )
        assert response.results
        assert response.results[0].source["chunk_ids"]
        assert "doc_" in response.results[0].source["document_ids"][0]
        assert response.results[0].metadata["source_ids"]
