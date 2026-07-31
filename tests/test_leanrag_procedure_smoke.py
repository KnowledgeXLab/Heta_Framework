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
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    LeanRAGProcedure,
    TextParser,
)


class SmokeLanguageModel:
    model_name = "fake-leanrag-smoke-language"

    def __init__(self) -> None:
        self.aggregate_index = 0
        self.requests = []

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        trace = request.trace_context or {}
        stage = str(trace.get("stage") or "")
        if trace.get("step") == "extract_universal_graph" and stage == "entity_extraction":
            return ModelResult(
                text="",
                parsed={
                    "entities": [
                        {
                            "name": f"entity_{index}",
                            "type": "concept",
                            "subtype": None,
                            "description": f"Entity {index} appears with Alice and Bob.",
                            "attributes": {},
                        }
                        for index in range(10)
                    ]
                },
                model_name=self.model_name,
            )
        if trace.get("step") == "extract_universal_graph" and stage == "relation_extraction":
            return ModelResult(
                text="",
                parsed={
                    "relations": [
                        {
                            "source": "entity_0",
                            "target": "entity_2",
                            "type": "supports",
                            "name": "supports",
                            "description": "Entity 0 supports Entity 2.",
                            "attributes": {"weight": "1.0"},
                        }
                    ]
                },
                model_name=self.model_name,
            )
        if stage == "hi_entity_extraction":
            records = [
                f'("entity"<|>"entity_{index}"<|>"concept"<|>"Entity {index} appears with Alice and Bob.")'
                for index in range(10)
            ]
            return ModelResult(text="##".join(records + ["<|COMPLETE|>"]), model_name=self.model_name)
        if stage == "hi_entity_extraction:if_loop":
            return ModelResult(text="no", model_name=self.model_name)
        if stage == "hi_relation_extraction":
            return ModelResult(
                text='("relationship"<|>"entity_0"<|>"entity_2"<|>"Entity 0 supports Entity 2."<|>"1.0")<|COMPLETE|>',
                model_name=self.model_name,
            )
        if stage == "hi_relation_extraction:if_loop":
            return ModelResult(text="no", model_name=self.model_name)
        if stage == "aggregate_entities":
            name = "ROOT" if self.aggregate_index >= 5 else f"AGG_{self.aggregate_index}"
            self.aggregate_index += 1
            return ModelResult(
                text=json.dumps(
                    {
                        "entity_name": name,
                        "entity_description": f"{name} description",
                        "findings": [{"summary": name, "explanation": f"{name} evidence"}],
                    }
                ),
                model_name=self.model_name,
            )
        if stage == "rag_response":
            return ModelResult(text="LeanRAG smoke answer.", model_name=self.model_name)
        raise AssertionError(f"unexpected language model request: {trace}")

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


class SmokeEmbeddingModel:
    model_name = "fake-leanrag-smoke-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[_vector(text) for text in request.texts],
            model_name=self.model_name,
        )

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]


def _vector(text: str) -> list[float]:
    lowered = text.lower()
    if "entity_0" in lowered or "entity 0" in lowered or "alice" in lowered:
        return [1.0, 0.0, 0.0]
    if "entity_2" in lowered or "entity 2" in lowered:
        return [0.9, 0.1, 0.0]
    if "agg" in lowered or "root" in lowered:
        return [0.0, 1.0, 0.0]
    return [0.2, 0.2, 1.0]


def test_leanrag_procedure_smoke_returns_real_chunk_sources(tmp_path: Path) -> None:
    asyncio.run(_run_leanrag_smoke(tmp_path))


async def _run_leanrag_smoke(tmp_path: Path) -> None:
    object_store = LocalObjectStore(tmp_path / "objects")
    graph_store = InMemoryGraphStore()
    vector_store = InMemoryVectorStore()
    sql_store = SQLStore(f"sqlite:///{tmp_path / 'leanrag.db'}")

    await object_store.put(
        "raw/alice.txt",
        b"Alice and Bob discuss entity 0 through entity 9 in a compact knowledge graph.",
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
            LeanRAGProcedure(
                chunk_token_size=128,
                chunk_overlap_token_size=16,
                entity_extract_max_gleaning=0,
                aggregation_cluster_size=2,
                aggregation_clustering_backend="deterministic",
            ),
        ),
    )
    recipe.require_valid()
    kb = await KnowledgeBase.create(recipe=recipe, name="leanrag-smoke")

    assert kb.run_record.status == "succeeded"
    assert "lean_rag_query" in kb.available_queries
    assert await graph_store.count_nodes() >= 10
    assert await vector_store.count("leanrag_entities") >= 10

    response = await kb.query(
        "How does Alice relate to entity 0?",
        mode="lean_rag_query",
        top_k=3,
        options={"level_mode": 2, "text_unit_k": 2, "generate_answer": False},
        trace=True,
    )

    assert response.results
    assert response.results[0].source["chunk_ids"]
    assert response.results[0].source["hash_codes"]
    assert response.results[0].source["source_keys"] == ("raw/alice.txt",)
    assert "text_units:" in response.results[0].text
    assert response.trace[0].metadata["parent_chains"]
