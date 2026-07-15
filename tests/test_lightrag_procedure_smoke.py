import asyncio
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
from heta_framework.common.stores import (  # noqa: E402
    InMemoryGraphStore,
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
)
from heta_framework.kb import (  # noqa: E402
    DocumentParserRegistry,
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    LightRAGProcedure,
    ParseDocuments,
    SplitDocuments,
    TextParser,
)


class SmokeLanguageModel:
    model_name = "fake-lightrag-smoke-language"

    def __init__(self) -> None:
        self.requests = []

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        trace = request.trace_context or {}
        prompt = f"{request.system_prompt or ''}\n{request.prompt or ''}"
        if trace.get("step") == "extract_light_rag_graph":
            if "relation<|#|>" in prompt or "entity<|#|>" in prompt:
                return ModelResult(
                    text=(
                        "entity<|#|>Alice<|#|>Person<|#|>Alice is a researcher.\n"
                        "entity<|#|>Bob<|#|>Person<|#|>Bob is a collaborator.\n"
                        "relation<|#|>Alice<|#|>Bob<|#|>collaboration<|#|>"
                        "Alice collaborates with Bob.\n"
                        "<|COMPLETE|>"
                    ),
                    model_name=self.model_name,
                )
            return ModelResult(
                text=(
                    '{"entities":['
                    '{"name":"Alice","type":"Person","description":"Alice is a researcher."},'
                    '{"name":"Bob","type":"Person","description":"Bob is a collaborator."}'
                    '],"relationships":['
                    '{"source":"Alice","target":"Bob","keywords":"collaboration",'
                    '"description":"Alice collaborates with Bob."}'
                    "]}"
                ),
                model_name=self.model_name,
            )
        if trace.get("stage") == "keyword_extraction":
            return ModelResult(
                text=(
                    '{"high_level_keywords":["collaboration"],'
                    '"low_level_keywords":["Alice","Bob"]}'
                ),
                parsed={
                    "high_level_keywords": ["collaboration"],
                    "low_level_keywords": ["Alice", "Bob"],
                },
                model_name=self.model_name,
            )
        if trace.get("stage") == "answer_generation":
            return ModelResult(
                text="Alice collaborates with Bob.",
                model_name=self.model_name,
            )
        raise AssertionError(f"unexpected language model request: {trace}")

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


class SmokeEmbeddingModel:
    model_name = "fake-lightrag-smoke-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[_vector_for_text(text) for text in request.texts],
            model_name=self.model_name,
            trace_context=request.trace_context,
        )

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]


def test_lightrag_procedure_json_smoke_returns_real_chunk_sources(tmp_path: Path) -> None:
    asyncio.run(_run_lightrag_smoke(tmp_path, extraction_format="json", run_queries=True))


def test_lightrag_procedure_tuple_smoke_builds_graph(tmp_path: Path) -> None:
    asyncio.run(_run_lightrag_smoke(tmp_path, extraction_format="tuple", run_queries=False))


async def _run_lightrag_smoke(
    tmp_path: Path,
    *,
    extraction_format: str,
    run_queries: bool,
) -> None:
    object_store = LocalObjectStore(tmp_path / "objects")
    graph_store = InMemoryGraphStore()
    vector_store = InMemoryVectorStore()
    sql_store = SQLStore(f"sqlite:///{tmp_path / 'lightrag.db'}")

    await object_store.put(
        "raw/alice.txt",
        b"Alice collaborates with Bob on knowledge graph research.",
    )

    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
        models=KnowledgeModels(
            language=SmokeLanguageModel(),
            embedding=SmokeEmbeddingModel(),
        ),
        stores=KnowledgeStores(
            objects=object_store,
            graph=graph_store,
            vector=vector_store,
            sql=sql_store,
        ),
        steps=(
            ParseDocuments(),
            SplitDocuments(),
            LightRAGProcedure(
                extraction_format=extraction_format,  # type: ignore[arg-type]
                entity_extract_max_gleaning=0,
            ),
        ),
    )
    recipe.require_valid()
    kb = await KnowledgeBase.create(recipe=recipe, name=f"lightrag-{extraction_format}")

    assert kb.run_record.status == "succeeded"
    assert "light_rag_local_query" in kb.available_queries
    assert await graph_store.count_nodes() == 2
    assert await graph_store.count_edges() == 1
    entity_count = await sql_store.fetch_one(
        "SELECT COUNT(*) AS count FROM light_rag_entities"
    )
    relation_count = await sql_store.fetch_one(
        "SELECT COUNT(*) AS count FROM light_rag_relations"
    )
    chunk_row = await sql_store.fetch_one(
        "SELECT content, source_name FROM light_rag_chunks LIMIT 1"
    )
    assert entity_count == {"count": 2}
    assert relation_count == {"count": 1}
    assert chunk_row is not None
    assert "Alice collaborates with Bob" in chunk_row["content"]
    assert await vector_store.count("light_rag_entities") == 2
    assert await vector_store.count("light_rag_relationships") == 1
    assert await vector_store.count("light_rag_chunks") == 1

    if run_queries:
        for mode in (
            "light_rag_local_query",
            "light_rag_global_query",
            "light_rag_hybrid_query",
            "light_rag_mix_query",
        ):
            response = await kb.query("How does Alice relate to Bob?", mode=mode, top_k=3)
            assert response.answer == "Alice collaborates with Bob."
            assert response.metadata["chunk_count"] == 1
            chunk_results = [result for result in response.results if result.kind == "chunk"]
            assert len(chunk_results) == 1
            assert "Alice collaborates with Bob" in chunk_results[0].text
            assert chunk_results[0].source["object_name"] == "alice.txt"

    await sql_store.aclose()
    await vector_store.aclose()
    await graph_store.aclose()
    await object_store.aclose()


def _vector_for_text(text: str) -> list[float]:
    buckets = [0.0] * 8
    for index, char in enumerate(text):
        buckets[index % len(buckets)] += (ord(char) % 31) / 31.0
    total = sum(abs(value) for value in buckets) or 1.0
    return [value / total for value in buckets]
