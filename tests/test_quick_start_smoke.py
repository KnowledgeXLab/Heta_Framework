import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (  # noqa: E402
    EmbeddingRequest,
    EmbeddingResult,
    ModelChunk,
    ModelRequest,
    ModelResult,
)
from heta_framework.common.stores import (  # noqa: E402
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
)
from heta_framework.kb import (  # noqa: E402
    DocumentParserRegistry,
    EmbedChunks,
    HetaGraphProcedure,
    IndexVectors,
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    ParseDocuments,
    SplitDocuments,
    TextParser,
)


class QuickStartLanguageModel:
    model_name = "test/quickstart-language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        step = (request.trace_context or {}).get("step")
        if step == "extract_entities":
            return ModelResult(
                text="",
                parsed={
                    "entities": [
                        {
                            "name": "Heta",
                            "type": "Framework",
                            "subtype": "Knowledge base framework",
                            "description": "Heta composes components to build knowledge bases.",
                            "attributes": {"domain": "knowledge engineering"},
                        },
                        {
                            "name": "KnowledgeBase",
                            "type": "Concept",
                            "subtype": "Runtime object",
                            "description": "KnowledgeBase is created from a KnowledgeRecipe.",
                            "attributes": {"created_by": "KnowledgeRecipe"},
                        },
                    ]
                },
                model_name=self.model_name,
            )
        if step == "extract_relations":
            return ModelResult(
                text="",
                parsed={
                    "relations": [
                        {
                            "source": "Heta",
                            "target": "KnowledgeBase",
                            "type": "build_flow",
                            "name": "creates",
                            "description": "Heta uses recipes and steps to create a KnowledgeBase.",
                            "attributes": {"evidence": "quickstart text"},
                        }
                    ]
                },
                model_name=self.model_name,
            )
        raise AssertionError(f"unexpected language model step: {step}")

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        yield ModelChunk(text_delta="", model_name=self.model_name)

    async def aclose(self) -> None:
        return None


class QuickStartEmbeddingModel:
    model_name = "test/quickstart-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[_vector_for_text(text) for text in request.texts],
            model_name=self.model_name,
            trace_context=request.trace_context,
        )

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]

    async def aclose(self) -> None:
        return None


def test_quick_start_recipe_builds_first_knowledge_base(tmp_path: Path) -> None:
    asyncio.run(_run_quick_start(tmp_path))


async def _run_quick_start(tmp_path: Path) -> None:
    object_store = LocalObjectStore(tmp_path / "objects")
    vector_store = InMemoryVectorStore()
    sql_store = SQLStore(f"sqlite:///{tmp_path / 'knowledge.db'}")

    await object_store.put(
        "raw/heta.txt",
        (
            "Heta is a framework for building knowledge bases. "
            "It uses recipes to compose parsers, language models, embedding models, "
            "object stores, vector stores, SQL stores, and graph-building steps. "
            "The HetaGraphProcedure extracts entities and relations from chunks."
        ).encode("utf-8"),
    )

    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
        models=KnowledgeModels(
            language=QuickStartLanguageModel(),
            embedding=QuickStartEmbeddingModel(),
        ),
        stores=KnowledgeStores(
            objects=object_store,
            vector=vector_store,
            sql=sql_store,
        ),
        steps=(
            ParseDocuments(),
            SplitDocuments(),
            EmbedChunks(),
            IndexVectors(),
            *HetaGraphProcedure.build().steps(),
        ),
    )
    recipe.require_valid()

    kb = await KnowledgeBase.create(
        recipe=recipe,
        name="quickstart",
        description="A first Heta knowledge base.",
    )

    assert kb.run_record.status == "succeeded"
    assert kb.run_record.capabilities.queries == frozenset({"vector_search", "heta_graph_search"})

    entity_count = await sql_store.fetch_one("SELECT COUNT(*) AS count FROM entities")
    relation_count = await sql_store.fetch_one("SELECT COUNT(*) AS count FROM relations")
    assert entity_count == {"count": 2}
    assert relation_count == {"count": 1}

    assert kb.available_queries == frozenset(
        {
            "vector_search",
            "heta_graph_search",
            "hybrid_search",
        }
    )
    response = await kb.query(
        "How does Heta build a knowledge base?",
        mode="vector_search",
        top_k=1,
    )
    assert len(response.results) == 1
    assert "Heta is a framework" in response.results[0].text
    graph_response = await kb.query("HetaGraphProcedure", mode="heta_graph_search", top_k=3)
    assert {result.kind for result in graph_response.results} >= {"entity", "relation"}

    assert await object_store.exists("raw/heta.txt")
    assert len(await object_store.list("parsed")) == 1
    assert len(await object_store.list("chunks")) == 1
    assert len(await object_store.list("embeddings")) == 1
    assert len(await object_store.list("deduplicated_entities")) == 2
    assert len(await object_store.list("deduplicated_relations")) == 1

    await sql_store.aclose()
    await vector_store.aclose()
    await object_store.aclose()


def _vector_for_text(text: str) -> list[float]:
    buckets = [0.0] * 8
    for index, char in enumerate(text):
        buckets[index % len(buckets)] += (ord(char) % 31) / 31.0
    total = sum(abs(value) for value in buckets) or 1.0
    return [value / total for value in buckets]
