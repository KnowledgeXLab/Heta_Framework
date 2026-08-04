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
    SQLStore,
)
from heta_framework.kb import (  # noqa: E402
    KnowledgeBaseBuilder,
    KnowledgeModels,
    KnowledgeRecipe,
    KnowledgeStores,
    QueryContext,
    QueryEngineRegistry,
    QueryRequest,
    SearchAssetCollection,
)
from heta_framework.kb.search.engines import LeanRAGQueryEngine  # noqa: E402
from heta_framework.kb.steps import BuildLeanRAGGraph  # noqa: E402


class FakeStepContext:
    def __init__(self, components):
        self.components = components
        self.artifacts = {}

    def get_component(self, key):
        return self.components[key]

    def get_artifact(self, key):
        return self.artifacts[key]

    def set_artifact(self, key, value):
        self.artifacts[key] = value


class FakeObjectStore:
    def __init__(self):
        self.objects = {}

    async def put(self, key, data):
        self.objects[key] = data

    async def get(self, key):
        return self.objects[key]

    async def exists(self, key):
        return key in self.objects

    async def list(self, prefix=""):
        return []

    async def delete(self, key):
        self.objects.pop(key, None)

    async def aclose(self):
        return None


class FakeEmbeddingModel:
    @property
    def model_name(self):
        return "fake-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[_vector(text) for text in request.texts],
            model_name=self.model_name,
        )

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]


class FakeLanguageModel:
    def __init__(self):
        self.requests = []

    @property
    def model_name(self):
        return "fake-language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        return ModelResult(text="leanrag answer", model_name=self.model_name)

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


def _vector(text: str) -> list[float]:
    lowered = text.lower()
    if "alice group" in lowered:
        return [0.0, 1.0, 0.0]
    if "bob" in lowered:
        return [0.8, 0.0, 0.0]
    return [1.0, 0.0, 0.0]


async def _build_context(*, with_language=True):
    graph = InMemoryGraphStore()
    objects = FakeObjectStore()
    sql = SQLStore("sqlite:///:memory:")
    vector = InMemoryVectorStore()
    embedding = FakeEmbeddingModel()
    language = FakeLanguageModel() if with_language else None
    step_context = FakeStepContext(
        {
            "stores.graph": graph,
            "stores.objects": objects,
            "stores.sql": sql,
            "stores.vector": vector,
            "models.embedding": embedding,
        }
    )
    _put_artifacts(step_context)
    build_step = BuildLeanRAGGraph()
    await build_step.run(step_context)
    recipe = KnowledgeRecipe(
        models=KnowledgeModels(embedding=embedding, language=language),
        stores=KnowledgeStores(objects=objects, graph=graph, sql=sql, vector=vector),
    )
    build_result = await KnowledgeBaseBuilder().build(recipe)
    query_context = QueryContext(
        recipe=recipe,
        run_record=build_result.record,
        assets=SearchAssetCollection(build_step.capabilities.search_assets),
        engines=QueryEngineRegistry(
            [LeanRAGQueryEngine(prompts={"rag_response": "ANSWER WITH\n{context_data}"})]
        ),
    )
    return query_context, language


def _put_artifacts(context):
    context.set_artifact(
        "lean_rag_chunks",
        [
            {
                "chunk_id": "chunk_1",
                "hash_code": "hash_1",
                "document_id": "doc_1",
                "text": "Alice collaborates with Bob.",
                "content": "Alice collaborates with Bob.",
                "source_key": "raw/alice.txt",
                "source_name": "alice.txt",
                "source_file_type": "txt",
                "chunk_order_index": 0,
                "token_count": 5,
            }
        ],
    )
    context.set_artifact(
        "lean_rag_base_relations",
        [
            {
                "src_tgt": "Alice",
                "tgt_src": "Bob",
                "description": "Alice is related to Bob.",
                "weight": 1,
                "source_id": "hash_1",
                "source_ids": ["hash_1"],
                "level": 0,
                "is_generated": False,
            }
        ],
    )
    context.set_artifact(
        "lean_rag_all_entities_layers",
        [
            [
                {
                    "entity_name": "Alice",
                    "entity_type": "person",
                    "description": "Alice appears.",
                    "source_id": "hash_1",
                    "source_ids": ["hash_1"],
                    "degree": 1,
                    "parent": "Alice Group",
                    "level": 0,
                    "is_aggregate": False,
                },
                {
                    "entity_name": "Bob",
                    "entity_type": "person",
                    "description": "Bob appears.",
                    "source_id": "hash_1",
                    "source_ids": ["hash_1"],
                    "degree": 1,
                    "parent": "Alice Group",
                    "level": 0,
                    "is_aggregate": False,
                },
            ],
            [
                {
                    "entity_name": "Alice Group",
                    "entity_type": "aggregate entity",
                    "description": "Alice and Bob group.",
                    "source_id": "hash_1",
                    "source_ids": ["hash_1"],
                    "degree": 1,
                    "parent": "root",
                    "level": 1,
                    "is_aggregate": True,
                    "children": ["Alice", "Bob"],
                }
            ],
        ],
    )
    context.set_artifact("lean_rag_generated_relations", [])
    context.set_artifact(
        "lean_rag_communities",
        [
            {
                "entity_name": "Alice Group",
                "entity_description": "Alice and Bob group.",
                "findings": [{"summary": "Link", "explanation": "They are linked."}],
                "level": 0,
                "children": ["Alice", "Bob"],
                "source_id": "hash_1",
                "source_ids": ["hash_1"],
            }
        ],
    )
    context.set_artifact("lean_rag_parent_edges", [])


def test_leanrag_query_context_sections_and_answer():
    async def run():
        context, language = await _build_context()
        response = await context.query(
            "lean_rag_query",
            QueryRequest(
                text="Alice",
                mode="lean_rag_query",
                top_k=2,
                options={"level_mode": 0, "text_unit_k": 1},
                trace=True,
            ),
        )
        return response, language

    response, language = asyncio.run(run())

    assert response.answer == "leanrag answer"
    assert "entity_information:" in response.results[0].text
    assert "aggregation_entity_information:" in response.results[0].text
    assert "reasoning_path_information:" in response.results[0].text
    assert "text_units:" in response.results[0].text
    assert "Alice is related to Bob." in response.results[0].text
    assert response.results[0].source["hash_codes"] == ("hash_1",)
    assert response.results[0].source["entity_names"] == ("Alice", "Bob")
    assert response.trace[0].metadata["level_mode_filter"] == {"level": 0}
    assert "ANSWER WITH" in language.requests[0].prompt


def test_leanrag_query_only_need_context_and_generate_answer_false():
    async def run():
        context, _ = await _build_context(with_language=False)
        context_only = await context.query(
            "lean_rag_query",
            QueryRequest(
                text="Alice",
                mode="lean_rag_query",
                options={"only_need_context": True},
            ),
        )
        disabled = await context.query(
            "lean_rag_query",
            QueryRequest(
                text="Alice",
                mode="lean_rag_query",
                options={"generate_answer": False},
            ),
        )
        return context_only, disabled

    context_only, disabled = asyncio.run(run())

    assert context_only.answer is None
    assert context_only.metadata["answer_generation"] == "context_only"
    assert disabled.answer is None
    assert disabled.metadata["answer_generation"] == "disabled"


def test_leanrag_query_level_mode_aggregate_and_missing_parent_fallback():
    async def run():
        context, _ = await _build_context()
        response = await context.query(
            "lean_rag_query",
            QueryRequest(
                text="Alice Group",
                mode="lean_rag_query",
                top_k=5,
                options={"level_mode": 1, "generate_answer": False},
                trace=True,
            ),
        )
        return response

    response = asyncio.run(run())

    assert response.results[0].source["aggregate_entity_names"] == ("Alice Group",)
    assert response.trace[0].metadata["level_mode_filter"] == {"is_aggregate": True}
    assert response.results[0].metadata["parent_chains"]["Alice Group"] == ["Alice Group", "root"]


def test_leanrag_registry_and_build_capability():
    registry = QueryEngineRegistry.defaults()
    step = BuildLeanRAGGraph()

    assert "lean_rag_query" in registry.modes
    assert "lean_rag_query" in step.capabilities.queries
