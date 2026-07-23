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
from heta_framework.kb.search.assets import SearchAssetCollection  # noqa: E402
from heta_framework.kb.search.engines import (  # noqa: E402
    HiRAGBridgeQueryEngine,
    HiRAGFullQueryEngine,
    HiRAGGlobalQueryEngine,
    HiRAGLocalQueryEngine,
    HiRAGNobridgeQueryEngine,
)
from heta_framework.kb.search.types import QueryRequest  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    BuildHiRAGGraph,
    BuildHiRAGGraphConfig,
    HiRAGCommunity,
    HiRAGCommunityConfig,
    HiRAGTableNames,
    HiRAGVectorCollections,
)


TEST_PROMPTS = {
    "community_report": "REPORT {input_text}",
    "local_rag_response": "Use context:\n{context_data}\nResponse type: {response_type}",
}


class FakeRecipe:
    def __init__(self, components):
        self.components = components

    def get_component(self, ref):
        return self.components[ref.key]

    def has_component(self, ref):
        return ref.key in self.components


class FakeQueryContext:
    def __init__(self, recipe, assets):
        self.recipe = recipe
        self.assets = assets
        self.run_record = None
        self.engines = None
        self.call_stack = ()


class FakeBuildContext:
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
            upper = text.upper()
            vectors.append(
                [
                    10.0 if "ALICE" in upper else 1.0,
                    10.0 if "CAROL" in upper else 1.0,
                    float(len(text) % 7 + 1),
                ]
            )
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

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
        if request.trace_context and request.trace_context.get("stage") == "answer_generation":
            return ModelResult(text="final answer", model_name=self.model_name)
        return ModelResult(
            text=json.dumps(
                {
                    "title": "Community",
                    "summary": "Community summary.",
                    "findings": [{"summary": "Finding", "explanation": "Evidence."}],
                    "rating": 8,
                }
            ),
            model_name=self.model_name,
        )

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


def _node(entity_id, source_id):
    return {
        "id": entity_id,
        "labels": ["Entity", "PERSON"],
        "properties": {
            "name": entity_id,
            "entity_type": "PERSON",
            "description": f"{entity_id} appears.",
            "source_id": source_id,
            "source_ids": [source_id],
            "layer": 0,
            "cluster_id": None,
            "is_summary": False,
            "parent_entity_ids": [],
        },
    }


def _edge(source, target, chunk_id):
    return {
        "id": f"{source}--RELATED--{target}",
        "source_id": source,
        "target_id": target,
        "type": "RELATED",
        "properties": {
            "description": f"{source} links to {target}.",
            "weight": 1.0,
            "order": 1,
            "source_id": chunk_id,
            "source_ids": [chunk_id],
            "layer": 0,
            "cluster_id": None,
            "is_summary": False,
        },
    }


async def _build_runtime(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    sql_store = SQLStore("sqlite:///:memory:")
    vector_store = InMemoryVectorStore()
    graph_store = InMemoryGraphStore()
    embedding = FakeEmbeddingModel()
    language = FakeLanguageModel()
    build_context = FakeBuildContext(
        {
            "stores.objects": object_store,
            "stores.graph": graph_store,
            "stores.sql": sql_store,
            "stores.vector": vector_store,
            "models.embedding": embedding,
            "models.language": language,
        }
    )
    nodes = [_node("ALICE", "chunk_1"), _node("BOB", "chunk_1"), _node("CAROL", "chunk_2")]
    edges = [_edge("ALICE", "BOB", "chunk_1")]
    node_keys = []
    edge_keys = []
    for node in nodes:
        key = f"hi_rag/graph/nodes/{node['id'].lower()}.json"
        await object_store.put(key, json.dumps(node).encode("utf-8"))
        node_keys.append(key)
    for edge in edges:
        key = f"hi_rag/graph/edges/{edge['id'].lower()}.json"
        await object_store.put(key, json.dumps(edge).encode("utf-8"))
        edge_keys.append(key)
    build_context.set_artifact("hi_rag_graph_node_keys", tuple(node_keys))
    build_context.set_artifact("hi_rag_graph_edge_keys", tuple(edge_keys))
    build_context.set_artifact(
        "hi_rag_chunks",
        [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "content": "Alice and Bob collaborate.",
                "source_key": "raw/doc1.txt",
                "file_path": "doc1.txt",
                "chunk_order_index": 0,
                "tokens": 5,
                "full_doc_id": "doc_1",
            },
            {
                "chunk_id": "chunk_2",
                "document_id": "doc_2",
                "content": "Carol is separate.",
                "source_key": "raw/doc2.txt",
                "file_path": "doc2.txt",
                "chunk_order_index": 0,
                "tokens": 4,
                "full_doc_id": "doc_2",
            },
        ],
    )
    config = BuildHiRAGGraphConfig(
        table_names=HiRAGTableNames(
            entities="q_hi_entities",
            relations="q_hi_relations",
            communities="q_hi_communities",
            chunks="q_hi_chunks",
        ),
        vector_collections=HiRAGVectorCollections(entities="q_hi_entity_vectors"),
        graph_cluster_algorithm="connected_components",
        prompts=TEST_PROMPTS,
    )
    step = BuildHiRAGGraph(config)
    await step.run(build_context)
    await HiRAGCommunity(
        HiRAGCommunityConfig(
            table_names=config.table_names,
            prompts=TEST_PROMPTS,
        )
    ).run(build_context)
    assets = SearchAssetCollection(step.capabilities.search_assets)
    recipe = FakeRecipe(
        {
            "stores.sql": sql_store,
            "stores.vector": vector_store,
            "models.embedding": embedding,
            "models.language": language,
        }
    )
    return FakeQueryContext(recipe, assets), language


def test_hirag_query_modes_context_sections(tmp_path):
    context, _ = asyncio.run(_build_runtime(tmp_path))
    engines = [
        (HiRAGFullQueryEngine(prompts=TEST_PROMPTS), ["-----Backgrounds-----", "-----Reasoning Path-----", "-----Detail Entity Information-----", "-----Source Documents-----"]),
        (HiRAGNobridgeQueryEngine(prompts=TEST_PROMPTS), ["-----Reports-----", "-----Entities-----", "-----Relationships-----", "-----Sources-----"]),
        (HiRAGLocalQueryEngine(prompts=TEST_PROMPTS), ["-----Entities-----", "-----Relations-----", "-----Sources-----"]),
        (HiRAGGlobalQueryEngine(prompts=TEST_PROMPTS), ["-----Backgrounds-----", "-----Source Documents-----"]),
        (HiRAGBridgeQueryEngine(prompts=TEST_PROMPTS), ["-----Reasoning Path-----", "-----Source Documents-----"]),
    ]

    async def run():
        responses = []
        for engine, sections in engines:
            response = await engine.query(
                QueryRequest(
                    text="Alice and Carol",
                    top_k=3,
                    options={"generate_answer": False, "top_m": 3},
                    trace=True,
                ),
                context,
            )
            for section in sections:
                assert section in response.results[0].text
            responses.append(response)
        return responses

    responses = asyncio.run(run())

    assert responses[0].answer is None
    assert responses[0].results[0].metadata["entity_ids"]
    assert responses[0].trace[0].metadata["shortest_path"] == ["ALICE", "BOB", "CAROL"]


def test_hirag_only_need_context_and_source_provenance(tmp_path):
    context, _ = asyncio.run(_build_runtime(tmp_path))
    engine = HiRAGFullQueryEngine(prompts=TEST_PROMPTS)

    response = asyncio.run(
        engine.query(
            QueryRequest(
                text="Alice",
                top_k=2,
                options={"only_need_context": True, "top_m": 2},
                trace=True,
            ),
            context,
        )
    )

    assert response.answer == response.results[0].text
    assert response.metadata["answer_generation"] == "context_only"
    assert "doc_1" in response.results[0].source["document_ids"]
    assert "chunk_1" in response.results[0].source["chunk_ids"]
    assert response.results[0].metadata["community_ids"]
    assert response.citations


def test_hirag_generate_answer(tmp_path):
    context, language = asyncio.run(_build_runtime(tmp_path))
    engine = HiRAGNobridgeQueryEngine(prompts=TEST_PROMPTS)

    response = asyncio.run(
        engine.query(
            QueryRequest(text="Alice", top_k=2, options={"generate_answer": True}),
            context,
        )
    )

    assert response.answer == "final answer"
    assert response.metadata["answer_generation"] == "generated"
    assert any(request.system_prompt and "Use context:" in request.system_prompt for request in language.requests)
