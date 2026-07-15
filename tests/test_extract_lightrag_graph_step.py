import asyncio
import json
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import ModelChunk, ModelRequest, ModelResult  # noqa: E402
from heta_framework.common.stores import InMemoryGraphStore, LocalObjectStore  # noqa: E402
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.parsing import ParsedSource  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    ExtractLightRAGGraph,
    ExtractLightRAGGraphConfig,
)


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
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            return ModelResult(text=response, model_name=self.model_name)
        return ModelResult(
            text=json.dumps(response, ensure_ascii=False),
            parsed=response,
            model_name=self.model_name,
        )

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


def _source() -> ParsedSource:
    return ParsedSource(
        key="raw/alice.txt",
        name="alice.txt",
        file_type="txt",
        content_sha256="a" * 64,
    )


def _chunk(**overrides) -> ParsedChunk:
    values = {
        "chunk_id": "chunk_1",
        "document_id": "doc_1",
        "source": _source(),
        "page_index": 0,
        "chunk_index": 0,
        "text": "Alice collaborates with Bob.",
        "token_start": 0,
        "token_end": 6,
        "parent_chunk_ids": (),
    }
    values.update(overrides)
    return ParsedChunk(**values)


def _context(tmp_path, model):
    object_store = LocalObjectStore(tmp_path)
    graph_store = InMemoryGraphStore()
    return (
        FakeContext(
            {
                "stores.objects": object_store,
                "stores.graph": graph_store,
                "models.language": model,
            }
        ),
        object_store,
        graph_store,
    )


async def _run_step(tmp_path, model, *, config=None, chunk=None):
    context, object_store, graph_store = _context(tmp_path, model)
    chunk = chunk or _chunk()
    await object_store.put("chunks/chunk_1.json", chunk.to_json_bytes())
    context.set_artifact("chunk_keys", ("chunks/chunk_1.json",))
    await ExtractLightRAGGraph(config).run(context)
    return context, object_store, graph_store


def test_lightrag_json_extraction_success(tmp_path):
    model = FakeLanguageModel(
        [
            {
                "entities": [
                    {
                        "name": "Alice",
                        "type": "Person",
                        "description": "Alice is a collaborator.",
                    },
                    {
                        "name": "Bob",
                        "type": "Person",
                        "description": "Bob works with Alice.",
                    },
                ],
                "relationships": [
                    {
                        "source": "Alice",
                        "target": "Bob",
                        "keywords": "collaboration",
                        "description": "Alice collaborates with Bob.",
                    }
                ],
            }
        ]
    )

    context, object_store, graph_store = asyncio.run(_run_step(tmp_path, model))

    result = context.artifacts["extract_light_rag_graph_result"]
    assert result.entity_count == 2
    assert result.relation_count == 1
    assert result.failed_chunk_ids == ()
    assert len(graph_store.nodes) == 2
    assert len(graph_store.edges) == 1
    alice = graph_store.nodes["Alice"]
    assert alice.properties["entity_type"] == "person"
    assert alice.properties["file_path"] == "alice.txt"
    edge = graph_store.edges["Alice--RELATED--Bob"]
    assert edge.properties["keywords"] == "collaboration"
    assert edge.properties["weight"] == 1.0
    node_payload = json.loads(
        (asyncio.run(object_store.get(context.artifacts["light_rag_graph_node_keys"][0]))).decode(
            "utf-8"
        )
    )
    assert node_payload["properties"]["source_id"] == "chunk_1"


def test_lightrag_json_fenced_block(tmp_path):
    model = FakeLanguageModel(
        [
            """```json
{"entities":[{"name":"Alice","type":"Person","description":"Alice appears."}],"relationships":[]}
```"""
        ]
    )

    context, _, graph_store = asyncio.run(_run_step(tmp_path, model))

    assert context.artifacts["extract_light_rag_graph_result"].entity_count == 1
    assert "Alice" in graph_store.nodes


def test_lightrag_json_empty_arrays(tmp_path):
    model = FakeLanguageModel([{"entities": [], "relationships": []}])

    context, _, graph_store = asyncio.run(_run_step(tmp_path, model))

    result = context.artifacts["extract_light_rag_graph_result"]
    assert result.entity_count == 0
    assert result.relation_count == 0
    assert result.failed_chunk_ids == ()
    assert len(graph_store.nodes) == 0


def test_lightrag_json_malformed_marks_failed_chunk(tmp_path):
    model = FakeLanguageModel(["not json"])

    context, _, graph_store = asyncio.run(_run_step(tmp_path, model))

    result = context.artifacts["extract_light_rag_graph_result"]
    assert result.entity_count == 0
    assert result.failed_chunk_ids == ("chunk_1",)
    assert len(graph_store.nodes) == 0


def test_lightrag_tuple_extraction_success(tmp_path):
    model = FakeLanguageModel(
        [
            "\n".join(
                [
                    "entity<|#|>Alice<|#|>Person<|#|>Alice appears.",
                    "entity<|#|>Bob<|#|>Person<|#|>Bob appears.",
                    "relation<|#|>Alice<|#|>Bob<|#|>collaboration<|#|>Alice works with Bob.",
                    "<|COMPLETE|>",
                ]
            )
        ]
    )
    config = ExtractLightRAGGraphConfig(extraction_format="tuple")

    context, _, graph_store = asyncio.run(_run_step(tmp_path, model, config=config))

    assert context.artifacts["extract_light_rag_graph_result"].entity_count == 2
    edge = graph_store.edges["Alice--RELATED--Bob"]
    assert edge.properties["keywords"] == "collaboration"


def test_lightrag_gleaning_keeps_longer_description(tmp_path):
    model = FakeLanguageModel(
        [
            {
                "entities": [
                    {"name": "Alice", "type": "Person", "description": "Short."}
                ],
                "relationships": [],
            },
            {
                "entities": [
                    {
                        "name": "Alice",
                        "type": "Person",
                        "description": "Alice has a longer and more complete description.",
                    }
                ],
                "relationships": [],
            },
        ]
    )
    config = ExtractLightRAGGraphConfig(entity_extract_max_gleaning=1)

    _, _, graph_store = asyncio.run(_run_step(tmp_path, model, config=config))

    alice = graph_store.nodes["Alice"]
    assert alice.properties["description"] == "Alice has a longer and more complete description."
    assert len(model.requests) == 2


def test_lightrag_graph_store_and_object_artifacts_written(tmp_path):
    model = FakeLanguageModel(
        [
            {
                "entities": [
                    {"name": "Alice", "type": "Person", "description": "Alice appears."}
                ],
                "relationships": [],
            }
        ]
    )

    context, object_store, graph_store = asyncio.run(_run_step(tmp_path, model))

    node_keys = context.artifacts["light_rag_graph_node_keys"]
    assert len(node_keys) == 1
    assert context.artifacts["light_rag_entity_keys"] == node_keys
    assert len(graph_store.nodes) == 1
    payload = json.loads((asyncio.run(object_store.get(node_keys[0]))).decode("utf-8"))
    assert payload["id"] == "Alice"
    assert payload["properties"]["source_ids"] == ["chunk_1"]
