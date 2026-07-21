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
from heta_framework.common.stores import InMemoryGraphStore, LocalObjectStore  # noqa: E402
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.parsing import ParsedSource  # noqa: E402
from heta_framework.kb.steps import ExtractHiRAGGraph, ExtractHiRAGGraphConfig  # noqa: E402
from heta_framework.kb.steps.extract_hirag_graph import (  # noqa: E402
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _parse_hirag_records,
)


TEST_PROMPTS = {
    "DEFAULT_TUPLE_DELIMITER": "<|>",
    "DEFAULT_RECORD_DELIMITER": "##",
    "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",
    "META_ENTITY_TYPES": ["organization", "person", "location", "event"],
    "hi_entity_extraction": "ENTITIES {tuple_delimiter} {entity_types}\n{input_text}",
    "hi_relation_extraction": "RELATIONS {tuple_delimiter} {entities}\n{input_text}",
    "entiti_continue_extraction": "CONTINUE",
    "entiti_if_loop_extraction": "LOOP?",
    "summary_clusters": "SUMMARY {entity_description_list}",
    "summarize_entity_descriptions": "SUM {entity_name} {description_list}",
}


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
        return ModelResult(text=response, model_name=self.model_name)

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


class FakeEmbeddingModel:
    @property
    def model_name(self):
        return "fake-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        vectors = []
        for index, text in enumerate(request.texts):
            vectors.append([float(index + 1), float(len(text.split()) + 1)])
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]


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
        "text": "Alice collaborates with Bob at Acme.",
        "token_start": 0,
        "token_end": 8,
        "parent_chunk_ids": (),
    }
    values.update(overrides)
    return ParsedChunk(**values)


def _config(**overrides):
    values = {
        "prompts": TEST_PROMPTS,
        "clustering_backend": "deterministic",
        "hierarchical_layers": 2,
        "hierarchical_sparsity_threshold": 0.99,
        "hierarchical_sparsity_change_rate": 0.0,
    }
    values.update(overrides)
    return ExtractHiRAGGraphConfig(**values)


def test_tuple_entity_parse():
    entity = _handle_single_entity_extraction(
        ['"entity"', "alice", "person", "Alice is a researcher."],
        "chunk_1",
    )

    assert entity["entity_name"] == "ALICE"
    assert entity["entity_type"] == "PERSON"
    assert entity["source_id"] == "chunk_1"
    assert entity["is_summary"] is False


def test_tuple_relation_parse():
    relation = _handle_single_relationship_extraction(
        ['"relationship"', "alice", "bob", "Alice works with Bob.", "3.5"],
        "chunk_1",
    )

    assert relation["src_id"] == "ALICE"
    assert relation["tgt_id"] == "BOB"
    assert relation["weight"] == 3.5
    assert relation["source_ids"] == ["chunk_1"]


async def _run_step(tmp_path, model, *, config=None):
    object_store = LocalObjectStore(tmp_path)
    graph_store = InMemoryGraphStore()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.graph": graph_store,
            "models.language": model,
            "models.embedding": FakeEmbeddingModel(),
        }
    )
    chunk = _chunk()
    await object_store.put("chunks/chunk_1.json", chunk.to_json_bytes())
    context.set_artifact("chunk_keys", ("chunks/chunk_1.json",))
    await ExtractHiRAGGraph(config or _config()).run(context)
    return context, object_store, graph_store


def test_hirag_gleaning_flow_and_relation_entity_list(tmp_path):
    model = FakeLanguageModel(
        [
            '("entity"<|>"alice"<|>"person"<|>"Alice appears.")<|COMPLETE|>',
            "yes",
            '("entity"<|>"bob"<|>"person"<|>"Bob appears.")<|COMPLETE|>',
            '("relationship"<|>"alice"<|>"bob"<|>"Alice works with Bob."<|>"1.0")<|COMPLETE|>',
            "no",
        ]
    )

    context, _, graph_store = asyncio.run(
        _run_step(tmp_path, model, config=_config(hierarchical_layers=0))
    )

    result = context.artifacts["extract_hi_rag_graph_result"]
    assert result.base_entity_count == 2
    assert result.base_relation_count == 1
    assert "ALICE" in graph_store.nodes
    assert "BOB" in graph_store.nodes
    assert "ALICE--RELATED--BOB" in graph_store.edges
    relation_prompt = model.requests[3].prompt
    assert "ALICE,BOB" in relation_prompt
    entity_trace = context.artifacts["hi_rag_extraction_trace"][0]
    assert entity_trace["gleaning_count"] == 1


def test_deterministic_clustering_and_cluster_summary_parse(tmp_path):
    model = FakeLanguageModel(
        [
            "##".join(
                [
                    '("entity"<|>"alice"<|>"person"<|>"Alice appears.")',
                    '("entity"<|>"bob"<|>"person"<|>"Bob appears.")',
                    '("entity"<|>"acme"<|>"organization"<|>"Acme appears.")',
                    "<|COMPLETE|>",
                ]
            ),
            "no",
            '("relationship"<|>"alice"<|>"bob"<|>"Alice works with Bob."<|>"1.0")<|COMPLETE|>',
            "no",
            '("entity"<|>"collaboration"<|>"event"<|>"Alice and Bob collaborate.")##'
            '("relationship"<|>"collaboration"<|>"alice"<|>"Collaboration involves Alice."<|>"1.0")<|COMPLETE|>',
        ]
    )

    context, _, graph_store = asyncio.run(_run_step(tmp_path, model))

    assert "COLLABORATION" in graph_store.nodes
    summary_entities = context.artifacts["hi_rag_summary_entities"]
    assert summary_entities[0]["is_summary"] is True
    assert summary_entities[0]["layer"] == 1
    assert summary_entities[0]["parent_entity_ids"] == ["ALICE", "BOB"]
    layers = context.artifacts["hi_rag_hierarchical_layers"]
    assert layers[0]["clusters"][0]["entity_ids"] == ["ALICE", "BOB"]
    cluster_trace = [
        item
        for item in context.artifacts["hi_rag_extraction_trace"]
        if item["stage"] == "hierarchical_clustering"
    ][0]
    assert cluster_trace["backend"] == "deterministic"


def test_parser_skips_noise_and_merged_source_ids(tmp_path):
    parsed = _parse_hirag_records(
        """Here is output:
        ("entity"<|>"alice"<|>"person"<|>"Alice in first chunk.")##
        not a tuple
        <|COMPLETE|>
        """,
        "chunk_a",
        _config(),
        layer=0,
    )
    assert list(parsed.nodes) == ["ALICE"]
    assert parsed.trace["skipped_records"]

    model = FakeLanguageModel(
        [
            '("entity"<|>"alice"<|>"person"<|>"Alice first.")<|COMPLETE|>',
            "no",
            "",
        ]
    )

    async def run():
        object_store = LocalObjectStore(tmp_path)
        graph_store = InMemoryGraphStore()
        context = FakeContext(
            {
                "stores.objects": object_store,
                "stores.graph": graph_store,
                "models.language": model,
                "models.embedding": FakeEmbeddingModel(),
            }
        )
        await object_store.put("chunks/chunk_a.json", _chunk(chunk_id="chunk_a").to_json_bytes())
        await object_store.put(
            "chunks/chunk_b.json",
            _chunk(chunk_id="chunk_b", text="Alice again.").to_json_bytes(),
        )
        context.set_artifact("chunk_keys", ("chunks/chunk_a.json", "chunks/chunk_b.json"))
        model.responses[:] = [
            '("entity"<|>"alice"<|>"person"<|>"Alice first.")<|COMPLETE|>',
            "no",
            '("entity"<|>"alice"<|>"person"<|>"Alice second.")<|COMPLETE|>',
            "no",
            "<|COMPLETE|>",
            "no",
            "<|COMPLETE|>",
            "no",
        ]
        await ExtractHiRAGGraph(_config(hierarchical_layers=0)).run(context)
        return graph_store.nodes["ALICE"].properties["source_ids"]

    assert asyncio.run(run()) == ["chunk_a", "chunk_b"]
