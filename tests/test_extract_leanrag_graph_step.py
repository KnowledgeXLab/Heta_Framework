import asyncio
import hashlib
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
from heta_framework.kb.steps import ExtractLeanRAGGraph, ExtractLeanRAGGraphConfig  # noqa: E402
from heta_framework.kb.steps.extract_hirag_graph import _parse_hirag_records  # noqa: E402
from heta_framework.kb.steps.extract_leanrag_graph import (  # noqa: E402
    adapt_hirag_entities_to_leanrag,
    adapt_hirag_relations_to_leanrag,
    compute_leanrag_hash,
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
        return ModelResult(text=self.responses.pop(0), model_name=self.model_name)

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
        return EmbeddingResult(
            vectors=[[float(index + 1), float(len(text) + 1)] for index, text in enumerate(request.texts)],
            model_name=self.model_name,
        )

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
    values = {"prompts": TEST_PROMPTS}
    values.update(overrides)
    return ExtractLeanRAGGraphConfig(**values)


def test_leanrag_chunk_hash():
    text = "Alice collaborates with Bob."
    assert compute_leanrag_hash(text) == hashlib.md5(text.encode()).hexdigest()


def test_hirag_entity_extraction_adapter():
    entities = [
        {
            "entity_name": "ALICE",
            "entity_type": "PERSON",
            "description": "Alice first.",
            "source_id": "chunk_a",
            "source_ids": ["chunk_a"],
        },
        {
            "entity_name": "ALICE",
            "entity_type": "PERSON",
            "description": "Alice second.",
            "source_id": "chunk_b",
            "source_ids": ["chunk_b"],
        },
    ]
    relations = [{"src_id": "ALICE", "tgt_id": "BOB"}]

    adapted = adapt_hirag_entities_to_leanrag(entities, relations)

    assert adapted == [
        {
            "entity_name": "ALICE",
            "entity_type": "PERSON",
            "description": "Alice first.<SEP>Alice second.",
            "source_id": "chunk_a|chunk_b",
            "source_ids": ["chunk_a", "chunk_b"],
            "degree": 1,
            "parent": "",
            "level": 0,
            "is_aggregate": False,
            "properties": {"base_graph_source": "hirag_two_stage_extraction"},
        }
    ]


def test_hirag_relation_extraction_adapter_source_merge():
    relations = [
        {
            "src_id": "ALICE",
            "tgt_id": "BOB",
            "description": "Alice works with Bob.",
            "weight": 1.5,
            "source_id": "chunk_a",
            "source_ids": ["chunk_a"],
        },
        {
            "src_id": "BOB",
            "tgt_id": "ALICE",
            "description": "Bob collaborates with Alice.",
            "weight": 2.0,
            "source_id": "chunk_b",
            "source_ids": ["chunk_b"],
        },
    ]

    adapted = adapt_hirag_relations_to_leanrag(relations)

    assert adapted[0]["src_tgt"] == "ALICE"
    assert adapted[0]["tgt_src"] == "BOB"
    assert adapted[0]["description"] == "Alice works with Bob.<SEP>Bob collaborates with Alice."
    assert adapted[0]["weight"] == 3.5
    assert adapted[0]["source_id"] == "chunk_a|chunk_b"
    assert adapted[0]["source_ids"] == ["chunk_a", "chunk_b"]


def test_hirag_tuple_parser_reuse_path():
    parsed = _parse_hirag_records(
        '("entity"<|>"alice"<|>"person"<|>"Alice appears.")##'
        '("relationship"<|>"alice"<|>"bob"<|>"Alice knows Bob."<|>"1.0")<|COMPLETE|>',
        "chunk_a",
        _config(),
        layer=0,
    )
    assert "ALICE" in parsed.nodes
    assert ("ALICE", "BOB") in parsed.edges


async def _run_step(tmp_path, model):
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
    chunk_a = _chunk(chunk_id="chunk_a", text="Alice works with Bob.")
    chunk_b = _chunk(chunk_id="chunk_b", text="Alice works with Acme.", chunk_index=1)
    await object_store.put("chunks/chunk_a.json", chunk_a.to_json_bytes())
    await object_store.put("chunks/chunk_b.json", chunk_b.to_json_bytes())
    context.set_artifact("chunk_keys", ("chunks/chunk_a.json", "chunks/chunk_b.json"))
    await ExtractLeanRAGGraph(_config()).run(context)
    return context, graph_store


def test_extract_leanrag_graph_step_outputs_adapted_artifacts(tmp_path):
    model = FakeLanguageModel(
        [
            '("entity"<|>"alice"<|>"person"<|>"Alice first.")<|COMPLETE|>',
            "no",
            '("entity"<|>"alice"<|>"person"<|>"Alice second.")<|COMPLETE|>',
            "no",
            '("relationship"<|>"alice"<|>"bob"<|>"Alice works with Bob."<|>"1.0")<|COMPLETE|>',
            "no",
            '("relationship"<|>"alice"<|>"acme"<|>"Alice works with Acme."<|>"1.0")<|COMPLETE|>',
            "no",
        ]
    )

    context, graph_store = asyncio.run(_run_step(tmp_path, model))

    result = context.artifacts["extract_lean_rag_graph_result"]
    assert result.chunk_count == 2
    assert result.base_entity_count == 1
    assert result.base_relation_count == 2
    chunks = context.artifacts["lean_rag_chunks"]
    assert chunks[0]["hash_code"] == compute_leanrag_hash("Alice works with Bob.")
    entity = context.artifacts["lean_rag_base_entities"][0]
    assert entity["entity_name"] == "ALICE"
    assert entity["source_id"] == "chunk_a|chunk_b"
    assert entity["degree"] == 2
    assert graph_store.nodes["ALICE"].properties["source_ids"] == ["chunk_a", "chunk_b"]
    relation_prompt = model.requests[5].prompt
    assert "ALICE" in relation_prompt
