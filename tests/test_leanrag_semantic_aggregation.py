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
from heta_framework.kb.steps import (  # noqa: E402
    LeanRAGSemanticAggregation,
    LeanRAGSemanticAggregationConfig,
)
from heta_framework.kb.steps.leanrag_semantic_aggregation import (  # noqa: E402
    export_all_entities_json_lines,
    pack_single_community_describe,
    parse_aggregate_response,
)


PROMPTS = {
    "aggregate_entities": "AGGREGATE\n{input_text}",
    "cluster_cluster_relation": (
        "RELATE {entity_a} {entity_b} {entity_a_description} "
        "{entity_b_description} {relation_information} {tokens}"
    ),
    "rag_response": "{context_data}",
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
        vectors = [[float(index), float(len(text) + 1)] for index, text in enumerate(request.texts)]
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]


def _entity(index: int) -> dict:
    return {
        "entity_name": f"E{index}",
        "entity_type": "PERSON",
        "description": f"Entity {index} description",
        "source_id": f"chunk_{index}",
        "source_ids": [f"chunk_{index}"],
        "degree": index,
        "parent": "",
        "level": 0,
        "is_aggregate": False,
        "documents": [f"doc_{index}"],
        "document_names": {f"doc_{index}": f"doc_{index}.txt"},
        "document_tokens": {f"doc_{index}": index + 1},
        "document_token_count": index + 1,
    }


def _aggregate_json(name: str) -> str:
    return json.dumps(
        {
            "entity_name": name,
            "entity_description": f"{name} description",
            "findings": [{"summary": f"{name} summary", "explanation": f"{name} explanation"}],
        }
    )


def _config(**overrides):
    values = {
        "prompts": PROMPTS,
        "clustering_backend": "deterministic",
        "cluster_size": 2,
    }
    values.update(overrides)
    return LeanRAGSemanticAggregationConfig(**values)


async def _run(entities, relations, responses, config=None):
    context = FakeContext(
        {
            "models.language": FakeLanguageModel(responses),
            "models.embedding": FakeEmbeddingModel(),
        }
    )
    context.set_artifact("lean_rag_base_entities", entities)
    context.set_artifact("lean_rag_base_relations", relations)
    await LeanRAGSemanticAggregation(config or _config()).run(context)
    return context


def test_deterministic_embedding_and_clustering_fallback():
    entities = [_entity(index) for index in range(10)]
    responses = [_aggregate_json(f"A{index}") for index in range(5)] + [_aggregate_json("ROOT")]

    context = asyncio.run(_run(entities, [], responses))

    trace = context.artifacts["lean_rag_semantic_aggregation_trace"][0]
    assert trace["backend"] == "deterministic"
    assert trace["embedding_shape"] == [10, 2]
    assert trace["cluster_labels"] == [[0], [0], [1], [1], [2], [2], [3], [3], [4], [4]]


def test_single_node_cluster_parent_behavior():
    entities = [_entity(index) for index in range(11)]
    responses = [_aggregate_json(f"A{index}") for index in range(5)] + [_aggregate_json("ROOT")]

    context = asyncio.run(_run(entities, [], responses))

    first_layer = context.artifacts["lean_rag_all_entities_layers"][0]
    assert first_layer[-1]["parent"] == "E10"


def test_aggregate_prompt_input_format_and_response_parse():
    describe = pack_single_community_describe(
        [_entity(1), _entity(2)],
        {("E1", "E2"): {"src_tgt": "E1", "tgt_src": "E2", "description": "related"}},
    )
    parsed = parse_aggregate_response(_aggregate_json("PAIR"))

    assert "-----Entities-----" in describe
    assert "-----Relationships-----" in describe
    assert "```csv" in describe
    assert parsed["entity_name"] == "PAIR"
    assert parsed["entity_description"] == "PAIR description"


def test_aggregate_relation_direct_evidence_path():
    entities = [_entity(index) for index in range(10)]
    relations = [
        {
            "src_tgt": "E0",
            "tgt_src": "E2",
            "description": "E0 supports E2.",
            "weight": 1,
            "source_id": "chunk_0|chunk_2",
            "source_ids": ["chunk_0", "chunk_2"],
            "level": 0,
            "is_generated": False,
        }
    ]
    responses = [_aggregate_json(f"A{index}") for index in range(5)] + [_aggregate_json("ROOT")]

    context = asyncio.run(_run(entities, relations, responses))

    generated = context.artifacts["lean_rag_generated_relations"]
    assert generated[0]["src_tgt"] == "A0"
    assert generated[0]["tgt_src"] == "A1"
    assert "relationship<|>E0<|>E2<|>E0 supports E2." in generated[0]["description"]
    assert generated[0]["level"] == 1
    assert generated[0]["weight"] == 1


def test_root_aggregate_behavior_and_export():
    entities = [_entity(index) for index in range(10)]
    responses = [_aggregate_json(f"A{index}") for index in range(5)] + [_aggregate_json("ROOT")]

    context = asyncio.run(_run(entities, [], responses))

    result = context.artifacts["lean_rag_semantic_aggregation_result"]
    all_layers = context.artifacts["lean_rag_all_entities_layers"]
    assert result.root_entity_name == "ROOT"
    assert all_layers[-1]["entity_name"] == "ROOT"
    assert all_layers[-1]["parent"] == "root"
    assert all_layers[-1]["documents"] == [f"doc_{index}" for index in range(10)]
    assert all_layers[-1]["document_names"]["doc_0"] == "doc_0.txt"
    assert all_layers[-1]["document_details"][0] == {
        "document_id": "doc_0",
        "document_name": "doc_0.txt",
        "document_token_count": 1,
    }
    assert all_layers[-1]["document_token_count"] == sum(range(1, 11))
    assert all_layers[-2][0]["parent"] == "ROOT"
    assert export_all_entities_json_lines(all_layers).splitlines()[-1].startswith('{"entity_name": "ROOT"')
