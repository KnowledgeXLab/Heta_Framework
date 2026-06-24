import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb.steps import (  # noqa: E402
    KnowledgeStepProtocol,
    StepCleanupPlan,
    StepCapabilities,
    StepRequirements,
    model_ref,
    parser_ref,
    store_ref,
)


class FakeContext:
    def __init__(self):
        self.artifacts = {"chunks": ["chunk-1"]}
        self.components = {
            "models.embedding": object(),
            "stores.vector": object(),
        }

    def get_component(self, key):
        return self.components[key]

    def get_artifact(self, key):
        return self.artifacts[key]

    def set_artifact(self, key, value):
        self.artifacts[key] = value


class FakeEmbedStep:
    @property
    def name(self):
        return "embed_text"

    @property
    def requirements(self):
        return StepRequirements(
            components=frozenset({model_ref("embedding")}),
            artifacts=frozenset({"chunks"}),
        )

    @property
    def capabilities(self):
        return StepCapabilities(artifacts=frozenset({"embeddings"}))

    async def run(self, context):
        context.get_component("models.embedding")
        chunks = context.get_artifact("chunks")
        context.set_artifact("embeddings", [f"embedding:{chunk}" for chunk in chunks])

    def cleanup_plan(self, artifacts):
        return StepCleanupPlan()


def test_component_refs_have_stable_keys():
    assert model_ref("embedding").key == "models.embedding"
    assert model_ref("language", "strong").key == "models.language.strong"
    assert store_ref("vector").key == "stores.vector"
    assert store_ref("graph", "private").key == "stores.graph.private"
    assert parser_ref().key == "parsers.documents"
    assert parser_ref("strict").key == "parsers.documents.strict"


def test_step_requirements_normalize_names():
    requirements = StepRequirements(
        components={model_ref("embedding")},
        artifacts={" chunk_keys ", "parse_documents_result"},
        queries={" vector_search "},
    )

    assert requirements.components == frozenset({model_ref("embedding")})
    assert requirements.artifacts == frozenset({"chunk_keys", "parse_documents_result"})
    assert requirements.queries == frozenset({"vector_search"})


def test_step_capabilities_normalize_names():
    capabilities = StepCapabilities(
        artifacts={" embeddings "},
        queries={" vector_search "},
    )

    assert capabilities.artifacts == frozenset({"embeddings"})
    assert capabilities.queries == frozenset({"vector_search"})


def test_step_protocol_accepts_structural_step():
    step = FakeEmbedStep()
    context = FakeContext()

    assert isinstance(step, KnowledgeStepProtocol)

    async def run():
        await step.run(context)

    asyncio.run(run())

    assert context.artifacts["embeddings"] == ["embedding:chunk-1"]


def test_component_refs_reject_empty_names():
    with pytest.raises(ValueError, match="kind"):
        model_ref("")

    with pytest.raises(ValueError, match="name"):
        model_ref("language", " ")


def test_requirements_reject_empty_artifact_names():
    with pytest.raises(ValueError, match="artifacts"):
        StepRequirements(artifacts={"chunk_keys", " "})


def test_capabilities_reject_empty_query_names():
    with pytest.raises(ValueError, match="queries"):
        StepCapabilities(queries={"vector_search", ""})
