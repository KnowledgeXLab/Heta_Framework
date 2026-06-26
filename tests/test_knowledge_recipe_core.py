import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb import (  # noqa: E402
    IssueSubject,
    KnowledgeBase,
    KnowledgeBaseAlreadyExistsError,
    KnowledgeBaseBuilder,
    KnowledgeBaseBuilderConfig,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseNotReadyError,
    KnowledgeModels,
    KnowledgeRecipe,
    KnowledgeStores,
    MissingComponentError,
    RecipeValidationError,
    StepCleanupPlan,
    StepCapabilities,
    StepIssue,
    StepRequirements,
    model_ref,
    store_ref,
)


class FakeStep:
    def __init__(
        self,
        name,
        *,
        requirements=None,
        capabilities=None,
        output_key=None,
        output_value=None,
        fail=False,
    ):
        self._name = name
        self._requirements = requirements or StepRequirements()
        self._capabilities = capabilities or StepCapabilities()
        self.output_key = output_key
        self.output_value = output_value
        self.fail = fail
        self.run_count = 0

    @property
    def name(self):
        return self._name

    @property
    def requirements(self):
        return self._requirements

    @property
    def capabilities(self):
        return self._capabilities

    async def run(self, context):
        self.run_count += 1
        if self.fail:
            raise RuntimeError("boom")
        if self.output_key is not None:
            context.set_artifact(self.output_key, self.output_value)

    def cleanup_plan(self, artifacts):
        return StepCleanupPlan()


@dataclass(frozen=True)
class ResultWithIssues:
    issues: tuple[StepIssue, ...]


def test_component_lookup_supports_defaults_and_named_components():
    language = object()
    strong_language = object()
    vector = object()
    recipe = KnowledgeRecipe(
        models=KnowledgeModels(language=language, named={"language.strong": strong_language}),
        stores=KnowledgeStores(vector=vector),
    )

    assert recipe.get_component(model_ref("language")) is language
    assert recipe.get_component(model_ref("language", "strong")) is strong_language
    assert recipe.get_component(store_ref("vector")) is vector
    assert recipe.has_component(model_ref("embedding")) is False

    try:
        recipe.get_component(model_ref("embedding"))
    except MissingComponentError as exc:
        assert "models.embedding" in str(exc)
    else:
        raise AssertionError("expected MissingComponentError")


def test_recipe_validate_checks_ordered_artifacts_components_and_duplicates():
    step_a = FakeStep(
        "a",
        capabilities=StepCapabilities(artifacts=frozenset({"x"})),
    )
    step_b = FakeStep(
        "b",
        requirements=StepRequirements(
            components=frozenset({model_ref("language")}),
            artifacts=frozenset({"x", "missing"}),
        ),
        capabilities=StepCapabilities(artifacts=frozenset({"x"})),
    )
    recipe = KnowledgeRecipe(steps=(step_a, step_b))

    result = recipe.validate()

    assert result.valid is False
    assert {issue.code for issue in result.errors} == {
        "missing_component",
        "missing_artifact",
    }
    assert {issue.code for issue in result.warnings} == {"duplicate_artifact_output"}


def test_recipe_require_valid_raises_validation_error():
    recipe = KnowledgeRecipe(
        steps=(
            FakeStep(
                "needs_input",
                requirements=StepRequirements(artifacts=frozenset({"input"})),
            ),
        )
    )

    try:
        recipe.require_valid()
    except RecipeValidationError as exc:
        assert exc.result.valid is False
    else:
        raise AssertionError("expected RecipeValidationError")


def test_builder_records_successful_step_outputs_and_issues():
    issue = StepIssue(
        step="write",
        subject=IssueSubject(type="artifact", id="out"),
        code="non_fatal",
        message="Non-fatal issue.",
    )
    step = FakeStep(
        "write",
        requirements=StepRequirements(artifacts=frozenset({"input"})),
        capabilities=StepCapabilities(artifacts=frozenset({"out"}), queries=frozenset({"search"})),
        output_key="out",
        output_value=ResultWithIssues(issues=(issue,)),
    )
    recipe = KnowledgeRecipe(steps=(step,))

    result = asyncio.run(
        KnowledgeBaseBuilder().build(recipe, initial_artifacts={"input": "value"})
    )

    assert result.record.status == "succeeded"
    assert result.record.step_records[0].status == "succeeded"
    assert result.record.step_records[0].input_artifacts == ("input",)
    assert result.record.step_records[0].output_artifacts == ("out",)
    assert result.issues == (issue,)
    assert result.capabilities.queries == frozenset({"search"})


def test_builder_records_failure_and_stops_by_default():
    step_a = FakeStep(
        "a",
        capabilities=StepCapabilities(artifacts=frozenset({"a"})),
        output_key="a",
        output_value=1,
    )
    step_b = FakeStep(
        "b",
        requirements=StepRequirements(artifacts=frozenset({"a"})),
        fail=True,
    )
    step_c = FakeStep("c")
    recipe = KnowledgeRecipe(steps=(step_a, step_b, step_c))

    result = asyncio.run(KnowledgeBaseBuilder().build(recipe))

    assert result.record.status == "failed"
    assert [record.status for record in result.record.step_records] == ["succeeded", "failed"]
    assert step_c.run_count == 0
    assert "RuntimeError: boom" == result.record.step_records[1].error


def test_builder_resume_can_skip_previously_succeeded_steps():
    step_a = FakeStep(
        "a",
        capabilities=StepCapabilities(artifacts=frozenset({"a"})),
        output_key="a",
        output_value=1,
    )
    step_b = FakeStep(
        "b",
        requirements=StepRequirements(artifacts=frozenset({"a"})),
        capabilities=StepCapabilities(artifacts=frozenset({"b"})),
        output_key="b",
        output_value=2,
    )
    recipe = KnowledgeRecipe(steps=(step_a, step_b))
    first = asyncio.run(KnowledgeBaseBuilder().build(recipe))

    resumed = asyncio.run(
        KnowledgeBaseBuilder(
            KnowledgeBaseBuilderConfig(skip_succeeded_steps=True)
        ).build(recipe, previous_record=first.record)
    )

    assert [record.status for record in resumed.record.step_records] == ["skipped", "skipped"]
    assert step_a.run_count == 1
    assert step_b.run_count == 1
    assert resumed.artifacts["a"] == 1
    assert resumed.artifacts["b"] == 2


def test_knowledge_base_create_manifest_restore_and_resume():
    step = FakeStep(
        "write",
        capabilities=StepCapabilities(artifacts=frozenset({"out"})),
        output_key="out",
        output_value=object(),
    )
    recipe = KnowledgeRecipe(steps=(step,), metadata={"owner": "test"})

    kb = asyncio.run(
        KnowledgeBase.create(
            recipe=recipe,
            name="papers",
            description="Paper KB",
            metadata={"domain": "papers"},
        )
    )
    manifest = kb.manifest()
    restored = KnowledgeBase.restore(manifest=manifest, recipe=recipe)
    resumed = asyncio.run(restored.resume())

    assert manifest.name == "papers"
    assert manifest.recipe.metadata == {"owner": "test"}
    assert manifest.to_dict()["run_record"]["artifacts"]["out"]["manifest_note"] == (
        "runtime artifact omitted"
    )
    assert restored.name == kb.name
    assert resumed.run_record.step_records[0].status == "skipped"


def test_knowledge_base_create_persists_runtime_metadata(tmp_path):
    object_store = LocalObjectStore(tmp_path / "objects")
    step = FakeStep(
        "write",
        capabilities=StepCapabilities(artifacts=frozenset({"out"})),
        output_key="out",
        output_value=("artifact-key",),
    )
    recipe = KnowledgeRecipe(
        stores=KnowledgeStores(objects=object_store),
        steps=(step,),
    )

    kb = asyncio.run(KnowledgeBase.create(recipe=recipe, name="FAA Handbook"))

    async def read_runtime():
        latest = await object_store.get("_heta/knowledge_bases/faa_handbook/latest_run.json")
        manifest_exists = await object_store.exists(
            "_heta/knowledge_bases/faa_handbook/manifest.json"
        )
        state_exists = await object_store.exists(
            f"_heta/knowledge_bases/faa_handbook/runs/{kb.run_record.run_id}/state.json"
        )
        record_exists = await object_store.exists(
            f"_heta/knowledge_bases/faa_handbook/runs/{kb.run_record.run_id}/record.json"
        )
        return latest, manifest_exists, state_exists, record_exists

    latest, manifest_exists, state_exists, record_exists = asyncio.run(read_runtime())

    assert b'"status":"succeeded"' in latest
    assert manifest_exists is True
    assert state_exists is True
    assert record_exists is True


def test_knowledge_base_create_rejects_existing_successful_kb(tmp_path):
    object_store = LocalObjectStore(tmp_path / "objects")
    step = FakeStep(
        "write",
        capabilities=StepCapabilities(artifacts=frozenset({"out"})),
        output_key="out",
        output_value=("artifact-key",),
    )
    recipe = KnowledgeRecipe(stores=KnowledgeStores(objects=object_store), steps=(step,))

    asyncio.run(KnowledgeBase.create(recipe=recipe, name="papers"))

    try:
        asyncio.run(KnowledgeBase.create(recipe=recipe, name="papers"))
    except KnowledgeBaseAlreadyExistsError as exc:
        assert "papers" in str(exc)
    else:
        raise AssertionError("expected KnowledgeBaseAlreadyExistsError")


def test_knowledge_base_load_restores_successful_runtime_metadata(tmp_path):
    object_store = LocalObjectStore(tmp_path / "objects")
    step = FakeStep(
        "write",
        capabilities=StepCapabilities(artifacts=frozenset({"out"})),
        output_key="out",
        output_value=("artifact-key",),
    )
    recipe = KnowledgeRecipe(
        stores=KnowledgeStores(objects=object_store),
        steps=(step,),
    )

    created = asyncio.run(
        KnowledgeBase.create(
            recipe=recipe,
            name="papers",
            description="Paper KB",
            metadata={"owner": "test"},
        )
    )
    loaded = asyncio.run(KnowledgeBase.load(recipe=recipe, name="papers"))

    assert loaded.name == "papers"
    assert loaded.description == "Paper KB"
    assert loaded.metadata == {"owner": "test"}
    assert loaded.run_record.run_id == created.run_record.run_id
    assert loaded.run_record.status == "succeeded"
    assert loaded.run_record.artifacts["out"] == ["artifact-key"]
    assert loaded.run_record.step_records[0].status == "succeeded"


def test_knowledge_base_load_requires_existing_manifest(tmp_path):
    object_store = LocalObjectStore(tmp_path / "objects")
    recipe = KnowledgeRecipe(stores=KnowledgeStores(objects=object_store))

    try:
        asyncio.run(KnowledgeBase.load(recipe=recipe, name="missing"))
    except KnowledgeBaseNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected KnowledgeBaseNotFoundError")


def test_knowledge_base_load_rejects_unfinished_kb(tmp_path):
    object_store = LocalObjectStore(tmp_path / "objects")
    step = FakeStep("boom", fail=True)
    recipe = KnowledgeRecipe(stores=KnowledgeStores(objects=object_store), steps=(step,))

    failed = asyncio.run(KnowledgeBase.create(recipe=recipe, name="failed-kb"))

    assert failed.run_record.status == "failed"
    try:
        asyncio.run(KnowledgeBase.load(recipe=recipe, name="failed-kb"))
    except KnowledgeBaseNotReadyError as exc:
        assert "failed-kb" in str(exc)
        assert "status=failed" in str(exc)
    else:
        raise AssertionError("expected KnowledgeBaseNotReadyError")


def test_knowledge_base_create_resumes_unfinished_runtime_state(tmp_path):
    object_store = LocalObjectStore(tmp_path / "objects")
    step_a = FakeStep(
        "a",
        capabilities=StepCapabilities(artifacts=frozenset({"a"})),
        output_key="a",
        output_value=("a-key",),
    )
    step_b = FakeStep(
        "b",
        requirements=StepRequirements(artifacts=frozenset({"a"})),
        capabilities=StepCapabilities(artifacts=frozenset({"b"})),
        output_key="b",
        output_value=("b-key",),
        fail=True,
    )
    recipe = KnowledgeRecipe(
        stores=KnowledgeStores(objects=object_store),
        steps=(step_a, step_b),
    )

    failed = asyncio.run(KnowledgeBase.create(recipe=recipe, name="recoverable"))
    assert failed.run_record.status == "failed"
    assert step_a.run_count == 1
    assert step_b.run_count == 1

    step_b.fail = False
    recovered = asyncio.run(KnowledgeBase.create(recipe=recipe, name="recoverable"))

    assert recovered.run_record.status == "succeeded"
    assert [record.status for record in recovered.run_record.step_records] == [
        "succeeded",
        "succeeded",
    ]
    assert step_a.run_count == 1
    assert step_b.run_count == 2
    assert recovered.run_record.artifacts["a"] == ["a-key"]
    assert recovered.run_record.artifacts["b"] == ("b-key",)
