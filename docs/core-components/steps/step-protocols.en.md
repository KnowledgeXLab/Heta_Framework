# Step Protocols

Build steps are composable execution units in a Heta knowledge base recipe. Each step describes one stable build action: what it needs, what it produces, and which query modes it unlocks.

The protocol answers three questions:

- which models, stores, parsers, or artifacts the step needs
- which artifacts the step writes
- whether the step makes a new query mode available

This keeps recipes explicit. A recipe describes capabilities through ordered steps instead of a set of hard-to-extend boolean switches.

```python
steps = [
    ParseDocuments(),
    SplitDocuments(),
    EmbedChunks(),
    IndexVectors(),
    ExtractEntities(),
    ExtractRelations(),
    BuildGraph(),
]
```

## Step Groups

Steps are atomic at runtime. Documentation groups them by common build goals, but these groups are not a separate execution protocol.

| Group | Steps | Result |
| --- | --- | --- |
| Document indexing | `ParseDocuments`, `SplitDocuments`, `EmbedChunks`, `IndexVectors` | Builds chunk vectors and enables `vector_search`. |
| Full-text indexing | `ParseDocuments`, `SplitDocuments`, `IndexFullText` | Writes chunks into `TextIndexStore` and enables `full_text_search`. |
| SQL text indexing | `ParseDocuments`, `SplitDocuments`, `PersistChunks` | Writes chunks into `SQLStore` and enables `sql_text_search`. |
| Heta graph build | `MergeChunks`, `RechunkDocuments`, `PersistChunks`, `ExtractEntities`, `ExtractRelations`, `DeduplicateEntities`, `DeduplicateRelations`, `BuildGraph` | Builds a Heta-style SQL/vector graph and enables `heta_graph_search`. |
| Heta graph merge | `MergeChunks`, `RechunkDocuments`, `PersistChunks`, `ExtractEntities`, `ExtractRelations`, `DeduplicateEntities`, `DeduplicateRelations`, `MergeGraphIntoStore` | Merges a new batch into an existing graph store. |

`HetaGraphProcedure` only expands a common graph build path into steps. The builder still validates and runs the expanded steps one by one.

## Design

A step is a build action, not a component container.

Models, stores, and parser registries live at the top level of `KnowledgeRecipe`. A step references them by name:

```python
ExtractRelations(model="strong")
```

This means "use the language model named `strong` from the recipe." It does not embed a model instance inside the step.

This design keeps simple recipes short, allows advanced recipes to use multiple components of the same type, and lets the builder validate dependencies before execution.

## Step Contract

Every built-in step follows `KnowledgeStepProtocol`:

```python
class KnowledgeStepProtocol(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def requirements(self) -> StepRequirements: ...

    @property
    def capabilities(self) -> StepCapabilities: ...

    async def run(self, context: StepContextProtocol) -> None: ...

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan: ...
```

| Member | Meaning |
| --- | --- |
| `name` | Stable step name for logs, run records, and summaries. |
| `requirements` | Components, artifacts, or query modes needed before running. |
| `capabilities` | Artifacts or query modes provided after success. |
| `run` | Executes the step. |
| `cleanup_plan` | Declares resources created by the step for `KnowledgeBase.delete()`. |

`cleanup_plan` only describes derived resources. Raw input objects, such as files under `raw/`, are not deleted by default.

## Component References

Steps refer to recipe components through stable component keys:

```python
from heta_framework.kb.steps import model_ref, parser_ref, store_ref

model_ref("embedding").key
# "models.embedding"

model_ref("language", "strong").key
# "models.language.strong"

store_ref("vector").key
# "stores.vector"

parser_ref().key
# "parsers.documents"
```

Unnamed references use the recipe default. Named references target a named component:

```python
model_ref("language")           # default language model
model_ref("language", "strong") # named language model
```

## Requirements

`StepRequirements` declares what must exist before a step can run:

```python
StepRequirements(
    components=frozenset({
        model_ref("embedding"),
        store_ref("vector"),
    }),
    artifacts=frozenset({"chunk_keys"}),
    queries=frozenset(),
)
```

| Field | Meaning |
| --- | --- |
| `components` | Required recipe components, such as a model, store, or parser registry. |
| `artifacts` | Required intermediate artifacts, such as `parsed_document_keys` or `chunk_keys`. |
| `queries` | Required query modes already unlocked by earlier steps. |

## Capabilities

`StepCapabilities` declares what a step adds:

```python
StepCapabilities(
    artifacts=frozenset({"chunk_embedding_keys"}),
    queries=frozenset({"vector_search"}),
)
```

Query modes are intentionally explicit:

```text
IndexVectors -> vector_search
IndexFullText -> full_text_search
BuildGraph -> heta_graph_search
```

Hybrid query modes are not inferred automatically. They normally need score fusion, reranking, rewriting, or graph expansion strategies, so they should be registered as explicit query engines.

## Issues

Recoverable problems are written into step results as `StepIssue`. They are diagnostics, not primary artifacts. Downstream steps should continue to consume clean artifacts.

```python
StepIssue(
    step="deduplicate_entities",
    subject=IssueSubject(type="dedup_group", id="Shanghai"),
    code="invalid_llm_output",
    severity="warning",
    message="LLM output is missing a non-empty description.",
    resolution=IssueResolution(
        action="kept_original_records",
        outcome="The group was not merged.",
    ),
    details={"attempt_count": "3"},
)
```

| Field | Meaning |
| --- | --- |
| `step` | Step that produced the issue. |
| `subject` | Affected object, such as a document, chunk, or dedup group. |
| `code` | Stable code for tests, filtering, and statistics. |
| `severity` | `info`, `warning`, or `error`. |
| `message` | Developer-facing explanation. |
| `resolution` | Recovery action taken by the framework. |
| `details` | Small structured details for debugging. |

Recoverable failures should record an issue and continue. Non-recoverable failures should fail the step.

## Step Context

Steps use `StepContextProtocol` to read components and artifacts:

```python
class StepContextProtocol(Protocol):
    def get_component(self, key: str) -> Any: ...
    def get_artifact(self, key: str) -> Any: ...
    def set_artifact(self, key: str, value: Any) -> None: ...
```

The builder owns dependency validation, run state, trace records, cleanup, and query capability assembly. Step code only performs the build action.
