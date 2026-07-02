# Knowledge Procedures

`Procedure` is a reusable step composition. It does not execute work, read context, or access stores. It only expands a standard build pattern into real steps.

```text
Recipe
  -> Procedure
      -> Step
```

`Step` remains the actual execution unit. `Procedure` is static wiring.

## Protocol

```python
@runtime_checkable
class KnowledgeProcedureProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    def steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        ...
```

The procedure protocol does not repeat `requirements` or `capabilities`. The source of truth is the expanded steps:

```python
expanded_steps = procedure.steps()
```

The recipe and builder validate components, artifacts, and execution order against those real steps.

## HetaGraphProcedure

`HetaGraphProcedure` packages the Heta-style graph build path after `IndexVectors`. It covers HetaDB-style graph construction, not the base vector retrieval path.

The base vector path ends at `IndexVectors`:

```text
ParseDocuments
SplitDocuments
EmbedChunks
IndexVectors
```

To build Heta graph knowledge, continue with:

```text
MergeChunks
RechunkDocuments
PersistChunks
ExtractEntities
ExtractRelations
DeduplicateEntities
DeduplicateRelations
BuildGraph / MergeGraphIntoStore
```

`MergeChunks`, `RechunkDocuments`, and `PersistChunks` are optional preparation steps. They support graph extraction, evidence lookup, and provenance; they do not create another chunk vector retrieval collection.

## Build A New Graph

One-time graph write:

```python
from heta_framework.kb.procedures import HetaGraphProcedure

steps = [
    ParseDocuments(...),
    SplitDocuments(...),
    EmbedChunks(...),
    IndexVectors(...),
    *HetaGraphProcedure.build().steps(),
]
```

Default expansion:

```text
ExtractEntities
ExtractRelations
DeduplicateEntities
DeduplicateRelations
BuildGraph
```

To include HetaDB-style chunk merge, rechunk, and SQL chunk persistence, insert preparation steps before the procedure:

```python
steps = [
    ParseDocuments(...),
    SplitDocuments(...),
    EmbedChunks(...),
    IndexVectors(...),
    MergeChunks(...),
    RechunkDocuments(...),
    PersistChunks(...),
    *HetaGraphProcedure.build().steps(),
]
```

## Merge Into Existing Graph

Incrementally merge into an existing graph store:

```python
steps = [
    ParseDocuments(...),
    SplitDocuments(...),
    EmbedChunks(...),
    IndexVectors(...),
    *HetaGraphProcedure.merge_into_store().steps(),
]
```

Default expansion:

```text
ExtractEntities
ExtractRelations
DeduplicateEntities
DeduplicateRelations
MergeGraphIntoStore
```

Preparation steps can be inserted in the same way.

## Artifact Wiring

The procedure can configure artifact names:

```python
procedure = HetaGraphProcedure.build(
    chunk_keys_artifact="chunk_keys",
    entity_keys_artifact="entity_keys",
    relation_keys_artifact="relation_keys",
    deduplicated_entity_keys_artifact="deduplicated_entity_keys",
    deduplicated_relation_keys_artifact="deduplicated_relation_keys",
)
```

These names are written into the expanded step configs. The procedure itself does not read artifacts.

## Skip Deduplication

To skip batch-level graph deduplication:

```python
steps = [
    ...,
    *HetaGraphProcedure.build(deduplicate=False).steps(),
]
```

Expansion:

```text
ExtractEntities
ExtractRelations
BuildGraph
```

`BuildGraph` then reads:

```text
entity_keys
relation_keys
```

instead of:

```text
deduplicated_entity_keys
deduplicated_relation_keys
```

## Storage Names

Table names and vector collection names are injected from outside:

```python
from heta_framework.kb.steps import GraphTableNames, GraphVectorCollections

procedure = HetaGraphProcedure.merge_into_store(
    table_names=GraphTableNames(
        entities="papers_entities",
        relations="papers_relations",
        evidence="papers_graph_evidence",
    ),
    vector_collections=GraphVectorCollections(
        entities="papers_graph_entities",
        relations="papers_graph_relations",
    ),
)
```

`Procedure` does not introduce `dataset` and does not create a naming strategy. The application layer owns naming.

## Scope

Procedure expands steps, wires artifact names, and chooses build branches such as `build` or `merge_into_store`.

It does not execute steps, read ObjectStore, access SQL/VectorStore, validate runtime components, or manage artifact lifecycle.
