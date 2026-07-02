# Merge Graph Into Store

`MergeGraphIntoStore` incrementally merges a new graph extraction batch into an existing Heta-style graph store.

```text
current batch facts + existing graph store -> merged graph store
```

It is the dynamic knowledge-base update path. It is normally an alternative to `BuildGraph`, not something to run immediately after `BuildGraph` for the same batch.

## Contract

`MergeGraphIntoStore` uses:

```text
stores.objects
stores.sql
stores.vector
models.embedding
models.language
```

Default input artifacts:

```text
deduplicated_entity_keys
deduplicated_relation_keys
chunk_keys
```

## Entity Merge Flow

```text
load new entities
  -> embed new entity text
  -> search existing graph entity vectors
  -> load candidate SQL rows and evidence
  -> ask LLM for merge mapping
  -> delete merged old records
  -> insert merged entity records
  -> upsert merged entity vectors
  -> write merged evidence
```

If no candidate passes the similarity threshold, the new entity is inserted directly.

## Relation Merge Flow

Relations are merged after entities because relation endpoints may change during entity merge.

```text
apply entity mapping
  -> embed new relation text
  -> search existing graph relation vectors
  -> load candidate SQL rows and evidence
  -> ask LLM for merge mapping
  -> delete merged old records
  -> insert merged relation records
  -> upsert merged relation vectors
  -> write merged evidence
```

## HetaDB Alignment

The step follows the HetaDB incremental graph idea:

- batch-level deduplication happens first
- vector search recalls historical candidates
- LLM mapping decides merge or no-merge
- empty mapping means no merge
- merged old records are deleted
- merged new records are inserted
- SQL rows, vector rows, and evidence are kept in sync
- entities are processed before relations

The framework implementation keeps storage names externally configured. It does not introduce a dataset concept into the step.

## Output

`merge_graph_into_store_result` records counts for inserted, merged, deleted, and evidence rows, plus any recoverable `StepIssue` records.

After success, the same graph assets are available to:

```text
heta_graph_search
```
