# Deduplicate Relations

`DeduplicateRelations` merges duplicate relation records inside the current build batch.

```text
ExtractedRelation JSON -> deduplicated ExtractedRelation JSON
```

It can run after `DeduplicateEntities` so relation endpoints can be rewritten to canonical entity IDs.

## Contract

`DeduplicateRelations` uses:

```text
stores.objects
models.language
models.embedding
```

Default input artifacts:

```text
relation_keys
entity_id_mapping
```

Execution flow:

```text
load relation records
  -> apply entity_id_mapping to endpoints
  -> exact group by source, target, type, and name
  -> optionally find semantically similar groups
  -> ask LLM for merge mapping
  -> write deduplicated_relations/{relation_id}.json
  -> expose deduplicated_relation_keys and relation_id_mapping
```

## HetaDB Alignment

The step follows HetaDB's relation deduplication idea:

- normalize relation endpoints after entity deduplication
- group exact duplicate relations first
- use embedding similarity for broader candidates
- use LLM mapping output to decide merges
- keep original records when output is invalid

## Artifacts

```text
deduplicate_relations_result
deduplicated_relation_keys
relation_id_mapping
```

The output records still use `ExtractedRelation`, so `BuildGraph` can consume either raw relations or deduplicated relations.

## Boundary

This step only deduplicates the current build batch. It does not query existing graph tables or vector graph collections. Use `MergeGraphIntoStore` for incremental global merge.
