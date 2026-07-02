# Deduplicate Entities

`DeduplicateEntities` merges duplicate entity records inside the current build batch.

```text
ExtractedEntity JSON -> deduplicated ExtractedEntity JSON
```

It does not merge against historical graph storage. Incremental global merge is handled by `MergeGraphIntoStore`.

## Contract

`DeduplicateEntities` uses:

```text
stores.objects
models.language
models.embedding
```

The embedding model is used when semantic merge is enabled.

Default input artifact:

```text
entity_keys
```

Execution flow:

```text
load entity records
  -> exact group by normalized name
  -> optionally find semantically similar groups
  -> ask LLM for merge mapping
  -> write deduplicated_entities/{entity_id}.json
  -> expose deduplicated_entity_keys and entity_id_mapping
```

## HetaDB Alignment

The step follows HetaDB's batch deduplication idea:

- first merge same-name entities
- then use embedding similarity to find possible duplicates
- use LLM output and a mapping table to decide merges
- keep original records when LLM output is invalid
- record recoverable problems as issues

## Artifacts

```text
deduplicate_entities_result
deduplicated_entity_keys
entity_id_mapping
```

`entity_id_mapping` maps original entity IDs to their canonical entity IDs. `DeduplicateRelations` uses it to retarget relation endpoints.

## Output

The output records still use the `ExtractedEntity` structure. This means `BuildGraph` can consume either raw `entity_keys` or `deduplicated_entity_keys`.

## Failure Handling

If the LLM returns malformed merge output, the step keeps the original records and records a `StepIssue`. This protects pipeline continuity without hiding quality problems.
