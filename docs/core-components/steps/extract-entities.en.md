# Extract Entities

`ExtractEntities` uses a language model to extract entity facts from chunks.

```text
ParsedChunk JSON -> ExtractedEntity JSON
```

It only extracts entities. Relation extraction, deduplication, and graph persistence are separate steps.

## Contract

`ExtractEntities` uses:

```text
stores.objects
models.language
```

Default input artifact:

```text
chunk_keys
```

In graph procedures, it commonly consumes:

```text
rechunked_chunk_keys
```

Execution flow:

```text
read chunk keys
  -> load ParsedChunk JSON
  -> ask LLM for strict JSON entities
  -> validate and normalize fields
  -> write entities/{chunk_id}/{entity_id}.json
  -> expose entity_keys
```

## Output

Each record is an `ExtractedEntity`:

```python
ExtractedEntity(
    entity_id="entity_...",
    chunk_id="chunk_...",
    document_id="doc_...",
    name="Shanghai",
    type="objective_entity",
    subtype="administrative_division",
    description="Shanghai is a municipality in China.",
    attributes={"country": "China"},
    source_chunk_ids=("chunk_...",),
)
```

The LLM produces semantic fields such as name, type, subtype, description, and attributes. The framework adds IDs, source chunk links, and validation.

## Artifacts

```text
extract_entities_result
entity_keys
```

`entity_keys` is consumed by `ExtractRelations`, `DeduplicateEntities`, or `BuildGraph`.

## Failure Handling

LLM output is validated. Recoverable invalid output is recorded as `StepIssue` and the step continues when possible. A chunk with no valid entities can still be a valid result.

This behavior keeps long RAG builds from failing because a single chunk produces weak extraction output.
