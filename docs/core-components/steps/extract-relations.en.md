# Extract Relations

`ExtractRelations` uses a language model to extract relations between entities found in the same extraction unit.

```text
ParsedChunk JSON + ExtractedEntity JSON -> ExtractedRelation JSON
```

It follows the HetaDB-style graph extraction boundary: relations are extracted from chunk-level evidence, not from arbitrary cross-document reasoning.

## Contract

`ExtractRelations` uses:

```text
stores.objects
models.language
```

Default input artifacts:

```text
chunk_keys
entity_keys
```

In graph procedures, it often consumes `rechunked_chunk_keys` and extracted entities from those chunks.

Execution flow:

```text
read chunk and entity keys
  -> group entities by chunk
  -> skip chunks with fewer than two entities
  -> ask LLM for strict JSON relations
  -> validate endpoint names against extracted entities
  -> write relations/{chunk_id}/{relation_id}.json
  -> expose relation_keys
```

## Output

Each record is an `ExtractedRelation`:

```python
ExtractedRelation(
    relation_id="relation_...",
    chunk_id="chunk_...",
    document_id="doc_...",
    source_entity_id="entity_...",
    target_entity_id="entity_...",
    source_entity_name="Shanghai",
    target_entity_name="Xuhui District",
    type="spatial_relation",
    name="contains_administrative_area",
    description="Xuhui District is an administrative district of Shanghai.",
    attributes={},
    source_chunk_ids=("chunk_...",),
)
```

Endpoint IDs are resolved from the entities available in the same chunk. If the LLM names an endpoint that cannot be matched, the relation is rejected and recorded as an issue.

## Artifacts

```text
extract_relations_result
relation_keys
```

`relation_keys` is consumed by `DeduplicateRelations` or `BuildGraph`.

## Boundary

The step does not infer relations across unrelated chunks or documents. That keeps relation evidence clear and makes graph citations traceable.
