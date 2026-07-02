# Build Graph

`BuildGraph` writes extracted entities, relations, and evidence into Heta-style graph storage.

```text
entities + relations + chunks -> SQL graph tables + graph vector index
```

After this step succeeds, the KB can use:

```text
heta_graph_search
```

## Contract

`BuildGraph` uses:

```text
stores.objects
stores.sql
stores.vector
models.embedding
```

Default input artifacts:

```text
deduplicated_entity_keys
deduplicated_relation_keys
chunk_keys
```

It can also be configured to consume raw `entity_keys` and `relation_keys` if a recipe intentionally skips deduplication.

Execution flow:

```text
load entities, relations, and chunk evidence
  -> create graph SQL tables if needed
  -> upsert entity rows
  -> upsert relation rows
  -> upsert evidence rows
  -> embed graph fact text
  -> upsert graph entity and relation vectors
  -> declare graph assets
  -> unlock heta_graph_search
```

## SQL Tables

`BuildGraph` stores graph facts in SQL:

```text
entities
relations
graph_evidence
```

Entity rows contain:

```text
entity_id
entity_name
entity_type
entity_subtype
description
attributes
```

Relation rows contain:

```text
relation_id
source_entity_id
target_entity_id
source_entity_name
target_entity_name
relation_type
relation_name
description
attributes
```

Evidence rows connect graph facts back to source chunks:

```text
fact_id
fact_type
chunk_id
document_id
source_key
source_name
metadata
```

## Vector Collections

The step also writes graph facts into vector collections so `heta_graph_search` can recall relevant entities and relations.

```text
graph_entities
graph_relations
```

These are separate from the chunk vector index created by `IndexVectors`.

## Capabilities

`BuildGraph` declares graph SQL tables and graph vector collections as searchable assets. The query layer uses those assets to enable `heta_graph_search`.

## BuildGraph vs MergeGraphIntoStore

Use `BuildGraph` when writing the current batch as the graph state.

Use `MergeGraphIntoStore` when the graph already exists and a new batch needs to be merged into historical facts with vector recall and LLM merge decisions.
