# Heta Graph Search

`heta_graph_search` retrieves Heta-style graph facts written by `BuildGraph` or `MergeGraphIntoStore`.

It follows HetaDB graph retrieval semantics: first recall entities and relations from graph vector collections, then hydrate structured facts and evidence from SQL graph tables.

## Required Assets

`BuildGraph` and `MergeGraphIntoStore` declare two assets:

```python
SearchAsset(
    kind="graph_tables",
    name="entities",
    store="stores.sql",
    metadata={
        "entities_table": "entities",
        "relations_table": "relations",
        "evidence_table": "graph_evidence",
    },
)

SearchAsset(
    kind="graph_vector_index",
    name="graph_entities",
    store="stores.vector",
    metadata={
        "entity_collection": "graph_entities",
        "relation_collection": "graph_relations",
    },
)
```

When both assets exist in the latest run record, the default query registry enables:

```text
heta_graph_search
```

## Retrieval Flow

```text
query text
  -> models.embedding.embed()
  -> search graph entity vectors
  -> search graph relation vectors
  -> hydrate facts from SQL graph tables
  -> attach evidence from graph_evidence
  -> QueryResponse
```

Additional graph expansion:

```text
entity hit
  -> add matched entity
  -> add one-hop relations where source_entity_name or target_entity_name matches

relation hit
  -> add matched relation
  -> add source / target endpoint entities
```

This returns local graph context instead of isolated vector hits.

## Usage

```python
response = await kb.query(
    "What is the relationship between Shanghai and Xuhui District?",
    mode="heta_graph_search",
    top_k=8,
    options={"evidence_top_k": 3},
)
```

Each `QueryResult` represents a graph fact:

```text
kind = "entity" | "relation"
id
text
score
source
metadata
```

`metadata["matched_by"]` identifies how the fact was found:

```text
entity_vector
entity_one_hop
relation_vector
relation_endpoint
```

`metadata["evidence"]` contains related chunk sources.

## Scope

`heta_graph_search` only retrieves graph facts and local graph context.

It does not generate answers, rerank, fuse BM25, or perform multi-hop reasoning. Higher-level hybrid / rewrite / rerank / multi-hop query modes compose those capabilities.
