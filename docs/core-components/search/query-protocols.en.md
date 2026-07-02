# Query Protocols

The query module connects two sides of the system:

```text
searchable assets produced by build steps
query engines that consume those assets
```

Users query through `KnowledgeBase.query(...)`. Internally, Heta uses `SearchAsset` and `QueryEngine` to keep retrieval modes extensible.

## How It Fits Together

```text
Step
  -> declares SearchAsset

QueryEngine
  -> declares required SearchAssetRef

KnowledgeBase
  -> checks current run assets
  -> enables matching query modes
```

A query mode is available only when the current KB actually built the assets it needs.

## SearchAsset

`SearchAsset` describes a searchable asset built by a step:

```python
SearchAsset(
    kind="chunk_vector_index",
    name="chunks",
    store="stores.vector",
    metadata={"collection": "chunks"},
)
```

Common asset kinds:

```text
chunk_vector_index
chunk_text_index
chunk_full_text_index
graph_vector_index
graph_tables
```

Assets describe location and structure. They do not execute queries.

## SearchAssetRef

`SearchAssetRef` is a query engine dependency:

```python
SearchAssetRef(kind="chunk_vector_index")
SearchAssetRef(kind="graph_tables", name="default")
```

`name=None` means any asset of that kind can satisfy the dependency. A specific `name` requires the same name.

## StepCapabilities

Steps declare what they provide through `StepCapabilities`:

```python
StepCapabilities(
    artifacts=frozenset({"index_vectors_result"}),
    queries=frozenset({"vector_search"}),
    search_assets=(
        SearchAsset(
            kind="chunk_vector_index",
            name="chunks",
            store="stores.vector",
            metadata={"collection": "chunks"},
        ),
    ),
)
```

| Field | Meaning |
| --- | --- |
| `artifacts` | Intermediate build artifacts. |
| `queries` | User-visible query capabilities. |
| `search_assets` | Storage assets query engines depend on. |

## QueryEngine

The first query engine protocol requires `required_assets`. Engines that need runtime components can also declare `required_components`.

```python
class QueryEngineProtocol(Protocol):
    @property
    def mode(self) -> str:
        ...

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        ...

    # Optional:
    # required_components: frozenset[ComponentRef]

    async def query(
        self,
        request: QueryRequest,
        context: QueryContext,
    ) -> QueryResponse:
        ...
```

`heta_rewrite_search` and `heta_multihop_search` require `models.language`. `heta_rerank_search` treats `models.reranker` as an optional enhancement, so it is not a hard dependency.

## QueryRequest And QueryResponse

`QueryRequest` carries:

```text
text
mode
top_k
filters
options
trace
```

`options` holds mode-specific parameters such as rerank settings, RRF settings, graph depth, or max hops.

`QueryResponse` supports:

```text
results
answer
citations
trace
metadata
```

This single response shape can represent vector search, keyword search, graph search, hybrid rerank, multi-hop answers, and citations.

## Registry

`QueryEngineRegistry` manages query engines:

```python
registry = QueryEngineRegistry([VectorSearchEngine()])
available = registry.available_modes(kb_assets)
```

`KnowledgeBase.available_queries` checks both assets and components:

```text
engine.required_assets
  vs
KB run_record.capabilities.search_assets

engine.required_components
  vs
recipe.models / recipe.stores / recipe.parsers
```

`KnowledgeBase.available_queries` and `KnowledgeBase.query()` both use this registry.
