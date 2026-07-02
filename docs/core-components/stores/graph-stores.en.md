# Graph Stores

Graph Stores are Heta's property-graph storage protocol. They are intended for storing entity nodes, relation edges, and properties in future adapters such as Neo4j, NebulaGraph, or JanusGraph.

Current Heta-style `BuildGraph` writes to a PostgreSQL-style schema and does not depend on `GraphStoreProtocol`. This page documents the general property-graph extension interface, not the current Heta graph procedure storage path.

## Quick Start

```python
from heta_framework.common.stores import GraphEdge, GraphNode, InMemoryGraphStore

store = InMemoryGraphStore()

await store.upsert_nodes(
    [
        GraphNode(
            id="entity_shanghai",
            labels=("Entity", "objective_entity", "administrative_region"),
            properties={
                "name": "Shanghai",
                "description": "Shanghai is a municipality in China.",
            },
        ),
        GraphNode(
            id="entity_xuhui",
            labels=("Entity", "objective_entity", "administrative_region"),
            properties={"name": "Xuhui District"},
        ),
    ]
)

await store.upsert_edges(
    [
        GraphEdge(
            id="relation_contains",
            source_id="entity_shanghai",
            target_id="entity_xuhui",
            type="contains_administrative_region",
            properties={
                "type": "spatial_relation",
                "description": "Xuhui District is an administrative district under Shanghai.",
            },
        )
    ]
)
```

## Implementations

| Store | Use |
| --- | --- |
| `InMemoryGraphStore` | In-memory implementation for tests, examples, and small local pipelines. |
| Future adapters | Neo4j, NebulaGraph, JanusGraph, and other property graph databases can implement the same protocol. |

## Core Objects

| Object | Meaning |
| --- | --- |
| `GraphStoreProtocol` | Property-graph storage capability protocol for future graph steps and custom stores. |
| `GraphNode` | Node record with stable id, labels, and properties. |
| `GraphEdge` | Directed edge record with stable id, source, target, relation type, and properties. |

## Protocol

```python
class GraphStoreProtocol:
    async def upsert_nodes(self, nodes: Sequence[GraphNode]) -> None: ...
    async def upsert_edges(self, edges: Sequence[GraphEdge]) -> None: ...
    async def delete_nodes(self, node_ids: Sequence[str]) -> None: ...
    async def delete_edges(self, edge_ids: Sequence[str]) -> None: ...
    async def get_node(self, node_id: str) -> GraphNode | None: ...
    async def get_edge(self, edge_id: str) -> GraphEdge | None: ...
    async def count_nodes(self) -> int: ...
    async def count_edges(self) -> int: ...
    async def aclose(self) -> None: ...
```

Custom graph stores only need to implement these methods.

## Nodes

```python
GraphNode(
    id="entity_...",
    labels=("Entity", "objective_entity", "administrative_region"),
    properties={
        "name": "Shanghai",
        "type": "objective_entity",
        "subtype": "administrative_region",
        "description": "...",
        "source_chunk_ids": ["chunk_1", "chunk_2"],
    },
)
```

`id` is the stable upsert/delete key. Heta usually uses `ExtractedEntity.entity_id` as the node id and stores the display name in `properties["name"]`.

## Edges

```python
GraphEdge(
    id="relation_...",
    source_id="entity_shanghai",
    target_id="entity_xuhui",
    type="contains_administrative_region",
    properties={
        "name": "contains_administrative_region",
        "type": "spatial_relation",
        "description": "...",
        "source_chunk_ids": ["chunk_1"],
    },
)
```

`type` is the concrete graph edge type. If a database restricts edge type names, the adapter should handle escaping or mapping internally without changing Heta relation semantics.

## Scope

Graph Stores handle node and edge upsert, delete, lookup, and counting.

They do not extract entities, extract relations, deduplicate facts, run vector recall, persist SQL tables, merge historical graph stores, or manage the `KnowledgeBase` lifecycle.
