"""In-memory graph store implementation."""

from __future__ import annotations

from collections.abc import Sequence

from heta_framework.common.stores.graph.types import GraphEdge, GraphNode


class InMemoryGraphStore:
    """Simple in-memory graph store for tests, demos, and local pipelines."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}

    async def upsert_nodes(self, nodes: Sequence[GraphNode]) -> None:
        """Insert or update graph nodes."""
        for node in nodes:
            self.nodes[node.id] = node

    async def upsert_edges(self, edges: Sequence[GraphEdge]) -> None:
        """Insert or update directed graph edges."""
        for edge in edges:
            self.edges[edge.id] = edge

    async def delete_nodes(self, node_ids: Sequence[str]) -> None:
        """Delete graph nodes and their incident edges by id."""
        deleted = set(node_ids)
        for node_id in deleted:
            self.nodes.pop(node_id, None)
        self.edges = {
            edge_id: edge
            for edge_id, edge in self.edges.items()
            if edge.source_id not in deleted and edge.target_id not in deleted
        }

    async def delete_edges(self, edge_ids: Sequence[str]) -> None:
        """Delete graph edges by id."""
        for edge_id in edge_ids:
            self.edges.pop(edge_id, None)

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Return a graph node by id, if it exists."""
        return self.nodes.get(node_id)

    async def get_edge(self, edge_id: str) -> GraphEdge | None:
        """Return a graph edge by id, if it exists."""
        return self.edges.get(edge_id)

    async def count_nodes(self) -> int:
        """Return the number of stored nodes."""
        return len(self.nodes)

    async def count_edges(self) -> int:
        """Return the number of stored edges."""
        return len(self.edges)

    async def aclose(self) -> None:
        """Release resources held by the store."""
