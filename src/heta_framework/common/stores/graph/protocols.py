"""Graph store capability protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from heta_framework.common.stores.graph.types import GraphEdge, GraphNode


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Capability protocol for property graph stores."""

    async def upsert_nodes(self, nodes: Sequence[GraphNode]) -> None:
        """Insert or update graph nodes."""
        ...

    async def upsert_edges(self, edges: Sequence[GraphEdge]) -> None:
        """Insert or update directed graph edges."""
        ...

    async def delete_nodes(self, node_ids: Sequence[str]) -> None:
        """Delete graph nodes by id."""
        ...

    async def delete_edges(self, edge_ids: Sequence[str]) -> None:
        """Delete graph edges by id."""
        ...

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Return a graph node by id, if it exists."""
        ...

    async def get_edge(self, edge_id: str) -> GraphEdge | None:
        """Return a graph edge by id, if it exists."""
        ...

    async def count_nodes(self) -> int:
        """Return the number of stored nodes."""
        ...

    async def count_edges(self) -> int:
        """Return the number of stored edges."""
        ...

    async def aclose(self) -> None:
        """Release resources held by the store."""
        ...
